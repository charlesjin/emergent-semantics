import os
import math
import argparse
from tqdm import tqdm

import torch
from torch.nn import functional as F

from transformers import AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
host = torch.device("cpu")

from data import karel
from utils import model as model_utils
from lm import utils as syn_utils
from probe.dataset import SemanticKarelDataset
from probe.alt import semantics_transformer
from utils.cache import CACHE
from utils.config import Config
import warnings

warnings.simplefilter("ignore", UserWarning)

TOTAL_EXAMPLES = 10


class Loader:
    def __init__(self, *datasets, batch_size=None, max_samples=None, resume_from=None):
        self.datasets = datasets
        self.batch_size = batch_size if batch_size else 1
        self.max_samples = max_samples
        self.resume_from = resume_from

    def __iter__(self):
        outputs = [[] for _ in range(len(self.datasets))]
        num_samples = 0
        for datas in zip(*self.datasets):
            [o.append(d) for d, o in zip(datas, outputs)]
            num_samples += 1
            if self.max_samples and num_samples >= self.max_samples:
                break
            if len(outputs[0]) == self.batch_size or num_samples == self.resume_from:
                yield tuple(outputs)
                outputs = [[] for _ in range(len(self.datasets))]
        if outputs[0]:
            yield tuple(outputs)

    def __len__(self):
        ds_len = min(len(ds) for ds in self.datasets)
        if self.max_samples:
            ds_len = min(self.max_samples, ds_len)
        return math.ceil(ds_len / self.batch_size)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval a finetuned transformers model on a causal language modeling task."
    )
    Config.add_train_args(parser)
    Config.add_eval_args(parser)

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--make_semantic_dataset",
        action="store_true",
        help="Whether to save the evaluation as a semantic dataset.",
    )
    parser.add_argument(
        "--generate_trace",
        action="store_true",
        help="Whether to generate traces.",
    )
    parser.add_argument(
        "--k", type=int, default=1, help="How many samples to return in decoding."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature to use in decoding.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If the evaluation should try to resume first.",
    )
    args = parser.parse_args()

    return args


def eval():
    args = parse_args()
    print()
    print(args)

    config = Config(args)

    batch_size = args.per_device_eval_batch_size
    make_semantic_dataset = args.make_semantic_dataset
    generate_trace = args.generate_trace
    max_eval_samples = args.max_eval_samples
    k = args.k
    temperature = args.temperature
    resume = args.resume

    return eval_with_config(
        config,
        batch_size,
        make_semantic_dataset,
        generate_trace,
        max_eval_samples,
        k,
        temperature,
        resume,
    )


@torch.no_grad()
def make_semantic_dataset(
    config,
    batch_size,
    max_eval_samples,
    resume,
    generate_trace,
):
    make_semantic_dataset = True
    k = 1
    temperature = None
    return eval_with_config(
        config,
        batch_size,
        make_semantic_dataset,
        generate_trace,
        max_eval_samples,
        k,
        temperature,
        resume,
    )


@torch.no_grad()
def eval_dataset(
    config,
    batch_size,
    max_eval_samples,
    resume,
):
    make_semantic_dataset = False
    generate_trace = False
    k = 1
    temperature = None
    return eval_with_config(
        config,
        batch_size,
        make_semantic_dataset,
        generate_trace,
        max_eval_samples,
        k,
        temperature,
        resume,
    )


@torch.no_grad()
def make_meta_dataset(config):
    semantic_ds_path = config.semantic_ds_path
    semantic_ds_meta_path = config.semantic_ds_meta_path

    print(f"Making meta dataset at {semantic_ds_meta_path} from {semantic_ds_path}.")

    semantic_eval_ds = CACHE.load(semantic_ds_path)
    keys = (
        "idx",
        "text",
        "tokens",
        "scores",
        "prompt_nll",
        "active",
        "results",
    )
    semantic_eval_ds_meta = []
    for d in semantic_eval_ds:
        keep = {k: d[k] for k in keys if k in d}
        if "np_trace" in d:
            keep["err"] = None in d["np_trace"]
        semantic_eval_ds_meta.append(keep)
    torch.save(semantic_eval_ds_meta, semantic_ds_meta_path)
    if CACHE.load(semantic_ds_meta_path) is not None:
        CACHE.evict(semantic_ds_meta_path)
    CACHE.insert(semantic_ds_meta_path, semantic_eval_ds_meta)


@torch.no_grad()
def eval_with_config(
    config,
    batch_size,
    make_semantic_dataset,
    generate_trace,
    max_eval_samples,
    k,
    temperature,
    resume,
):
    model, tokenizer = model_utils.load_pretrained(
        config, load_in_8bit=False, use_fast_tokenizer=True
    )

    return eval_with_config_and_model(
        config,
        model,
        tokenizer,
        batch_size,
        make_semantic_dataset,
        generate_trace,
        max_eval_samples,
        k,
        temperature,
        resume,
    )


@torch.no_grad()
def load_datasets(config):
    eval_dataset = config.eval_dataset
    if not eval_dataset:
        eval_dataset = config.dataset

    if "nocond" in config.dataset and "nocond" not in eval_dataset:
        print(
            "Warning! Trained on programs without conditionals, and testing with conditionals."
        )
    if "noloops" in config.dataset and "noloops" not in eval_dataset:
        print("Warning! Trained on programs without loops, and testing with loops.")

    raw_dataset = karel.load_karel_raw(config.split, data_folder=eval_dataset)["train"]
    state_dataset = karel.load_karel_raw(
        config.split, data_folder=eval_dataset, data_for="state"
    )["records"]

    print(f"Loaded {eval_dataset} {config.split}...")

    return raw_dataset, state_dataset


@torch.no_grad()
def get_token_id(token, tokenizer):
    token_ids = tokenizer(token)["input_ids"]
    assert len(token_ids) == 1
    return token_ids[0]


@torch.no_grad()
def eval_with_config_and_model(
    config,
    model,
    tokenizer,
    batch_size,
    make_semantic_dataset,
    generate_trace,
    max_eval_samples,
    k,
    temperature,
    resume,
    max_length=None,
    raw_dataset=None,
    state_dataset=None,
    save_results=True,
):
    forced_decoding = config.forced_decoding
    grammar_decoding = config.grammar_decoding
    num_examples = config.num_examples
    mode = config.mode

    assert torch.cuda.is_available()
    assert temperature is None and k == 1, "Greedy decoding only."
    if forced_decoding and not make_semantic_dataset:
        raise ValueError(
            "Using forced decoding but make_semantic_dataset is "
            "False... nothing to evaluate."
        )

    if generate_trace and not make_semantic_dataset:
        raise ValueError("generate_trace is True but not making semantic dataset")

    if not save_results and make_semantic_dataset:
        raise ValueError("Not saving results but also asked to make semantic dataset.")

    if max_length is None:
        if mode == "interp":
            max_length = 512
        else:
            max_length = 48

    token_to_id = lambda token: get_token_id(token, tokenizer)
    if mode == "synthesis":
        bos_token_id = token_to_id("Code:")
        eos_token_id = token_to_id("}")
        keep_eos_token = True
    else:
        assert mode == "interp"
        bos_token_id = token_to_id("Outputs:")
        eos_token_id = token_to_id("<EOS>")
        keep_eos_token = False
    pad_token_id = token_to_id("<PAD>")

    code_bos_token_id = token_to_id("Code:")
    code_eos_token_id = token_to_id("}")

    eval_num_active_examples = config.eval_num_active_examples
    if eval_num_active_examples is None:
        eval_num_active_examples = config.num_active_examples

    if raw_dataset is None or state_dataset is None:
        raw_dataset, state_dataset = load_datasets(config)

    model = model.to(device)
    model.eval()

    if save_results:
        results_path = config.results_path
        print(f"Writing results to {results_path}")

    if make_semantic_dataset:
        assert config.step is not None
        semantic_ds_dir = config.semantic_dir
        os.makedirs(semantic_ds_dir, exist_ok=True)

        semantic_ds_path = config.semantic_ds_path
        print(f"Making semantic dataset at {semantic_ds_path}")

    semantic_eval_ds = []
    all_results = []

    resume_count = 0
    cache_hit = False
    if save_results and resume:
        # Try loading previous results
        try:
            all_results = torch.load(results_path)
            if make_semantic_dataset:
                semantic_eval_ds = CACHE.load(
                    semantic_ds_path, max_length=max_eval_samples
                )
                assert semantic_eval_ds is not None, "Dataset does not exist"
                cache_hit = True
                assert len(all_results) * k == len(semantic_eval_ds)

                # check if it is done (and make sure that it is not corrupted)
                done = len(all_results) >= min(max_eval_samples, len(raw_dataset))
                if done:
                    if generate_trace:
                        # This line will throw an error if the loading doesn't work out.
                        _ = SemanticKarelDataset(
                            config,
                            tokenizer,
                            drop_last=False,
                            filter_correct=False,
                            filter_inactive=config.eval_alt_active in ["0", "1"],
                        )
                    print(
                        "ALERT: --resume passed and previous results equals "
                        "dataset size. Skipping eval."
                    )
                    return semantic_eval_ds, all_results
                # No assert because can restart from here.

        except Exception as e:
            print(f"Exception: {e}")
            semantic_eval_ds = []
            all_results = []

        resume_count = len(all_results)
        if resume_count == 0:
            print(
                "WARNING: --resume passed but either previous results were "
                "corrupted, or no previous results found."
            )
        else:
            assert resume_count <= min(len(raw_dataset), max_eval_samples)

    if not forced_decoding and config.eval_inactive_mode is not None:
        _config = config.update(alt_idx=None, eval_num_active_examples=5)
        semantic_ds_meta_path = _config.semantic_ds_meta_path
        gen_dataset = CACHE.load(semantic_ds_meta_path)
        assert len(gen_dataset) >= min(max_eval_samples, len(raw_dataset))
        use_gen = True
    else:
        gen_dataset = [None] * max_eval_samples
        use_gen = False

    correct_gen = correct = total = last_total = 0
    if resume_count > 0:
        print(f"Attempting to resume evaluation from {resume_count}.")
        correct_gen = correct = total = last_total = 0
        loader = Loader(
            raw_dataset,
            state_dataset,
            gen_dataset,
            batch_size=batch_size,
            max_samples=max_eval_samples,
            resume_from=resume_count,
        )
        pbar = tqdm(loader, total=len(loader), desc="0/0 (0)", leave=True)

        try:
            for raw_batch, _, _ in pbar:
                batch_size = len(raw_batch)
                if resume_count == total:
                    # This was the last batch evaluated previously, so start eval
                    # from next batch
                    last_total = total
                    pbar.close()
                    break
                elif resume_count > total:
                    for results_for_responses in all_results[
                        total : total + batch_size
                    ]:
                        correct += any(
                            all(results["results"][:num_examples])
                            for results in results_for_responses
                        )
                        correct_gen += any(
                            all(results["results"]) for results in results_for_responses
                        )
                    total += batch_size
                    assert resume_count >= total, "Attempt to resume failed."
        except Exception as e:
            print(f"Exception: {e}")
            print("WARNING: --resume passed but previous results were corrupted")
            resume_count = correct_gen = correct = total = last_total = 0
            semantic_eval_ds = []
            all_results = []

    # Get prompt break down
    example = raw_dataset[0]
    spec = karel.make_spec(
        example,
        num_examples=num_examples,
        include_response=False,
        use_special_tokens=True,
        active_examples=config.eval_num_active_examples,
        randomize_active=config.eval_randomize_active,
        inactive_mode=config.eval_inactive_mode,
        inactive_alt=config.eval_inactive_alt,
        alt_spec=config.eval_alt_spec,
        use_alt=config.use_alt,
        alt_active=config.eval_alt_active,
        mode=mode,
    )["spec"]
    spec_tokens = tokenizer(spec)["input_ids"]
    if mode == "synthesis":
        spec_len = len(spec_tokens)
        # spec_start_id = token_to_id("Examples:")
        ex1_start_id = token_to_id("Input0")
        ex1_end_id = token_to_id("Output0")
        spec_prefix_len = spec_tokens.index(ex1_start_id)
        spec_ex_len = spec_tokens.index(ex1_end_id) - spec_tokens.index(ex1_start_id)
        spec_postfix_len = spec_len - spec_prefix_len - spec_ex_len * num_examples * 2

        def get_prompt_nll(prompt_scores, prompt_tokens):
            assert prompt_tokens.ne(pad_token_id).all()
            num_tokens = len(prompt_scores)
            # postfix includes "Code:" which is not included in prompt scores
            prompt_end = num_tokens - spec_postfix_len + 1
            start_scores, prompt_scores, end_scores = (
                prompt_scores[:spec_prefix_len],
                prompt_scores[spec_prefix_len:prompt_end],
                prompt_scores[prompt_end:],
            )
            assert len(prompt_scores) == num_examples * spec_ex_len * 2
            prompt_scores = [
                prompt_scores[i : i + spec_ex_len]
                for i in range(0, len(prompt_scores), spec_ex_len)
            ]
            assert all([len(s) == spec_ex_len for s in prompt_scores])
            assert len(prompt_scores) == num_examples * 2
            prompt_scores.insert(0, start_scores)
            prompt_scores.append(end_scores)
            prompt_nll = [
                torch.mean(-torch.log(torch.Tensor(s))).item() for s in prompt_scores
            ]
            return prompt_nll

    else:
        # spec_len = len(spec_tokens)
        # spec_start_id = token_to_id("Examples:")

        ex1_start_id = token_to_id("Input0")
        spec_prefix_len = spec_tokens.index(ex1_start_id)

        # assert num_examples >= 2
        if num_examples >= 2:
            ex1_end_id = token_to_id("Input1")
            ex1_end = spec_tokens.index(ex1_end_id)
        else:
            ex1_end_id = token_to_id("Code:")
            ex1_end = spec_tokens.index(ex1_end_id) - 1  # Extra new line
        spec_input_len = ex1_end - spec_tokens.index(ex1_start_id)

        code_start_id = token_to_id("Code:")
        code_start = spec_tokens.index(code_start_id)
        # Extra new line => + 1
        assert code_start == spec_prefix_len + spec_input_len * num_examples + 1

        def get_prompt_nll(prompt_scores, prompt_tokens):
            prompt_scores = prompt_scores[prompt_tokens.ne(pad_token_id)]
            start_scores, prompt_scores, end_scores = (
                prompt_scores[:spec_prefix_len],
                prompt_scores[spec_prefix_len:code_start],
                prompt_scores[code_start:],
            )
            assert len(prompt_scores) == spec_input_len * num_examples
            prompt_scores = [
                prompt_scores[i : i + spec_input_len]
                for i in range(0, len(prompt_scores), spec_input_len)
            ]
            assert all([len(s) == spec_input_len for s in prompt_scores])
            assert len(prompt_scores) == num_examples
            prompt_scores.insert(0, start_scores)
            prompt_scores.append(end_scores)
            prompt_nll = [
                torch.mean(-torch.log(torch.Tensor(s))).item() for s in prompt_scores
            ]
            return prompt_nll

    # Make the lr_grammar
    if grammar_decoding:
        spec = karel.make_spec(
            example,
            num_examples=num_examples,
            include_response=True,
            use_special_tokens=True,
            active_examples=config.eval_num_active_examples,
            randomize_active=config.eval_randomize_active,
            inactive_mode=config.eval_inactive_mode,
            inactive_alt=config.eval_inactive_alt,
            alt_spec=config.eval_alt_spec,
            use_alt=config.use_alt,
            alt_active=config.eval_alt_active,
            mode=mode,
        )["spec"]
        lr_grammar = karel.make_lr_grammar(tokenizer, spec, mode=mode)
    else:
        lr_grammar = None

    loader = Loader(
        raw_dataset,
        state_dataset,
        gen_dataset,
        batch_size=batch_size,
        max_samples=max_eval_samples,
        resume_from=resume_count,
    )
    if total > 0:
        desc = f"{correct/total*100:.1f}|{correct_gen/total*100:.1f} ({total})"
    else:
        desc = "0.0|0.0 (0)"
    pbar = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        leave=True,
        disable=not save_results,
    )
    total = 0

    first = True
    for raw_batch, state_batch, gen_batch in pbar:
        batch_size = len(raw_batch)
        if resume_count:
            total += batch_size
            assert total <= resume_count
            if total == resume_count:
                resume_count = None
            continue

        spec_batch = [
            karel.make_spec(
                example,
                num_examples=num_examples,
                include_response=False,
                use_special_tokens=True,
                active_examples=config.eval_num_active_examples,
                randomize_active=config.eval_randomize_active,
                inactive_mode=config.eval_inactive_mode,
                inactive_alt=config.eval_inactive_alt,
                alt_spec=config.eval_alt_spec,
                use_alt=config.use_alt,
                alt_active=config.eval_alt_active,
                mode=mode,
            )
            for example in raw_batch
        ]

        response_batch = [example["code"].replace(";", "\n ") for example in raw_batch]

        def coerce(tokens):
            text = (
                tokenizer.decode(tokens).replace("()\n  ", "(); ").replace("; }", "\n}")
            )
            text += "}" if not text.endswith("}") else ""
            return text

        if use_gen:
            gen_batch = [coerce(example["tokens"]) for example in gen_batch]

        if config.eval_alt_spec:
            assert mode == "synthesis"
            response_batch = [
                semantics_transformer(code, config.eval_inactive_alt)
                for code in response_batch
            ]

        active_batch = [b["active"] for b in spec_batch]
        spec_batch = [b["spec"] for b in spec_batch]

        if first:
            print()
            print("=" * 20)
            print("First spec, active, code" + (", gen" if use_gen else ""))
            print("=" * 20)
            print(spec_batch[0])
            print(active_batch[0])
            print(response_batch[0])
            if use_gen:
                print(gen_batch[0])

        if forced_decoding or use_gen:
            if use_gen:
                response_batch = gen_batch
            (
                hs_batch,
                scores_batch,
                tokens_batch,
                prompt_hs_batch,
                prompt_scores_batch,
                prompt_tokens_batch,
            ) = syn_utils.force_decode(
                model,
                tokenizer,
                spec_batch,
                response_batch,
                include_prompt=True,
            )
        else:
            assert not config.eval_alt_spec
            (
                response_batch,
                tokens_batch,
                hs_batch,
                scores_batch,
            ) = syn_utils.nucleus_sample(
                model,
                tokenizer,
                spec_batch,
                temperature=temperature,
                max_length=max_length,
                n=k,
                output_hidden_states_and_scores=True,
                lr_grammar=lr_grammar,
            )

        # Add back batch dimension
        def make_batch(batch):
            return [batch[i : i + k] for i in range(batch_size)]

        hs_batch = make_batch(hs_batch)
        scores_batch = make_batch(scores_batch)
        tokens_batch = make_batch(tokens_batch)
        response_batch = make_batch(response_batch)

        if first:
            if not forced_decoding and not use_gen:
                print(response_batch[0][0])
            print("=" * 20)
            print()
            first = False

        if forced_decoding:
            prompt_hs_batch = make_batch(prompt_hs_batch)
            prompt_scores_batch = make_batch(prompt_scores_batch)
            prompt_tokens_batch = make_batch(prompt_tokens_batch)

        for idx in range(batch_size):
            state_examples = state_batch[idx]
            raw = raw_batch[idx]

            responses = response_batch[idx]
            hs_for_responses = hs_batch[idx]
            scores_for_responses = scores_batch[idx]
            tokens_for_responses = tokens_batch[idx]
            active_examples = active_batch[idx]
            if forced_decoding:
                prompt_hs_for_responses = prompt_hs_batch[idx]
                prompt_scores_for_responses = prompt_scores_batch[idx]
                prompt_tokens_for_responses = prompt_tokens_batch[idx]
            else:
                prompt_hs_for_responses = [None] * len(responses)
                prompt_scores_for_responses = [None] * len(responses)
                prompt_tokens_for_responses = [None] * len(responses)

            spec_examples = [
                (state_examples[f"input{i}"], state_examples[f"output{i}"])
                for i in range(TOTAL_EXAMPLES)
            ]
            spec_code = raw["code"]
            results_for_responses = [
                karel.semantic_eval(
                    response,
                    spec_code=spec_code,
                    spec_examples=spec_examples,
                    generate_trace=generate_trace,
                    mode=mode,
                )
                for response in responses
            ]
            all_results.append(results_for_responses)

            is_correct = 0
            is_correct_gen = 0
            for r in results_for_responses:
                r = r["results"]
                c = [not a or r for a, r in zip(active_examples, r)]
                if mode == "synthesis":
                    if all(c[:num_examples]):
                        is_correct = 1
                    if all(r):
                        is_correct_gen = 1
                else:
                    assert mode == "interp"
                    is_correct = max(is_correct, sum(c[:num_examples]) / num_examples)
                    is_correct_gen = max(is_correct_gen, sum(r) / len(r))
            correct += is_correct
            correct_gen += is_correct_gen

            # correct += any(
            #    all([not a or results["results"][:num_examples])
            #    for results in results_for_responses
            # )
            # correct_gen += any(
            #    all(results["results"]) for results in results_for_responses
            # )

            if make_semantic_dataset:
                for jdx in range(len(responses)):
                    response = responses[jdx]
                    results = results_for_responses[jdx]
                    hs = hs_for_responses[jdx]
                    scores = scores_for_responses[jdx]
                    tokens = tokens_for_responses[jdx].tolist()
                    prompt_hs = prompt_hs_for_responses[jdx]
                    prompt_scores = prompt_scores_for_responses[jdx]
                    prompt_tokens = prompt_tokens_for_responses[jdx]

                    # Strip extra text
                    if "<EOS>" in response:
                        response = response[: response.find("<EOS>") + 1]
                    elif "<PAD>" in response:
                        response = response[: response.find("<PAD>") + 1]

                    # Remove padding after EOS
                    assert len(scores) == len(tokens)
                    try:
                        eos_idx = tokens.index(eos_token_id)
                        if keep_eos_token:
                            eos_idx += 1
                    except ValueError:
                        eos_idx = None
                    response_tokens = tokens[:eos_idx]
                    response_scores = scores[:eos_idx]
                    assert len(response_scores) == len(response_tokens)

                    sem_result = {
                        "idx": total,
                        "text": response,
                        "tokens": response_tokens,
                        "scores": response_scores,
                        "active": active_examples,
                    }

                    if prompt_scores is not None:
                        sem_result["prompt_nll"] = get_prompt_nll(
                            prompt_scores, prompt_tokens
                        )

                    if generate_trace:
                        if mode == "synthesis":
                            code_tokens, code_hs = tokens, hs
                        else:
                            code_tokens, code_hs = prompt_tokens, prompt_hs

                        if config.hidden_state_layer is None:
                            code_hs = [torch.mean(h, dim=0) for h in code_hs]
                        elif config.hidden_state_layer != "full":
                            hsl = int(config.hidden_state_layer)
                            code_hs = [h[hsl] for h in code_hs]
                        # hs = hs[:eos_idx]

                        try:
                            bos_idx = code_tokens.index(code_bos_token_id)
                        except ValueError:
                            bos_idx = 0
                        try:
                            eos_idx = code_tokens.index(code_eos_token_id) + 1
                        except ValueError:
                            eos_idx = None
                        code_tokens = code_tokens[bos_idx:eos_idx]
                        code_hs = code_hs[bos_idx:eos_idx]
                        assert len(code_tokens) == len(code_hs)

                        assert len(results["np_trace"]) == TOTAL_EXAMPLES
                        sem_result.update(
                            {
                                "code_tokens": code_tokens,
                                "hidden_states": [h.to(host) for h in code_hs],
                                "np_trace": results["np_trace"],
                                "trace": results["trace"],
                                "results": results["results"],
                            }
                        )
                    semantic_eval_ds.append(sem_result)
            total += 1

        if save_results:
            if total - last_total >= 10000:
                torch.save(all_results, results_path)
                if make_semantic_dataset:
                    idx = last_total // 10000
                    assert idx >= 0
                    idx = str(idx).zfill(3)
                    semantic_ds_part = f"{semantic_ds_path}-{idx}"
                    torch.save(semantic_eval_ds[last_total:total], semantic_ds_part)
                last_total = total

        if forced_decoding:
            pbar.set_description(f"n/a accuracy: forced_decoding ({total})")
        else:
            pbar.set_description(
                f"{correct/total*100:.1f}|{correct_gen/total*100:.1f} ({total})"
            )
        torch.cuda.empty_cache()

    if save_results:
        if total > last_total:
            torch.save(all_results, results_path)
            if make_semantic_dataset:
                idx = last_total // 10000
                assert idx >= 0
                idx = str(idx).zfill(3)
                semantic_ds_part = f"{semantic_ds_path}-{idx}"
                torch.save(semantic_eval_ds[last_total:total], semantic_ds_part)

    if make_semantic_dataset:
        print(
            f"{correct} / {total} ({correct/total*100:.2f}%) of {config.split}set specs "
            f"(n={num_examples}) satisfied."
        )
        print(
            f"{correct_gen} / {total} ({correct_gen/total*100:.2f}%) of {config.split}set "
            f"totally (n={TOTAL_EXAMPLES}) satisfied."
        )

        if cache_hit:
            CACHE.evict(semantic_ds_path)
        CACHE.insert(semantic_ds_path, semantic_eval_ds)

        return all_results, semantic_eval_ds
    else:
        return all_results, correct, total


if __name__ == "__main__":
    eval()
