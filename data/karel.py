import os
import copy
import numpy as np
import random

import torch
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from datasets import load_dataset
from data.lib.karel_lib.karel import KarelWithCurlyParser
from data.utils import stdout_silencing

from probe.alt import semantics_transformer


BASE_DIR = os.path.dirname(os.path.realpath(__file__)) + "/lib/karel_lib/data"

MAX_EXAMPLES = 100
WORLD_SIZE = (8, 8)

TYPE_TO_EXT = {
    "huggingface": "jsonl",
    "state": "npz",
    "text": "txt",
}


def load_karel_raw(split, data_folder, data_for="huggingface"):
    if not data_for in TYPE_TO_EXT:
        raise ValueError(f"{data_for=} not in {TYPE_TO_EXT.keys()=}")
    ext = TYPE_TO_EXT[data_for]
    fn = f"{BASE_DIR}/{data_folder}/{split}.{ext}"
    if ext == "jsonl":
        return load_dataset("json", data_files=fn)
    elif ext == "npz":
        return np.load(fn, allow_pickle=True)
    else:
        with open(fn, "r") as f:
            return f.readlines()


PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"

DEFAULT_TOKENS = [
    "Examples:",
    "Outputs:",
    "Code:",
    "def",
    "run()",
    " {",
    "}",
    "__wall__",
    "__space__",
    "__up__",
    "__right__",
    "__left__",
    "__down__",
    # "__endline__",
    # "__ws__",
    PAD_TOKEN,
    EOS_TOKEN,
    "  ",
    " ",
    "\n",
]

ACTION_TOKENS = [
    "pick_marker()",
    "put_marker()",
    "turn_left()",
    "turn_right()",
    "move()",
]

CONDITIONAL_TOKENS = [
    "front_is_clear()",
    "left_is_clear()",
    "right_is_clear()",
    "markers_present()",
    "no_markers_present()",
]


@torch.no_grad()
def get_token_id(token, tokenizer):
    token_ids = tokenizer(token)["input_ids"]
    assert len(token_ids) == 1
    return token_ids[0]


@torch.no_grad()
def make_lr_grammar(tokenizer, sample_spec, mode="synthesis"):
    g = {
        "synthesis": _make_lr_grammar_synthesis,
        "interp": _make_lr_grammar_interp,
    }.get(mode, None)
    if g is None:
        raise ValueError("{mode=} not in grammars")
    return g(tokenizer, sample_spec)


def _make_lr_grammar_interp(tokenizer, sample_spec):
    """Make an lr grammar for interp responses.

    For interp,
    - the prompt is inputs and program
    - the response is outputs
    """
    assert "Outputs:" in sample_spec
    token_to_id = lambda token: get_token_id(token, tokenizer)
    sample_tokenized = tokenizer(sample_spec)["input_ids"]

    # Removed the BOS tokens from the full sample to get just the spec
    bos_token_ids = tokenizer("Outputs:\n")["input_ids"]
    bos_id = sample_tokenized.index(bos_token_ids[0])
    bos_len = len(bos_token_ids)
    if bos_id >= 0:
        spec_tokenized = sample_tokenized[bos_id + bos_len :]
    else:
        spec_tokenized = sample_tokenized

    # Get the number of examples
    for num_examples in range(MAX_EXAMPLES):
        tok = token_to_id(f"Output{num_examples}")
        if tok not in spec_tokenized:
            break
    assert num_examples >= 0

    # Get the number of lines
    new_line_tok = token_to_id("\n")
    num_lines = spec_tokenized.count(new_line_tok)
    if not spec_tokenized[-1] == new_line_tok:
        num_lines += 1

    tokens_in_spec = len(spec_tokenized)

    assert num_lines % num_examples == 0
    lines_per_example = num_lines // num_examples - 1

    assert tokens_in_spec % num_examples == 0
    tokens_per_example = tokens_in_spec // num_examples

    state_tokens = [
        "__wall__",
        "__space__",
        "__up__",
        "__right__",
        "__left__",
        "__down__",
    ]
    state_tokens += [f"__{i}_Marker__" for i in range(1, 10)]
    try:
        state_tokens = [token_to_id(tok) for tok in state_tokens]
    except:
        # New tokenizer adds a space before all state tokens
        state_tokens = [f" {tok}" for tok in state_tokens]
        state_tokens = [token_to_id(tok) for tok in state_tokens]

    # Check if we should add a border
    # Note that this logic assumes a fixed world size
    first_line_start = spec_tokenized.index(new_line_tok) + 1
    first_line_end = spec_tokenized.index(new_line_tok, first_line_start)
    tokens_per_line = first_line_end - first_line_start
    use_border = lines_per_example == WORLD_SIZE[0]
    # and tokens_per_line == WORLD_SIZE[1]

    # wall_tok = token_to_id(" __wall__")
    # first_line = spec_tokenized[first_line_start:first_line_end]
    # use_border = all(tok == wall_tok for tok in first_line)

    if not use_border:
        assert lines_per_example == WORLD_SIZE[0] - 2
        # assert tokens_per_line == WORLD_SIZE[1] - 2

    fixed = []
    line = 0
    tok_in_line = 0
    for tok in spec_tokenized:
        if line == 0:
            fixed.append(True)
        elif use_border and (
            line == 1
            or line == tokens_per_line
            or tok_in_line == 0
            or tok_in_line == tokens_per_line - 1
        ):
            fixed.append(True)
        elif tok in state_tokens:
            fixed.append(False)
        else:
            fixed.append(True)

        if tok == new_line_tok:
            line += 1
            tok_in_line = 0
        elif tok in state_tokens:
            tok_in_line += 1

    def lr_grammar(prefix, pad_token_id):
        num_tokens = len(prefix)
        if num_tokens >= tokens_in_spec:
            return [pad_token_id]
        if fixed[num_tokens]:
            return [spec_tokenized[num_tokens]]
        return state_tokens

    return lr_grammar


def _make_lr_grammar_synthesis(tokenizer, sample_spec):
    """Make an lr grammar for synthesis responses.

    For synthesis,
    - the prompt is inputs and outputs
    - the response is program
    """
    assert "Outputs:" not in sample_spec
    token_to_id = lambda token: get_token_id(token, tokenizer)
    sample_tokenized = tokenizer(sample_spec)["input_ids"]

    # Removed the BOS tokens from the full sample to get just the spec
    bos_token_ids = tokenizer("Code:\n")["input_ids"]
    bos_id = sample_tokenized.index(bos_token_ids[0])
    bos_len = len(bos_token_ids)
    if bos_id >= 0:
        spec_tokenized = sample_tokenized[bos_id + bos_len :]
    else:
        spec_tokenized = sample_tokenized

    # Remove the preamble
    preamble = "def run() {\n"
    preamble_tokenized = tokenizer(preamble)["input_ids"]
    preamble_id = spec_tokenized.index(preamble_tokenized[0])
    preamble_len = len(preamble_tokenized)

    # Remove the postamble
    postamble = "}"
    postamble_tokenized = tokenizer(postamble)["input_ids"]
    postamble_id = spec_tokenized.index(postamble_tokenized[0])
    postamble_len = len(postamble_tokenized)

    # Get just the body
    body_tokenized = spec_tokenized[preamble_id + preamble_len : postamble_id]

    # Now, count the number of lines of code in the body
    nw_token_id = token_to_id("\n")
    num_lines = sum([nw_token_id == token_id for token_id in body_tokenized])
    assert len(body_tokenized) % num_lines == 0
    tokens_per_action = len(body_tokenized) // num_lines
    example_action = body_tokenized[:tokens_per_action]

    # Finally, infer the grammar for the body
    action_tokens = [token_to_id(token) for token in ACTION_TOKENS]
    is_token = [ea in action_tokens for ea in example_action]
    assert sum(is_token) == 1

    # Does not check if prefix is correct.
    # Guaranteed to give a parseable output if each step is built following the grammar.
    def lr_grammar(prefix, pad_token_id):
        if len(prefix):
            if prefix[-1] == pad_token_id:
                return [pad_token_id]
            if prefix[-postamble_len:] == postamble_tokenized:
                return [pad_token_id]

        if len(prefix) < preamble_len:
            return [preamble_tokenized[len(prefix)]]

        prefix = prefix[preamble_len:]

        action_id = 0
        for idx, tok in enumerate(prefix):
            if tok == postamble_tokenized[0]:
                # In the postamble
                assert action_id == 0
                action_id = None
                break
            action_id = (action_id + 1) % tokens_per_action

        if action_id is None:
            # In the postamble
            prefix = prefix[idx:]
            if len(prefix) < postamble_len:
                return [postamble_tokenized[len(prefix)]]
            else:
                return [pad_token_id]
        else:
            # In an action
            if is_token[action_id]:
                return action_tokens
            else:
                possible = [example_action[action_id]]
                # If starting a new action, and there is already something previously
                # Then can start the postamble (empty progs don't parse)
                if action_id == 0 and len(prefix):
                    possible.append(postamble_tokenized[0])
                return possible

    return lr_grammar


def add_special_tokens(
    tokenizer,
    model=None,
    initialize_embeddings=True,
    add_conditionals=False,
    mode="synthesis",
):
    """Adds special tokens, and also resizes model embedding if passed in."""
    orig_tokenizer = copy.deepcopy(tokenizer)
    old_count = len(tokenizer)

    new_tokens = ["\n"]
    new_tokens += DEFAULT_TOKENS
    new_tokens += ACTION_TOKENS
    if add_conditionals:
        new_tokens += CONDITIONAL_TOKENS
    new_tokens += [f"__{i}_Marker__" for i in range(1, 10)]
    for i in range(10):
        new_tokens.append(f"Input{i}")
        new_tokens.append(f"Output{i}")
    tokenizer.add_tokens(new_tokens)
    new_count = len(tokenizer)
    print(f"Added {new_count - old_count} tokens.")

    tokenizer.bos_token = "Examples:"
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = "}" if mode == "synthesis" else EOS_TOKEN
    tokenizer.padding_side = "right"

    if model is None:
        return new_tokens

    model.resize_token_embeddings(len(tokenizer))

    if not initialize_embeddings:
        return new_tokens

    # initialize new embedding weights as mean of original tokens
    weights = model.transformer.wte.weight
    with torch.no_grad():
        emb = []
        for idx, token in enumerate(new_tokens):
            tok_ids = orig_tokenizer(token)["input_ids"]
            tok_weights = weights[tok_ids]

            # average over tokens in original tokenization
            weight_mean = torch.mean(tok_weights, axis=0)
            emb.append(weight_mean)
        weights[-len(new_tokens) :, :] = torch.vstack(emb).requires_grad_()

    return new_count


def get_active_examples(
    num_examples, active_examples, randomize_active, use_alt=False, alt_active=None
):
    if use_alt is not None:
        assert num_examples == 2
        alt_active = str(alt_active)
        assert alt_active in ["random", "all", "0", "1"]
        active_examples = [True] * num_examples
        if alt_active == "random":
            s = random.random()
            p = 0.4
            if s < p:
                active_examples[0] = False
            elif s < 2 * p:
                active_examples[1] = False
        elif alt_active == "all":
            pass
        else:
            if alt_active == "0":
                active_examples[1] = False
            else:
                assert alt_active == "1"
                active_examples[0] = False
        return active_examples

    if isinstance(active_examples, int):
        if active_examples == num_examples:
            active_examples = [True] * num_examples
        else:
            # assert active_examples < num_examples
            if randomize_active:
                active = random.sample(list(range(num_examples)), active_examples)
            else:
                active = list(range(active_examples))
            active_examples = [i in active for i in range(num_examples)]
    else:
        active_examples = [True] * num_examples
    assert len(active_examples) == num_examples
    return active_examples


def _format_state(state):
    states = state.split("\n")
    assert len(states) == 8
    # The outer edge of the state is all walls
    # Remove:
    #  - first and last row, and
    #  - first and last token in each remaining row
    states = [state[1:-1] for state in states[1:-1]]
    return "\n".join(states)


def _format_special_tokens(text):
    text = text.replace("#", " __wall__")
    text = text.replace(".", " __space__")
    text = text.replace("<", " __left__")
    text = text.replace(">", " __right__")
    text = text.replace("^", " __up__")
    text = text.replace("v", " __down__")
    for m in range(1, 10):
        text = text.replace(f"put{m}", "XXX")
        text = text.replace(str(m), f" __{m}_Marker__")
        text = text.replace("XXX", f"put{m}")
    return text


def make_spec(
    example,
    num_examples,
    include_response=True,
    use_special_tokens=True,
    active_examples=None,
    randomize_active=False,
    inactive_mode=None,
    inactive_alt=None,
    alt_spec=False,
    use_alt=None,
    alt_active=None,
    mode="synthesis",
):
    if use_alt is not None:
        assert (
            active_examples is None or active_examples == num_examples
        ), active_examples
        assert not randomize_active
        assert not inactive_mode
        assert num_examples == 2, num_examples

    active_examples = get_active_examples(
        num_examples,
        active_examples,
        randomize_active,
        use_alt,
        alt_active,
    )
    if inactive_mode == "permute":
        inactive = [i for i, active in enumerate(active_examples) if not active]
        assert len(inactive) > 1
        inactive = inactive[1:] + [inactive[0]]

    def format_special_tokens(text):
        if use_special_tokens:
            return _format_special_tokens(text)
        return text

    parser = KarelWithCurlyParser()

    def parser_to_text(p):
        return _format_state(
            "\n".join(
                p.draw(
                    no_print=True,
                )
            )
        )

    def make_inputs():
        texts = []
        for idx, active in enumerate(active_examples):
            text = f"Input{idx}\n"
            text += _format_state(example[f"input{idx}"])
            texts.append(text)
        return texts

    if inactive_mode == "wall":
        inactive_state = (
            "########\n"
            "########\n"
            "########\n"
            "########\n"
            "########\n"
            "########\n"
            "########\n"
            "########"
        )
        inactive_state = _format_state(inactive_state)
    elif inactive_mode == "space":
        inactive_state = (
            "########\n"
            "#......#\n"
            "#......#\n"
            "#......#\n"
            "#......#\n"
            "#......#\n"
            "#......#\n"
            "########"
        )
        inactive_state = _format_state(inactive_state)

    def rerun_code(orig_state, code):
        # TODO: Hack, ideally load state dataset directly
        old_state = orig_state.split("\n")
        old_state = [state[1:-1] for state in old_state[1:-1]]
        parser.new_game(world_string=old_state)
        old_state = "\n".join(parser.draw(no_print=True))
        assert old_state == orig_state, (old_state, orig_state)
        parser.run(code)
        return parser_to_text(parser)

    def make_outputs():
        texts = []

        orig_code = example["code"]
        if use_alt is not None:
            alt_code = semantics_transformer(orig_code, use_alt)
            assert inactive_alt is None
        elif inactive_alt is not None:
            alt_code = semantics_transformer(orig_code, inactive_alt)
        for idx, active in enumerate(active_examples):
            text = f"Output{idx}\n"
            if active:
                text += _format_state(example[f"output{idx}"])
            else:
                if inactive_mode == "permute":
                    inactive_idx = inactive.pop(0)
                    text += _format_state(example[f"output{inactive_idx}"])
                elif inactive_mode in ["wall", "space"]:
                    text += inactive_state
                elif inactive_mode == "input":
                    text += _format_state(example[f"input{idx}"])
                elif inactive_mode == "random":
                    parser.new_game(world_size=WORLD_SIZE)
                    text += parser_to_text(parser)
                elif inactive_mode == "alt":
                    orig_state = example[f"input{idx}"]
                    text += rerun_code(orig_state, alt_code)
                else:
                    assert False, f"{inactive_mode=} not supported in make_spec."
            texts.append(text)
        return texts

    def make_code():
        code = example["code"]
        if alt_spec:
            assert inactive_alt is not None
            code = semantics_transformer(code, inactive_alt)
        code = code.replace(";", "\n ")
        return code

    if mode == "synthesis":

        def make_prompt():
            inputs = make_inputs()
            outputs = make_outputs()

            texts = ["Examples:\n"]
            for i, o in zip(inputs, outputs):
                texts.append(i)
                texts.append(o)
            return format_special_tokens("\n".join(texts))

        def make_response():
            text = "\nCode:\n"
            if include_response:
                text += make_code()
                assert text[-1] == "}"  # EOS
            return text

    elif mode == "interp":

        def make_prompt():
            texts = ["Examples:", format_special_tokens("\n".join(make_inputs()))]
            texts.extend(["\nCode:", make_code()])
            return "\n".join(texts)

        def make_response():
            text = "\nOutputs:\n"
            if include_response:
                text += format_special_tokens("\n".join(make_outputs()))
                text += EOS_TOKEN
            return text

    else:
        assert False, f"Mode must be one of synthesis or interp, but got {mode=}"

    text = make_prompt() + "\n" + make_response()
    return {"spec": text, "active": active_examples}


def load_karel(
    split,
    dataset_name="karel",
    num_examples=None,
    use_special_tokens=True,
    active_examples=None,
    randomize_active=False,
    inactive_mode="wall",
    inactive_alt=None,
    alt_spec=False,
    use_alt=None,
    alt_active=None,
    mode="synthesis",
    lengths_to_filter=None,
):
    raw_dataset = load_karel_raw(split, data_folder=dataset_name)

    # Infer the max number of examples if not provided.
    if num_examples is None:
        keys = raw_dataset["train"][0].keys()
        for num_examples in range(MAX_EXAMPLES):
            if f"input{num_examples+1}" not in keys:
                break
        num_examples += 1

    # Drop all examples with programs that have lengths in lengths_to_filter.
    if lengths_to_filter:
        # Code includes `def run()`
        # So it will have, e.g., 4 `()` when the program length is 3
        def _filter(example):
            return example["code"].count("()") - 1 not in lengths_to_filter

        filtered_dataset = raw_dataset.filter(_filter, load_from_cache_file=False)
    else:
        filtered_dataset = raw_dataset

    # Convert examples to specs.
    def _make_spec(example):
        return make_spec(
            example,
            num_examples=num_examples,
            use_special_tokens=use_special_tokens,
            active_examples=active_examples,
            randomize_active=randomize_active,
            inactive_mode=inactive_mode,
            inactive_alt=inactive_alt,
            alt_spec=alt_spec,
            use_alt=use_alt,
            alt_active=alt_active,
            mode=mode,
        )

    return filtered_dataset.map(
        _make_spec,
        batched=False,
        remove_columns=raw_dataset["train"].column_names,
        load_from_cache_file=False,
    )


def post_process_output(code: str):
    """
    Given an output generated by LLM, remove
      (1) examples from beginning, and
      (2) any extra text after first closed set of curly braces.

    Returns None if code does not define a run function with balanced curly
    braces. The returned code is not guaranteed to be well-formed.
    """
    try:
        start_idx = code.index("def run(")
    except ValueError:
        # print(f"Code does not define run: {code=}")
        return None
    # if start_idx > 0:
    #    return None
    code = code[start_idx:]
    count = 0
    for idx, char in enumerate(code):
        if char == "{":
            count += 1
        elif char == "}":
            count -= 1
            if count == 0:
                return code[: idx + 1]
    # return None
    code += " }" * count
    return code


def pp_to_parseable(code):
    """Ad-hoc post-processing hacks to make pretty-printed code paresable."""
    code = code.replace("__endline__", "\n")
    code = code.replace("__ws__", "")
    code = code.replace("()\n", "(); ")  # Add semi-colons
    code = code.replace("() ", "(); ")
    code = code.replace("}", "};")
    code = " ".join(code.split())  # Replace all whitespace
    code = code.replace("(); }", "() }")  # Remove semicolon at stmt end
    code = code.replace("}; else", "} else")  # Remove semicolon in if-else
    code = code.replace("}; }", "} }")  # Needs to run twice for } } }
    code = code.replace("}; }", "} }")
    code = code.replace("(); {", "() {")  # Remove semicolon after run()
    code = code[:-1]  # Remove semicolon at end of prog
    return code


def semantic_eval(
    response,
    spec_code,
    spec_examples,
    generate_trace=False,
    verbose=False,
    mode="synthesis",
):
    if not mode in ["synthesis", "interp"]:
        raise ValueError("semantic_eval not implemented for {mode=}")

    if mode == "synthesis":
        code = response
    else:
        assert mode == "interp"
        code = spec_code

    parser = KarelWithCurlyParser(run_with_trace=generate_trace)
    _parser = KarelWithCurlyParser()

    results = []
    str_traces = []
    np_traces = []

    try:
        pcode = post_process_output(code)
        pcode = pp_to_parseable(pcode)
    except:
        pcode = None

    def state_to_str(state):
        # _parser.new_game(state=state)
        # return "\n".join(_parser.draw(no_print=True))
        try:
            _parser.new_game(state=state)
            return "\n".join(_parser.draw(no_print=True))
        except:
            return state

    for idx, (input, output) in enumerate(spec_examples):
        expected = state_to_str(output)

        parser.new_game(state=input)
        try:
            with stdout_silencing():
                parser.run(pcode)
        except:
            results.append(None)
            np_traces.append(None)
            str_traces.append(None)
            continue

        if generate_trace:
            np_trace = parser.get_trace()
            np_traces.append(np_trace)
            str_trace = {
                key: [state_to_str(s) for s in states]
                for key, states in np_trace.items()
            }
            str_traces.append(str_trace)

        # Compare the string reps.
        if mode == "synthesis":
            result = "\n".join(parser.draw(no_print=True))
            results.append(_format_state(result) == _format_state(expected))
        else:
            assert mode == "interp"
            if not f"Output{idx}" in response:
                results.append(0)
            else:
                expected = _format_state(expected)
                expected = f"Output{idx}\n{expected}"
                expected = _format_special_tokens(expected)
                # results.append(expected in response)

                result_start = response.find(f"Output{idx}")
                if result_start < 0:
                    results.append(0)
                    continue
                result_end = response.find(f"Output{idx+1}")
                if result_end < 0:
                    result_end = None
                result = response[result_start:result_end]

                result = result.split()
                expected = expected.split()
                count = sum(r == e for r, e in zip(result, expected))
                results.append(count / len(expected))

    if verbose and None in results:
        print(f"Failed: {code}")

    stats = {"results": results}
    if generate_trace:
        stats["np_trace"] = np_traces
        stats["trace"] = str_traces
    return stats
