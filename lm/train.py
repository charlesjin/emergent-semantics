#!/usr/bin/env python
# coding=utf-8

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py

import argparse
import json
import copy
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# from datasets import load_dataset
# from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

# from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from lm.eval import eval_with_config_and_model, load_datasets
from data.karel import load_karel
from data import karel
from utils.config import Config
import utils.model as model_utils

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")

logger = get_logger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformers model on a causal language modeling task"
    )
    Config.add_train_args(parser)

    parser.add_argument(
        "--lengths_to_filter",
        nargs="+",
        type=int,
        default=None,
        help="Drops programs of the given lengths from the training set.",
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        const="",
        type=str,
        default=None,
        action="store",
        nargs="?",
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(f"{args=}")
    config = Config(args)

    lengths_to_filter = args.lengths_to_filter
    use_slow_tokenizer = args.use_slow_tokenizer
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_train_epochs = args.num_train_epochs
    max_train_steps = args.max_train_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    lr_scheduler_type = args.lr_scheduler_type
    num_warmup_steps = args.num_warmup_steps
    seed = args.seed
    block_size = args.block_size
    preprocessing_num_workers = args.preprocessing_num_workers
    overwrite_cache = args.overwrite_cache
    no_keep_linebreaks = args.no_keep_linebreaks
    checkpointing_steps = args.checkpointing_steps
    resume_from_checkpoint = args.resume_from_checkpoint
    with_tracking = args.with_tracking
    report_to = args.report_to

    output_dir = config.base_dir

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if with_tracking:
        accelerator_log_kwargs["log_with"] = report_to
        accelerator_log_kwargs["logging_dir"] = output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None and not config.debug:
            os.makedirs(output_dir, exist_ok=True)
            accelerator.print(f"{output_dir=}")
            try:
                old_args = torch.load(f"{output_dir}/args.pt")
                if not old_args == args:

                    def _args_to_vars(_args):
                        return set(
                            (k, tuple(v) if isinstance(v, list) else v)
                            for k, v in vars(_args).items()
                        )

                    set1 = _args_to_vars(args)
                    set2 = _args_to_vars(old_args)
                    raise ValueError(
                        f"Args provided\n"
                        f"  {set1 - set2}\n"
                        f"Saved at {output_dir}/args.pt\n"
                        f"  {set2 - set1}\n"
                    )
            except FileNotFoundError:
                torch.save(args, f"{output_dir}/args.pt")
    del args

    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # if args.model_name:
    #    model_config = AutoConfig.from_pretrained(args.model_name)
    # elif args.model_name_or_path:
    #    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    # else:
    #    model_config = CONFIG_MAPPING[args.model_type]()
    #    logger.warning("You are instantiating a new config instance from scratch.")
    # if args.model_name_or_path:
    #    model = AutoModelForCausalLM.from_pretrained(
    #        args.model_name_or_path,
    #        from_tf=bool(".ckpt" in args.model_name_or_path),
    #        config=model_config,
    #    )
    # else:

    logger.info("Training new model from scratch")
    model_config = AutoConfig.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_config(model_config)

    # Load or create tokenizer
    try:
        tokenizer = model_utils.load_pretrained(
            config, use_fast_tokenizer=not use_slow_tokenizer, load_tokenizer_only=True
        )
    except:
        add_conditionals = "nocond" not in config.dataset
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, use_fast=not use_slow_tokenizer
        )
        karel.add_special_tokens(
            tokenizer, model=model, add_conditionals=add_conditionals
        )
        if accelerator.is_main_process and output_dir is not None and not config.debug:
            tokenizer.save_pretrained(output_dir)
    model.resize_token_embeddings(len(tokenizer))

    # Load in the raw datasets.
    if "karel" in config.dataset:
        load_dataset = lambda *a, **kw: load_karel(
            *a,
            dataset_name=config.dataset,
            active_examples=config.num_active_examples,
            randomize_active=config.randomize_active,
            inactive_mode=config.inactive_mode,
            inactive_alt=config.inactive_alt,
            alt_spec=config.alt_spec,
            use_alt=config.use_alt,
            alt_active="random",
            mode=config.mode,
            **kw,
        )
    else:
        raise ValueError("Dataset {config.dataset} not recognized.")

    # Test tokenizer
    val_dataset = load_dataset("val", num_examples=config.num_examples)
    sample = val_dataset["train"][0]["spec"]
    tokenizer(sample)

    raw_datasets = load_dataset(
        "val" if config.debug else "train",
        num_examples=config.num_examples,
        lengths_to_filter=lengths_to_filter,
    )
    raw_datasets["val"] = val_dataset["train"]

    logger.info(
        f"Sample 0 of the raw training set:\n" f"{raw_datasets['train'][0]['spec']}"
    )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=preprocessing_num_workers,
            load_from_cache_file=not overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["val"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Prepare eval datasets
    raw_dataset, state_dataset = load_datasets(config)

    eval_config = config.update(
        grammar_decoding=True,
        split="val",
        eval_alt_active="random" if config.use_alt is not None else None,
    )

    @torch.no_grad()
    def eval(m, fast=True):
        m.eval()
        max_eval_samples = 2 * per_device_eval_batch_size if fast else len(raw_dataset)
        _, correct, total = eval_with_config_and_model(
            eval_config,
            m,
            tokenizer,
            batch_size=per_device_eval_batch_size,
            make_semantic_dataset=False,
            generate_trace=False,
            max_eval_samples=max_eval_samples,
            k=1,
            temperature=None,
            resume=False,
            raw_dataset=raw_dataset,
            state_dataset=state_dataset,
            save_results=False,
        )

        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = m(**batch)

            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size))
            )
            if fast and step:
                break

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses).half().item()
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        m.train()
        return correct / total, perplexity, loss

    # Train!
    total_batch_size = (
        per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    checkpoint_path = None
    resume_step = None
    all_results = {"train": [], "val": []}
    if resume_from_checkpoint is not None and output_dir is not None:
        if resume_from_checkpoint != "":
            path = resume_from_checkpoint
            checkpoint_path = os.path.join(output_dir, path)
            checkpoint_paths = [checkpoint_path]
        else:
            # Get the most recent checkpoint
            checkpoint_paths = [
                os.path.join(output_dir, f.name)
                for f in os.scandir(output_dir)
                if f.is_dir()
            ]

            def key(p):
                path = os.path.basename(p)
                training_difference = os.path.splitext(path)[0]
                return -int(training_difference.replace("step_", ""))

            checkpoint_paths = sorted(checkpoint_paths, key=key)

        for checkpoint_path in checkpoint_paths:
            try:
                accelerator.print(f"Try resuming from checkpoint: {checkpoint_path}")

                path = os.path.basename(checkpoint_path)
                training_difference = os.path.splitext(path)[0]
                step = int(training_difference.replace("step_", ""))

                # starting new finetune run, so don't load optimizer / scheduler state
                if step == 0:
                    model.load_state_dict(
                        torch.load(f"{checkpoint_path}/pytorch_model.bin")
                    )
                    # step_config = config.update(step=step)
                    # model, _ = load_pretrained(config, model=model)
                else:
                    accelerator.load_state(checkpoint_path)

                # if step == 0:
                #    model.resize_token_embeddings(len(tokenizer))
                #    optimizers = accelerator._optimizers
                #    accelerator._optimizers = []
                #    schedulers = accelerator._schedulers
                #    accelerator._schedulers = []
                # accelerator.load_state(checkpoint_path)
                # if step == 0:
                #    accelerator._optimizers = optimizer
                #    accelerator._schedulers = schedulers

                all_results_path = os.path.join(checkpoint_path, "synthesis_train.pth")
                all_results = torch.load(all_results_path)
                break
            except Exception as e:
                print(f"Failed to resume from {checkpoint_path}, exception {e}")
        else:
            checkpoint_path = None

    if checkpoint_path is not None:
        # Extract `epoch_{i}` or `step_{i}`
        path = os.path.basename(checkpoint_path)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    first_step = False
    train_losses = []
    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        if with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if resume_from_checkpoint is not None and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            if not first_step:
                progress_bar.write(f"First step: {completed_steps}")
                first_step = True
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss += loss.detach().float()
                train_losses.append(loss.detach().half())
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % 16 == 0:
                    train_loss = (sum(train_losses) / len(train_losses)).half().item()
                    try:
                        perplexity = math.exp(train_loss)
                    except OverflowError:
                        perplexity = float("inf")
                    all_results["train"].append(
                        (completed_steps, perplexity, train_loss)
                    )
                    train_losses = []
                    progress_bar.write(
                        f"step {completed_steps}: perplexity: {perplexity} train_loss: {train_loss}"
                    )

                if completed_steps % 128 == 1:
                    torch.cuda.empty_cache()
                    acc, perplexity, loss = eval(model)
                    all_results["val"].append((completed_steps, perplexity, loss))
                    progress_bar.write(
                        f"step {completed_steps}: acc: {acc} perplexity: {perplexity} val_loss: {loss}"
                    )

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        if output_dir is not None and not config.debug:
                            step_output_dir = f"step_{completed_steps}"
                            step_output_dir = os.path.join(output_dir, step_output_dir)
                            accelerator.save_state(step_output_dir)
                            all_results_path = os.path.join(
                                step_output_dir, "synthesis_train.pth"
                            )
                            torch.save(all_results, all_results_path)
            if completed_steps >= max_train_steps:
                break

        acc, perplexity, loss = eval(model, fast=False)
        logger.info(
            f"epoch {epoch}: acc: {acc} perplexity: {perplexity} val_loss: {loss}"
        )

        if with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if checkpointing_steps == "epoch":
            if output_dir is not None and not config.debug:
                epoch_output_dir = f"epoch_{epoch}"
                epoch_output_dir = os.path.join(output_dir, epoch_output_dir)
                accelerator.save_state(epoch_output_dir)

    if with_tracking:
        accelerator.end_training()

    if output_dir is not None and not config.debug:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            with open(os.path.join(output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
