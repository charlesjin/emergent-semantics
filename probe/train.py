#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import copy
import logging
import math
import os
import random
import numpy as np

import torch
from torch.nn import functional as F

from transformers import AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
host = torch.device("cpu")

criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
full_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from data import karel
from utils import model as model_utils
from probe.dataset import SemanticKarelDataset
from probe.model import MLP
from utils.cache import CACHE
from utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train a semantic decoding model.")
    Config.add_train_args(parser)
    Config.add_eval_args(parser)
    Config.add_semantic_args(parser)

    parser.add_argument(
        "--train_split", type=str, default="train", choices=["test", "train"]
    )
    parser.add_argument(
        "--eval_split", type=str, default="test", choices=["test", "train"]
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=256,
        help="Batch size (per device) for the train dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1024,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_train_steps", type=int, default=None, help="Number of training steps."
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip if dataset already exists.",
    )
    args = parser.parse_args()

    return args


def make_meta_results(config, results):
    keys_to_del = ("correct_by_prog", "loss_by_prog")
    filtered = []
    for r in results:
        f = {k: v for k, v in r.items() if k not in keys_to_del}
        filtered.append(f)

    results_fn = config.semantic_results_meta_path

    torch.save(filtered, results_fn)
    print(f"Wrote meta results to {results_fn}.")


def train():
    args = parse_args()
    print()
    print(args)
    config = Config(args)

    train_split = args.train_split
    eval_split = args.eval_split
    train_batch_size = args.per_device_train_batch_size
    eval_batch_size = args.per_device_eval_batch_size
    num_train_steps = args.num_train_steps
    skip_if_exists = args.skip_if_exists

    return train_with_config(
        config,
        train_split,
        eval_split,
        train_batch_size,
        eval_batch_size,
        num_train_steps,
        skip_if_exists,
    )


@torch.no_grad()
def eval_ensemble(ensemble, eval_dataloader, all_stats=False):
    stats = []

    num_layers = len(ensemble[0][0])
    for layer_idx in range(num_layers):
        e_total = [0] * len(ensemble)
        e_correct = [0] * len(ensemble)
        e_loss = [0.0] * len(ensemble)

        e_total_by_task = []
        e_correct_by_task = []

        e_correct_by_start = []
        e_total_by_start = []
        e_correct_by_end = []
        e_total_by_end = []

        e_correct_by_prog = [[] for _ in ensemble]
        e_correct_for_prog = [[] for _ in ensemble]
        e_loss_by_prog = [[] for _ in ensemble]
        e_loss_for_prog = [[] for _ in ensemble]

        for task_ensemble in ensemble:
            for layer_ensemble in task_ensemble:
                if layer_ensemble[layer_idx]:
                    layer_ensemble[layer_idx].eval()

            e_total_by_task.append([0] * len(task_ensemble))
            e_correct_by_task.append([0] * len(task_ensemble))

            e_correct_by_start.append({idx: 0 for idx in range(30)})
            e_total_by_start.append({idx: 0 for idx in range(30)})
            e_correct_by_end.append({idx: 0 for idx in range(30)})
            e_total_by_end.append({idx: 0 for idx in range(30)})

        stats.append(
            (
                e_total,
                e_correct,
                e_loss,
                e_total_by_task,
                e_correct_by_task,
                e_correct_by_start,
                e_total_by_start,
                e_correct_by_end,
                e_total_by_end,
                e_correct_by_prog,
                e_correct_for_prog,
                e_loss_by_prog,
                e_loss_for_prog,
            )
        )

    for inputs, labels, idxs, lengths in eval_dataloader:
        # assert int(labels.size(-1)) == len(task_ensemble)
        inputs, labels = inputs.to(device), [l.to(device) for l in labels]

        batch_shape = labels[0].shape[:-1]

        for layer_idx, (
            e_total,
            e_correct,
            e_loss,
            e_total_by_task,
            e_correct_by_task,
            e_correct_by_start,
            e_total_by_start,
            e_correct_by_end,
            e_total_by_end,
            e_correct_by_prog,
            e_correct_for_prog,
            e_loss_by_prog,
            e_loss_for_prog,
        ) in enumerate(stats):
            for task_idx, task_ensemble in enumerate(ensemble):
                correct_all = torch.zeros(batch_shape, device=device)
                total_all = torch.zeros(batch_shape, device=device)
                loss_all = torch.zeros(batch_shape, device=device)
                task_labels = labels[task_idx]
                for ensemble_idx, layer_ensemble in enumerate(task_ensemble):
                    eval_model = layer_ensemble[layer_idx]
                    if not eval_model:
                        continue
                    ensemble_labels = task_labels[:, :, ensemble_idx]

                    with torch.no_grad():
                        outputs = eval_model(inputs)

                    e_loss[task_idx] += criterion(outputs, ensemble_labels)
                    _loss_all = full_criterion(outputs, ensemble_labels)

                    _predicted = torch.argmax(outputs, dim=1)
                    _correct = _predicted == ensemble_labels

                    mask = ensemble_labels >= 0

                    e_total_by_task[task_idx][ensemble_idx] += mask.sum().item()
                    e_correct_by_task[task_idx][ensemble_idx] += (
                        _correct[mask].sum().item()
                    )

                    correct_all[mask] += _correct[mask]
                    total_all[mask] += 1
                    loss_all[mask] += _loss_all[mask]

                correct_all = (correct_all == total_all).cpu()
                mask = (total_all > 0).cpu()
                correct_all[~mask] = 0

                e_correct[task_idx] += correct_all.sum().item()
                e_total[task_idx] += mask.sum().item()
                loss_all = loss_all.cpu()

                if not all_stats:
                    continue

                # Collect results by program, for each token
                for b_idx, tok_idx in enumerate(idxs):
                    # Starting a new program, so save last program
                    if tok_idx == 0 and e_correct_for_prog[task_idx]:
                        e_correct_by_prog[task_idx].append(e_correct_for_prog[task_idx])
                        e_correct_for_prog[task_idx] = []
                        e_loss_by_prog[task_idx].append(e_loss_for_prog[task_idx])
                        e_loss_for_prog[task_idx] = []

                    correct_for_tok = correct_all[b_idx].detach()
                    loss_for_tok = loss_all[b_idx].detach().clone()
                    # Save results for program only if label is not ignored
                    loss_for_tok[~mask[b_idx]] = -1

                    e_correct_for_prog[task_idx].append(correct_for_tok)
                    e_loss_for_prog[task_idx].append(loss_for_tok)

                # Collect results by depth in program
                # Assumes program length is at most 30
                for idx in range(30):
                    start_mask = idxs == idx
                    e_total_by_start[task_idx][idx] += mask[start_mask].sum().item()
                    e_correct_by_start[task_idx][idx] += (
                        correct_all[start_mask].sum().item()
                    )

                    end_mask = idxs == (lengths - idx - 1)
                    e_total_by_end[task_idx][idx] += mask[end_mask].sum().item()
                    e_correct_by_end[task_idx][idx] += (
                        correct_all[end_mask].sum().item()
                    )

    l_results = []
    for layer_idx, (
        e_total,
        e_correct,
        e_loss,
        e_total_by_task,
        e_correct_by_task,
        e_correct_by_start,
        e_total_by_start,
        e_correct_by_end,
        e_total_by_end,
        e_correct_by_prog,
        e_correct_for_prog,
        e_loss_by_prog,
        e_loss_for_prog,
    ) in enumerate(stats):
        e_results = []
        for task_idx, task_ensemble in enumerate(ensemble):
            if e_correct_for_prog[task_idx]:
                e_correct_by_prog.append(e_correct_for_prog[task_idx])
                e_loss_by_prog.append(e_loss_for_prog[task_idx])

            e_loss[task_idx] /= len(task_ensemble)
            if e_total[task_idx]:
                e_loss[task_idx] /= e_total[task_idx]

            results = {
                "correct": e_correct[task_idx],
                "total": e_total[task_idx],
                "correct_by_prog": e_correct_by_prog[task_idx],
                "loss_by_prog": e_loss_by_prog[task_idx],
                "correct_by_task": e_correct_by_task[task_idx],
                "total_by_task": e_total_by_task[task_idx],
                "loss": e_loss[task_idx],
            }
            results.update(
                {
                    f"correct_start{k}": v
                    for k, v in e_correct_by_start[task_idx].items()
                }
            )
            results.update(
                {f"total_start{k}": v for k, v in e_total_by_start[task_idx].items()}
            )
            results.update(
                {f"correct_end{k}": v for k, v in e_correct_by_end[task_idx].items()}
            )
            results.update(
                {f"total_end{k}": v for k, v in e_total_by_end[task_idx].items()}
            )
            e_results.append(results)
        l_results.append(e_results)
    return l_results


def train_with_config(
    config,
    train_split,
    eval_split,
    train_batch_size,
    eval_batch_size,
    num_train_steps,
    skip_if_exists,
    task=None,
    train_dataset=None,
    eval_dataset=None,
    val_dataset=None,
    ema_decay=0,
    ema_steps=1000,
    gamma=None,
    dropout_p=0.2,
    eval_steps=0,
    train_samples=100000,
    eval_samples=None,
    val_samples=10000,
    min_train_samples=10000,
    max_train_epochs=200,
):
    assert torch.cuda.is_available(), "no cuda"
    if config.forced_decoding:
        if not isinstance(config.offset, int) or config.offset >= 0:
            print(
                "WARNING: using forced decoding dataset but offset is "
                "nonnegative (predicting the future)."
            )

    if train_split == eval_split:
        print(f"WARNING: train and eval on same split={train_split}!")

    try:
        tokenizer = model_utils.load_pretrained(config, load_tokenizer_only=True)
    except:
        print("could not load tokenizer, reconstructing.")
        args = torch.load(f"{config.train_base_dir}/args.pt")
        add_conditionals = "nocond" not in config.dataset
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        karel.add_special_tokens(
            tokenizer,
            model=None,
            add_conditionals=add_conditionals
        )

    if config.mlp_layers is None:
        layers = [1, 2, 3]
        # layers = [3]
    else:
        layers = [config.mlp_layers]

    if config.mlp_icml_scheduler:
        ema_decay = False
        train_batch_size = 512
        num_train_steps = 5000000

    num_tasks = 6

    layers_to_skip = []
    for layer in layers:
        config = config.update(mlp_layers=layer)
        base_dir = config.base_dir
        results_fn = config.semantic_results_path
        probe_fn = config.semantic_probe_path

        skip = False
        if skip_if_exists and not config.debug and CACHE.check(results_fn):
            meta_results_fn = config.semantic_results_meta_path
            if not CACHE.check(meta_results_fn):
                try:
                    res = torch.load(results_fn, map_location="cpu")
                    make_meta_results(config, res)
                    skip = True
                except:
                    pass
            else:
                skip = True
        if skip:
            layers_to_skip.append(layer)
            print(f"{results_fn} already exists, skipping mlp_layers={layer}.")
        else:
            try:
                os.remove(results_fn)
            except:
                pass
        try:
            # os.remove(results_fn)
            os.remove(probe_fn)
        except OSError:
            pass

    print(f"{layers_to_skip=}")
    for layer in layers_to_skip:
        layers.remove(layer)
    if not layers:
        return

    if val_dataset is None:
        val_config = config.update(split="val")
        val_dataset = SemanticKarelDataset(
            val_config,
            tokenizer,
            drop_last=config.drop_last,
            filter_correct=False,
            filter_inactive=config.eval_alt_active in ["0", "1"],
            single_label=not config.all_labels,
        )
    mean, std = val_dataset.mean, val_dataset.std

    if eval_dataset is None:
        if train_split == eval_split and train_dataset is not None:
            eval_dataset = train_dataset
        else:
            eval_config = config.update(
                split=eval_split,
                filter_lengths=None,
            )
            eval_dataset = SemanticKarelDataset(
                eval_config,
                tokenizer,
                drop_last=config.drop_last,
                filter_correct=False,
                filter_inactive=config.eval_alt_active in ["0", "1"],
                single_label=not config.all_labels,
                mean=mean,
                std=std,
            )

    # assert len(eval_dataset.input_shape) == 1
    assert num_tasks == len(eval_dataset.num_classes)

    if train_dataset is None:
        if train_split == eval_split:  # or config.debug:
            if train_samples:
                train_dataset, eval_dataset = eval_dataset.split(train_samples)
                train_samples = None
            else:
                train_dataset = eval_dataset
        else:
            train_config = config.update(split=train_split)
            train_dataset = SemanticKarelDataset(
                train_config,
                tokenizer,
                drop_last=config.drop_last,
                filter_correct=False,
                filter_inactive=config.eval_alt_active in ["0", "1"],
                single_label=not config.all_labels,
                mean=mean,
                std=std,
                filter_lengths=config.filter_lengths,
            )
            # max_samples=40000)

    if train_samples:
        train_dataset.resample(train_samples, random=True)
    if eval_samples:
        eval_dataset.resample(eval_samples, random=True)
    if val_samples:
        val_dataset.resample(val_samples, random=True)

    train_val_dataset = train_dataset.split(num_samples=len(val_dataset))[0]
    # train_val_dataset.train = False

    if len(train_dataset) < min_train_samples:
        print(
            f"Only {len(train_dataset)} train samples, "
            f"but need at least {min_train_samples}... skipping."
        )
        return

    print(f"Writing eval results to {results_fn}.")  # and saving probe to {probe_fn}.")

    num_train_steps = min(num_train_steps, len(train_dataset) * max_train_epochs)
    total_epochs = num_train_steps / len(train_dataset)
    steps_per_epoch = len(train_dataset) / train_batch_size
    num_train_steps //= train_batch_size
    num_examples = config.num_examples

    if eval_steps == 0:
        eval_steps = num_train_steps // 20
        eval_steps = int(max(eval_steps, steps_per_epoch))
    elif eval_steps is not None:
        eval_steps //= train_batch_size

    print(train_dataset.num_classes)

    print("***** Running training *****")
    print(f"  Num train = {len(train_dataset)}")
    print(f"  Num test = {len(eval_dataset)}")
    print(f"  Num val = {len(val_dataset)}")
    print(f"  Num batches = {num_train_steps}")
    print(f"  Num epochs = {int(total_epochs)}")

    # DataLoaders creation:
    train_val_dataloader = DataLoader(
        train_val_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=eval_batch_size,
        num_workers=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=eval_batch_size,
        num_workers=2,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=eval_batch_size,
        num_workers=2,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        batch_size=train_batch_size,
        num_workers=2,
    )

    # Create model ensemble
    ensemble = []
    best_models = []
    for task_idx, num_classes in enumerate(train_dataset.num_classes):
        if task is not None and task != task_idx:
            continue
        task_ensemble = []
        best_task_ensemble = []
        label_shape = train_dataset[0][1][task_idx].shape[:-1]
        for num_class in num_classes:
            layer_ensemble = []
            for mlp_layers in layers:
                model = MLP(
                    train_dataset.input_shape,
                    # train_dataset.label_shape[:-1],
                    label_shape,
                    num_classes=num_class,
                    num_layers=mlp_layers,
                    use_layernorm=config.mlp_layernorm,
                    dropout_p=dropout_p,
                )
                model = model.to(device)

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
                        "weight_decay": 2e-1,
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

                if config.mlp_icml_scheduler:
                    scheduler_steps = None
                    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=0.01)
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[int(x * total_epochs) for x in [0.75, 0.9]],
                        gamma=gamma if gamma else 0.1,
                    )
                else:
                    scheduler_steps = num_train_steps // 150
                    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=0.02)
                    # optimizer = torch.optim.AdamW(
                    #    optimizer_grouped_parameters, lr=0.002 * num_class
                    # )
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer,
                        gamma=gamma if gamma else 0.95,
                    )

                if ema_decay:
                    ema_updates = num_train_steps / ema_steps
                    ema_warmup = ema_updates / 10
                    decay = lambda x: ema_decay * (1 - math.exp(-x / ema_warmup))
                    ema_avg = (
                        lambda old_p, new_p, x: (1 - decay(x)) * old_p
                        + decay(x) * new_p
                    )
                    ema_model = torch.optim.swa_utils.AveragedModel(
                        model, avg_fn=ema_avg, use_buffers=True
                    )
                else:
                    ema_model = None

                layer_ensemble.append(
                    (model, scheduler_steps, optimizer, scheduler, ema_model)
                )
            task_ensemble.append(layer_ensemble)
            best_task_ensemble.append([None] * len(layer_ensemble))
        ensemble.append((task_idx, task_ensemble))
        best_models.append(best_task_ensemble)

    # Only show the progress bar once on each machine.
    pbar = tqdm(range(num_train_steps))

    completed_steps = 0
    epoch = 0
    best_acc = [[0.0] * len(ensemble) for _ in layers]
    all_results = [[[] for _ in ensemble] for _ in layers]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Train loop
    while completed_steps < num_train_steps:
        # Prepare models for training.
        for _, task_ensemble in ensemble:
            for layer_ensemble in task_ensemble:
                for model, *_ in layer_ensemble:
                    model.train()

        # Train one epoch.
        epoch += 1
        for inputs, labels, idxs, _ in train_dataloader:
            # inputs = inputs[idxs > 10]
            # labels = [l[idxs > 10] for l in labels]
            inputs, labels = inputs.to(device), [l.to(device) for l in labels]

            pbar.update(1)
            completed_steps += 1

            # Train one step.
            for task_idx, task_ensemble in ensemble:
                task_labels = labels[task_idx]
                for ensemble_idx, layer_ensemble in enumerate(task_ensemble):
                    ensemble_labels = task_labels[:, :, ensemble_idx]
                    for (
                        model,
                        scheduler_steps,
                        optimizer,
                        scheduler,
                        ema_model,
                    ) in layer_ensemble:
                        outputs = model(inputs)

                        optimizer.zero_grad()
                        loss = criterion(outputs, ensemble_labels)
                        loss.backward()
                        optimizer.step()

                        if ema_decay and ema_steps:
                            if (
                                completed_steps >= num_train_steps
                                or completed_steps % ema_steps == 0
                            ):
                                ema_model.update_parameters(model)

                        if scheduler_steps and completed_steps % scheduler_steps == 0:
                            scheduler.step()

            # Run evaluation.
            if (
                completed_steps >= num_train_steps
                or eval_steps
                and completed_steps % eval_steps == 0
            ):
                accs = []
                ensemble_models = []
                for _, task_ensemble in ensemble:
                    task_models = []
                    for layer_ensemble in task_ensemble:
                        layer_models = [
                            ema_model if ema_decay else model
                            for model, _, _, _, ema_model in layer_ensemble
                        ]
                        task_models.append(layer_models)
                    ensemble_models.append(task_models)

                accs = [[], []]
                for is_train, dl in enumerate([eval_dataloader, train_val_dataloader]):
                    l_results = eval_ensemble(
                        ensemble_models,
                        dl,
                        all_stats=False,
                    )

                    for layer_idx, e_results in enumerate(l_results):
                        for task_idx, results in enumerate(e_results):
                            all_results[layer_idx][task_idx].append(results)

                            # Log results.
                            loss = results["loss"]
                            correct = results["correct"]
                            total = results["total"]
                            if total:
                                acc = correct / total * 100
                                if layer_idx == len(l_results) - 1:
                                    accs[is_train].append(f"{acc:.1f}")

                                if not is_train and acc > best_acc[layer_idx][task_idx]:
                                    best_acc[layer_idx][task_idx] = acc

                                    for ensemble_idx, task_models in enumerate(
                                        ensemble_models[task_idx]
                                    ):
                                        model = copy.deepcopy(task_models[layer_idx])
                                        best_models[task_idx][ensemble_idx][
                                            layer_idx
                                        ] = model
                            else:
                                if layer_idx == len(l_results) - 1:
                                    accs[is_train].append(f"n/a")

                val_accs = " ".join(accs[0])
                train_accs = " ".join(accs[1])
                pbar.set_description(
                    f"[{epoch}] " f"val: {val_accs} | " f"train: {train_accs}"
                )

                if completed_steps >= num_train_steps:
                    break

                for _, task_ensemble in ensemble:
                    for layer_ensemble in task_ensemble:
                        for model, *_ in layer_ensemble:
                            model.train()

        for _, task_ensemble in ensemble:
            for layer_ensemble in task_ensemble:
                for _, scheduler_steps, _, scheduler, _ in layer_ensemble:
                    if not scheduler_steps:
                        scheduler.step()

    l_results = eval_ensemble(best_models, eval_dataloader, all_stats=True)

    best_results = []
    for layer_idx, e_results in enumerate(l_results):
        pbar.write(f"layer={layers[layer_idx]}")
        final_acc = []
        best_results.append([])
        for task_idx, results in enumerate(e_results):
            best_results[layer_idx].append(results)

            loss = results["loss"]
            correct = results["correct"]
            total = results["total"]
            if total:
                acc = correct / total * 100
                final_acc.append(acc)
            else:
                final_acc.append("n/a")

            # task_models = ensemble_models[task_idx]
            # layer_models = [
            #    task_model[layer_idx].state_dict() for task_model in task_models
            # ]
            # best_model[layer_idx][task_idx] = layer_models
        pbar.write(f"Best during training: {best_acc[layer_idx]}")
        pbar.write(f"At final eval: {final_acc}")

    if config.debug:
        for layer_idx, layer in enumerate(layers):
            pbar.write(f"{layer=}")
            for task_idx, results in enumerate(best_results[layer_idx]):
                pbar.write(f"{task_idx=}")
                pbar.write(f"BEST: {best_acc[layer_idx][task_idx]:.1f}%")

                correct = results["correct"]
                total = results["total"]
                if total:
                    acc = correct / total * 100
                    pbar.write(f"FINAL: {acc:.1f}% ({correct:.1f} / {total})")

                # for idx in range(30):
                #    correct = results[f"correct_start{idx}"]
                #    total = results[f"total_start{idx}"]
                #    if not total:
                #        continue
                #    acc = correct / total * 100
                #    pbar.write(f"start{idx}: {acc:.1f}% ({correct:.1f} / {total})")
                # for idx in range(30):
                #    correct = results[f"correct_end{idx}"]
                #    total = results[f"total_end{idx}"]
                #    if not total:
                #        continue
                #    acc = correct / total * 100
                #    pbar.write(f"end{idx}: {acc:.1f}% ({correct:.1f} / {total})")
    pbar.close()

    if not config.debug:
        for layer_idx, layer in enumerate(layers):
            config = config.update(mlp_layers=layer)
            results_fn = config.semantic_results_path
            # probe_fn = config.semantic_probe_path

            torch.save(best_results[layer_idx], results_fn)
            print(f"Wrote eval results to {results_fn}.")

            make_meta_results(config, best_results[layer_idx])

            # torch.save(best_models[layer_idx].state_dict(), probe_fn)
            # print(f"Wrote probe to {probe_fn}.")
            # print()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # return all_results
    return best_results


if __name__ == "__main__":
    train()
