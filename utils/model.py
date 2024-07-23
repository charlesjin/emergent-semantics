import sys
import os
import importlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_arch(arch, load_in_8bit=True, device=None, use_fast_tokenizer=True):
    if load_in_8bit:
        assert torch.cuda.is_available()
        model = AutoModelForCausalLM.from_pretrained(
            arch, device_map="auto", load_in_8bit=load_in_8bit
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(arch)
        if device is not None:
            model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(arch, use_fast=use_fast_tokenizer)
    return model, tokenizer


def load_pretrained(
    config, load_in_8bit=False, use_fast_tokenizer=True, load_tokenizer_only=False
):
    def load_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained(
            config.train_base_dir, use_fast=use_fast_tokenizer
        )
        tokenizer.pad_token = "<PAD>"
        tokenizer.padding_side = "right"
        return tokenizer

    tokenizer = load_tokenizer()
    if load_tokenizer_only:
        return tokenizer

    model, _ = load_arch(
        config.model_name,
        load_in_8bit=load_in_8bit,
    )
    model.resize_token_embeddings(len(tokenizer))
    path = config.checkpoint_dir
    model.load_state_dict(torch.load(f"{path}/pytorch_model.bin"))
    print(f"Loaded model at {path}...")

    return model, tokenizer
