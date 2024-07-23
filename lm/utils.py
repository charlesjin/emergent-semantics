import inspect
import copy

import torch
import torch.nn.functional as F


####################
#
# Decoding
#
####################


@torch.no_grad()
def get_scores(logits, labels, offset=False):
    if offset:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    logits = F.softmax(logits, dim=-1)
    labels = labels.unsqueeze(-1)
    scores = torch.gather(logits, 2, labels)
    return scores


@torch.no_grad()
def decode_with_grammar(
    model,
    model_kwargs,
    input_ids,
    lr_grammar,
    max_length=128,
    output_hidden_states_and_scores=False,
):
    device = next(model.parameters()).device

    inputs_tensor, model_input_name, model_kwargs = model._prepare_model_inputs(
        input_ids, model_kwargs["bos_token_id"], model_kwargs
    )
    model_kwargs["output_hidden_states"] = True
    model_kwargs["use_cache"] = True
    assert "attention_mask" in model_kwargs

    # accepts_attention_mask = "attention_mask" in set(
    #    inspect.signature(model.forward).parameters.keys()
    # )
    # requires_attention_mask = "encoder_outputs" not in model_kwargs
    # if (
    #    model_kwargs.get("attention_mask", None) is None
    #    and requires_attention_mask
    #    and accepts_attention_mask
    # ):
    #    model_kwargs["attention_mask"] = model._prepare_attention_mask_for_generation(
    #        inputs_tensor, model_kwargs["pad_token_id"], model_kwargs["eos_token_id"]
    #    )

    input_ids = (
        inputs_tensor
        if model_input_name == "input_ids"
        else model_kwargs.pop("input_ids")
    )

    scores = ()
    hidden_states = ()

    num_tokens, _ = model.get_input_embeddings().weight.size()
    mask = torch.ones([num_tokens], dtype=bool, device=device)
    active = torch.ones([len(input_ids)], dtype=bool, device=device)

    input_len = len(input_ids[0])
    if input_len >= max_length:
        raise ValueError(f"Asked to generate {max_length=}, but {input_len=}")

    for _ in range(max_length - input_len):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=output_hidden_states_and_scores,
        )

        output_ids = torch.ones([len(input_ids)], dtype=torch.long, device=device)
        for idx, prog in enumerate(input_ids[:, input_len:]):
            legal_tokens = lr_grammar(prog, model_kwargs["pad_token_id"])
            mask[:] = 1
            mask[torch.LongTensor(legal_tokens)] = 0

            _scores = outputs.logits[idx, -1, :].clone().detach()
            _scores[mask] = float("-inf")
            output_ids[idx] = torch.argmax(_scores, dim=-1)
            active[idx] &= output_ids[idx] != model_kwargs["eos_token_id"]

        # _scores = outputs.logits[:, -1, :].clone().detach()
        # output_ids = torch.argmax(_scores, dim=-1)

        if output_hidden_states_and_scores:
            hidden_states += (outputs.hidden_states,)
            scores += (outputs.logits[:, -1, :],)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )

        input_ids = torch.cat([input_ids, output_ids[:, None]], dim=-1)
        if sum(active) == 0:
            break

    outputs.sequences = input_ids
    outputs.hidden_states = hidden_states
    outputs.scores = scores

    return outputs


@torch.no_grad()
def nucleus_sample(
    model,
    tokenizer,
    prompt,
    n=3,
    temperature=0.8,
    max_length=256,
    output_hidden_states_and_scores=False,
    lr_grammar=None,
):
    model.eval()
    device = next(model.parameters()).device
    if not isinstance(prompt, list):
        prompt = [prompt]

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_len = len(input_ids[0])

    if n > 1 or temperature is not None:
        assert lr_grammar is None
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_length=max_length + input_len,
            top_k=50,
            top_p=0.95,
            temperature=temperature,
            num_return_sequences=n,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_hidden_states=output_hidden_states_and_scores,
            output_scores=output_hidden_states_and_scores,
            return_dict_in_generate=True,
            generation_config=model.generation_config,
        )
    else:
        # Greedy
        if lr_grammar is None:
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_length=max_length + input_len,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_hidden_states=output_hidden_states_and_scores,
                output_scores=output_hidden_states_and_scores,
                return_dict_in_generate=True,
                generation_config=model.generation_config,
            )
        else:
            generation_config = copy.deepcopy(model.generation_config)
            model_kwargs = dict(
                return_dict=True,
                output_hidden_states=output_hidden_states_and_scores,
                output_scores=output_hidden_states_and_scores,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                bos_token_id=input_ids[0][0],
                attention_mask=attention_mask,
            )
            outputs = decode_with_grammar(
                model,
                model_kwargs,
                input_ids,
                lr_grammar,
                max_length=max_length + input_len,
                output_hidden_states_and_scores=output_hidden_states_and_scores,
            )
    sample_outputs = outputs.sequences
    if output_hidden_states_and_scores:
        hidden_states = outputs.hidden_states
        scores = outputs.scores

        # hidden_states is
        #   tuple of size len(generated tokens)
        #   tuple of size 1 + num_layers
        #   Tensor of shape (batch_size * k, len(input tokens), embed_dim)
        stacked = [torch.stack(t, dim=1) for t in hidden_states]
        # stacked is
        #   list of len(generated tokens)
        #   Tensor of shape (batch_size * k, 1 + num_layers, len(input tokens), embed_dim)
        hidden_states = torch.stack(
            [
                torch.stack([s[idx, :, -1, :] for s in stacked], dim=0)
                for idx in range(stacked[0].shape[0])
            ],
            dim=0,
        )
        # hidden_states is
        #   Tensor of shape (batch_size, generated_tokens, 1 + num_layers, embed_dim)
    else:
        hidden_states = None
        scores = None

    input_len = len(inputs["input_ids"][0])
    sample_outputs = [s[input_len:] for s in sample_outputs]
    decoded_outputs = [
        "".join(tokenizer.convert_ids_to_tokens(outputs)) for outputs in sample_outputs
    ]
    del inputs

    if output_hidden_states_and_scores:
        logits = torch.stack(scores, dim=1)
        output_ids = torch.stack(sample_outputs, dim=0)
        # for batch in range(len(logits)):
        #    l = logits[batch]
        #    o = output_ids[batch]
        #    assert len(l) == len(o)
        #    for _l, _o in zip(l, o):
        #        if _o == eos_token_id:
        #            break
        #        assert _o == torch.argmax(_l)
        scores = get_scores(logits, output_ids).cpu()

        torch.cuda.empty_cache()
        return decoded_outputs, sample_outputs, hidden_states, scores
    else:
        del sample_outputs
        torch.cuda.empty_cache()
        return decoded_outputs


@torch.no_grad()
def force_decode(model, tokenizer, prompt, response, include_prompt=False):
    model.eval()
    device = next(model.parameters()).device
    assert isinstance(prompt, type(response))
    if not isinstance(prompt, list):
        prompt = [prompt]
        response = [response]

    pad_token_id = tokenizer.pad_token_id

    input_tokens = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
    input_ids = input_tokens["input_ids"]

    output_tokens = tokenizer(response, padding=True, return_tensors="pt").to(device)
    output_ids = output_tokens["input_ids"]

    all_input_ids = torch.cat([input_ids, output_ids], dim=-1)
    attention_mask = all_input_ids.ne(pad_token_id).long()

    model_outputs = model(
        input_ids=all_input_ids,
        attention_mask=attention_mask,
        labels=all_input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    all_scores = get_scores(model_outputs.logits, all_input_ids, offset=True)

    hidden_states = model_outputs.hidden_states
    # hidden_states is
    #   tuple of num_layers
    #   Tensor of shape (batch_size, all_input_ids length, embedding size)
    hidden_states = torch.stack(hidden_states, dim=2)
    # hidden_states is
    #   Tensor of shape (batch_size, all_input_ids length, num_layers, embedding size)

    prompt_len = input_ids.shape[1] - 1
    prompt_hidden_states, hidden_states = (
        hidden_states[:, :prompt_len, :, :],
        hidden_states[:, prompt_len:, :, :],
    )
    prompt_scores, scores = (
        all_scores[:, :prompt_len, :],
        all_scores[:, prompt_len:, :],
    )

    # assert hidden_states.shape[1] == output_ids.shape[1]
    # assert scores.shape[1] == output_ids.shape[1]
    # assert prompt_scores.shape[1] == prompt_len

    torch.cuda.empty_cache()
    if include_prompt:
        return (
            hidden_states,
            scores,
            output_ids,
            prompt_hidden_states,
            prompt_scores,
            input_ids,
        )
    else:
        return hidden_states, scores, output_ids
