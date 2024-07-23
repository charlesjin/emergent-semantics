from itertools import chain
import numpy as np
from random import sample as rand_sample
import copy

import torch
from torch import nn
from torch.utils.data import Dataset

from data.karel import ACTION_TOKENS, CONDITIONAL_TOKENS
from utils.cache import CACHE
from probe.alt import semantics_transformer


# https://stackoverflow.com/questions/32954486/zip-iterators-asserting-for-equal-length-in-python/69485272#69485272
def zip_equal(*iterables):
    # For trivial cases, use pure zip.
    if len(iterables) < 2:
        return zip(*iterables)

    # Tail for the first iterable
    first_stopped = False

    def first_tail():
        nonlocal first_stopped
        first_stopped = True
        return
        yield

    # Tail for the zip
    def zip_tail():
        if not first_stopped:
            raise ValueError("zip_equal: first iterable is longer")
        for _ in chain.from_iterable(rest):
            raise ValueError("zip_equal: first iterable is shorter")
            yield

    # Put the pieces together
    iterables = iter(iterables)
    first = chain(next(iterables), first_tail())
    rest = list(map(iter, iterables))
    return chain(zip(first, *rest), zip_tail())


def load_dataset(
    config,
    tokenizer,
    filter_correct,
    filter_active,
    filter_inactive,
    filter_lengths=None,
    drop_last_state=False,
    label_relative_direction=True,
    label_facing=True,
    label_position=True,
    label_grid=True,
    max_samples=None,
    single_label=False,
):
    """Returns a list of (hidden_state, label).

    If offset >= 0, then predict the future.
    If offset < 0, then predict the past.
    By convention, at offset=0, we align the program state after applying the
    token with the model state that generated that token (predicting the
    future).

    """
    dataset = config.dataset
    offset = config.offset
    num_examples = config.num_examples

    ds_path = config.semantic_ds_path
    print(f"Loading semantic eval dataset from `{ds_path}`.")

    hidden_state_layer = config.hidden_state_layer
    ds = CACHE.load(ds_path, max_length=max_samples)
    if ds is not None:
        hidden_state_layer = None
        alt_ds_path = config.alt_semantic_ds_path

    elif config.hidden_state_layer is not None:
        ds_config = config.update(hidden_state_layer="full")
        ds_path = ds_config.semantic_ds_path
        print(f"Last load failed, loading semantic eval dataset from `{ds_path}`.")

        ds = CACHE.load(ds_path, max_length=max_samples)
        alt_ds_path = ds_config.alt_semantic_ds_path

    assert ds is not None

    if config.alt_idx is not None:
        if not "nocond" in dataset or not "noloops" in dataset:
            raise ValueError("Don't support conditionals and loops yet.")

        print(f"Loading alt semantic eval dataset from `{alt_ds_path}`.")
        alt_ds = CACHE.load(alt_ds_path, max_length=max_samples)
        assert alt_ds is not None

        idx_to_alt = {d["idx"]: d for d in alt_ds}
        to_delete = []
        for idx in range(len(ds)):
            old = ds[idx]
            old_idx = old["idx"]
            if not old_idx in idx_to_alt:
                to_delete.append(idx)
                continue
            new = idx_to_alt[old_idx]
            for old_trace, new_trace in zip_equal(
                old["np_trace"][:num_examples], new["np_trace"][:num_examples]
            ):
                if old_trace is None or new_trace is None:
                    if old_trace is not None or new_trace is not None:
                        print(
                            f"{old_trace is None} {old['idx']} {old['text']} | "
                            f"{new_trace is None} {new['idx']} {new['text']}"
                        )
                    break
                assert (
                    old_trace.keys() == new_trace.keys()
                ), f"{old_trace.keys()} {old['idx']} {old['text']} | {new_trace.keys()} {new['idx']} {new['text']}"
            else:
                old["np_trace"] = new["np_trace"]
                continue
            to_delete.append(idx)
        for idx in reversed(to_delete):
            del ds[idx]
        del alt_ds

        move_tokens = []
        for action_tok in ACTION_TOKENS:
            transformed_tok = semantics_transformer(action_tok, config.alt_idx)
            if transformed_tok == "move()":
                move_token = tokenizer(action_tok)["input_ids"]
                assert len(move_token) == 1
                move_tokens.append(move_token[0])
    else:
        move_token = tokenizer("move()")["input_ids"]
        assert len(move_token) == 1
        move_tokens = [move_token[0]]

    action_tokens = [tokenizer(action)["input_ids"] for action in ACTION_TOKENS]
    assert all(len(a) == 1 for a in action_tokens)
    action_tokens = [a[0] for a in action_tokens]
    keep_tokens = list(action_tokens)
    # noop_tokens = [tokenizer(t)["input_ids"][0] for t in ACTION_TOKENS if "marker" in t]
    noop_tokens = []

    num_classes = [4]  # up down left right

    if label_relative_direction:
        # greater, less than or equal
        # left, right, up, down
        num_classes += [[13, 13], [13, 13]]

    if label_facing:
        # wall or not wall
        # forward (in facing dir)
        num_classes += [2]

    if label_position:  # position in grid
        # trace = ds[0]["np_trace"][0]
        # state = trace[(0,)][0]
        # dim_y, dim_x, _ = state.shape
        # num_classes += [[dim_y, dim_x]]
        num_classes += [[6, 6]]
        # num_classes += [2] * dim_y * dim_x

    if label_grid:
        # wall or not wall
        # left, right, up, down
        num_classes += [[2, 2, 2, 2]]

    conditional_tokens = []
    has_bool = "nocond" not in dataset
    if has_bool:
        num_classes += [2]
        conditional_tokens = [
            tokenizer(conditional)["input_ids"] for conditional in CONDITIONAL_TOKENS
        ]
        assert all(len(c) == 1 for c in conditional_tokens)
        conditional_tokens = [c[0] for c in conditional_tokens]
        keep_tokens += conditional_tokens

    has_loops = "noloops" not in dataset
    if has_loops:
        num_classes += [11]  # integers 0-10

    print(f"{num_classes=}")

    if filter_correct:
        filter_fn = lambda x: all(x["results"][:num_examples])
        filtered_ds = list(filter(filter_fn, ds))
        print(f"Filtered out {len(ds) - len(filtered_ds)} incorrect examples.")
    else:
        filtered_ds = ds

    if offset is None:
        offset = 0

    def offset_list(lst, offset):
        """
        If offset is positive, then remove from the beginning.
        Otherwise, remove from the end.

        For instance, to use A to predict B one additional step into the future,
        we would want to align
            a_0 with b_1
            a_1 with b_2
            ...
            a_{k-1} with b_k
            ...

        This can be accomplished via
            zip_equal(offset_list(A, -1), offset_list(B, 1))
        """

        if offset == 0:
            return lst
        if offset > 0:
            return lst[offset:]
        return lst[:offset]

    make_list = lambda x: [x] if not isinstance(x, list) else x

    def label_to_tensor(labels):
        return [torch.LongTensor([make_list(_l) for _l in l]) for l in zip(*labels)]

    if drop_last_state:
        # TODO change the name to drop_first
        maybe_drop_last = lambda lst: lst[:-1]
    else:
        maybe_drop_last = lambda lst: lst

    def get_pos(state):
        y, x, facing_idx = zip(*np.where(state[:, :, :4] == 1)).__next__()
        return x, y, facing_idx

    def get_grid_label(state, start_x, start_y, end_x, end_y, token=None):
        labels = []

        facing = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0),
        }
        x, y, facing_idx = get_pos(state)
        labels.append(facing_idx)

        if label_relative_direction:
            # if start_x == end_x and start_y == end_y:
            #    labels.extend([[-1, -1], [-1, -1]])
            # else:
            labels.append([x - start_x + 6, y - start_y + 6])
            labels.append([x - end_x + 6, y - end_y + 6])

        if label_facing:
            # if token in move_tokens:
            dx, dy = facing[facing_idx]
            idx = state[y + dy, x + dx, 4]
            assert idx in [0, 1]
            labels.append(idx)
            # else:
            #    labels.append(-1)

        if label_position:
            # pos_labels = [0] * (dim_x * dim_y)
            # pos_labels[y * dim_y + x] = 1
            # labels.extend(pos_labels)
            # labels.append(x * dim_y + y)
            labels.append([x - 1, y - 1])

        if label_grid:
            label = []
            for dx in [-1, 1]:
                idx = state[y, x + dx, 4]
                assert idx in [0, 1]
                label.append(idx)
            for dy in [-1, 1]:
                idx = state[y + dy, x, 4]
                assert idx in [0, 1]
                label.append(idx)
            labels.append(label)

        return labels

    def traces_to_label(traces, key, token, active):  # , inputs, outputs):
        """
        traces is a list of num_examples traces
            Each trace is a dictionary of keys
            Each entry in the key is a single-element list (for backward compat)
            The only element in the list is the np.array representing the program state.
        key identifies a state in the program trace.
        token is the program token that left a state in the trace.
        active is a list of length num_examples, corresponding to the active examples.
        # inputs is the input state in the spec
        # outputs is the output state in the spec
        """
        label = []
        # for trace, inp, out in zip_equal(traces, inputs, outputs):
        for trace, a in zip_equal(traces, active):
            if filter_active and not a:
                continue
            if filter_inactive and a:
                continue
            # TODO this should be of the right shape
            if not key in trace:
                label.append(-1)
                continue

            # assert trace[(0,)][0] == inp
            inp = trace[(0,)][0]
            start_x, start_y, _ = get_pos(inp)
            out = trace[max(trace.keys())][0]
            end_x, end_y, _ = get_pos(out)

            state = trace[key]
            assert len(state) == 1
            state = state[0]

            if token in action_tokens:
                l = get_grid_label(state, start_x, start_y, end_x, end_y, token)
                if len(l) < len(num_classes):
                    assert len(l) == len(num_classes) - 1
                    # For the conditional branch prediction
                    l.append(-1)
            elif token in conditional_tokens:
                assert isinstance(state, bool)
                l = [-1] * (len(num_classes) - 1)
                l.append(int(state))
            else:
                assert False
            # assert sum([_l >= 0 for _l in l]) > 0
            assert len(l) == len(num_classes)
            label.append(l)
            if single_label:
                break
        return label_to_tensor(label)

    def make_state_and_labels(sample):
        # TODO get the intended end state as well
        traces = sample["np_trace"][:num_examples]
        if traces[0] is None:
            if not all(t is None for t in traces):
                print("Sample has partial failed traces:\n " f"{sample['text']}")
            return []

        # print(sample.keys())
        # inputs = [sample[f"input{idx}"] for idx in range(num_examples)]
        # outputs = [sample[f"output{idx}"] for idx in range(num_examples)]

        keys = set()
        for trace in traces:
            keys |= trace.keys()
        keys = sorted(keys)
        keys.remove((0,))

        code_tokens = sample.get("code_tokens", sample.get("tokens"))
        # No longer needed since we are loading the original tokenizer
        # new_code_tokens = tokenizer(text)["input_ids"]
        # assert len(old_code_tokens) == len(new_code_tokens)
        # code_tokens = new_code_tokens

        traj = list(zip_equal(sample["hidden_states"], code_tokens))
        if hidden_state_layer is not None:
            get_layer = lambda x: x[hidden_state_layer]
        else:
            get_layer = lambda x: x

        _keys = []
        for _, t in traj:
            if t in keep_tokens:
                _keys.append(keys.pop(0))
            else:
                _keys.append(None)
        assert not keys
        keys = _keys

        # If offset is positive...
        # We remove the last hidden state and the first token
        # So we align the next program state with the previous hidden state
        if offset:
            traj = [
                (hs, t)
                for (hs, _), (_, t) in zip_equal(
                    offset_list(traj, -offset), offset_list(traj, offset)
                )
            ]
            keys = offset_list(keys, offset)
        hs_and_token = [(get_layer(hs), t) for hs, t in traj if t in keep_tokens]
        hs_and_token = [
            (hs, t, idx, len(hs_and_token)) for idx, (hs, t) in enumerate(hs_and_token)
        ]
        keys = [k for k in keys if k is not None]

        if len(keys) != len(hs_and_token):
            print(f"{keys=}")
            print(f"{hs_and_token=}")

        if len(traces) != len(sample["active"]):
            print(f"{traces=}")
            print(f"{sample['active']=}")

        num_to_delete = 0
        for _, t, _, _ in reversed(hs_and_token):
            if t in noop_tokens:
                num_to_delete += 1
            else:
                break
        if num_to_delete:
            hs_and_token = hs_and_token[:-num_to_delete]
            keys = keys[:-num_to_delete]
        if not hs_and_token:
            return []

        out = [
            (hs, traces_to_label(traces, k, t, sample["active"]), idx, traj_len)
            for k, (hs, t, idx, traj_len) in zip_equal(
                maybe_drop_last(keys),
                maybe_drop_last(hs_and_token),
            )
            if not filter_lengths or idx in filter_lengths
        ]

        return out

        # if not offset:
        #    return out

        # return [
        #    (hs, label, idx, traj_len)
        #    for (hs, _, _, _), (_, label, idx, traj_len) in zip_equal(
        #        offset_list(out, -offset), offset_list(out, offset)
        #    )
        # ]

    mapped_ds = list(map(make_state_and_labels, filtered_ds))
    return mapped_ds, [make_list(n) for n in num_classes]


class SemanticKarelDataset(Dataset):
    def __init__(
        self,
        config,
        tokenizer,
        drop_last,
        filter_correct=True,
        filter_inactive=False,
        filter_active=False,
        filter_lengths=None,
        max_samples=-1,
        max_load_samples=None,
        single_label=True,
        flatten=True,
        mean=None,
        std=None,
        noise=None,
    ):
        if filter_active and filter_inactive:
            raise ValueError("Cannot filter both active and inactive: no samples left.")

        # if (config.mode == "synthesis" and config.forced_decoding) or (
        #    config.mode == "interp" and not config.forced_decoding
        # ):
        #    raise ValueError(
        #        f"Can't make semantic dataset with "
        #        f"{config.mode=} and {config.forced_decoding=}."
        #    )

        self.config = config
        self.tokenizer = tokenizer
        self.drop_last = drop_last
        self.filter_correct = filter_correct
        self.filter_active = filter_active
        self.filter_inactive = filter_inactive
        self.filter_lengths = filter_lengths
        self.max_samples = max_samples
        self.max_load_samples = max_load_samples
        self.single_label = single_label
        self.flatten = flatten
        self.task_idx = None
        self.mean = mean
        self.std = std
        self.noise = noise

        ds, self.num_classes = load_dataset(
            self.config,
            self.tokenizer,
            filter_correct=self.filter_correct,
            filter_active=self.filter_active,
            filter_inactive=self.filter_inactive,
            filter_lengths=self.filter_lengths,
            drop_last_state=drop_last,
            max_samples=max_load_samples,
            single_label=single_label,
        )
        if self.flatten:
            ds = list(chain(*ds))
        self._ds = ds
        self.resample()

    def get_mean_and_std(self):
        inputs = self.inputs.numpy()
        self.mean = inputs.mean(axis=0)
        self.std = inputs.std(axis=0)

    def normalize(self):
        if self.mean is None or self.std is None:
            self.get_mean_and_std()
        # self.inputs = torch.clamp(
        #    (self.inputs - self.mean) / self.std, min=-3.0, max=3.0
        # )
        self.inputs = (self.inputs - self.mean) / self.std

    def split(self, num_samples=None, p=None):
        if num_samples:
            if p:
                raise ValueError("Can't provide both num_samples and p")
            if num_samples >= len(self._ds):
                raise ValueError(
                    f"Can't split off {num_samples}, only have {len(self._ds)}."
                )
        else:
            if not p:
                raise ValueError("Must provide one of num_samples or p")
            if not 0 < p < 1:
                raise ValueError("p must be between 0 and 1.")
            num_samples = int(len(self._ds) * p)

        first = copy.copy(self)
        first._ds = first._ds[:num_samples]
        first.resample()

        second = copy.copy(self)
        second._ds = second._ds[num_samples:]
        second.resample()

        return first, second

    def resample(self, max_samples=None, random=False):
        if not max_samples:
            max_samples = self.max_samples
        if max_samples > 0 and len(self._ds) > max_samples:
            if random:
                ds = rand_sample(self._ds, max_samples)
                sample_method = "random"
            else:
                ds = self._ds[:max_samples]
                sample_method = "first"
        else:
            ds = self._ds
            sample_method = "all"

        self.inputs, self.labels, self.idxs, self.lengths = zip(*ds)
        self.inputs = torch.stack(self.inputs)

        self.normalize()

        print(
            f"Created semantic trace dataset with {sample_method} "
            f"{len(self)} examples."
        )

    def set_task(self, task_idx):
        self.task_idx = task_idx

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.noise:
            noise = self.noise * torch.clamp(
                torch.randn(self.inputs[idx].shape) / 3,
                min=-1.0,
                max=1.0,
            )
        else:
            noise = 0

        if self.task_idx is None:
            return (
                self.inputs[idx] + noise,
                self.labels[idx],
                self.idxs[idx],
                self.lengths[idx],
            )
        else:
            return (
                self.inputs[idx] + noise,
                self.labels[idx][self.task_idx],
                self.idxs[idx],
                self.lengths[idx],
            )

    @property
    def input_shape(self):
        return self[0][0].shape

    @property
    def label_shape(self):
        return self[0][1].shape
