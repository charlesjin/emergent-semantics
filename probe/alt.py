import os
import argparse
from tqdm import tqdm

import torch

from data import karel
from utils.cache import CACHE
from utils.config import Config
import warnings

warnings.simplefilter("ignore", UserWarning)

TOTAL_EXAMPLES = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Make an alt semantic dataset")
    Config.add_train_args(parser)
    Config.add_eval_args(parser)

    parser.add_argument(
        "--alternative",
        type=int,
        help="Which alternative semantics to use.",
        required=True,
    )

    parser.add_argument(
        "--eval_num_active_examples",
        type=int,
        default=None,
        help="Number of active examples in eval spec.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip if dataset already exists.",
    )
    args = parser.parse_args()

    return args


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


def semantics_transformer(prog, idx):
    for t in karel.ACTION_TOKENS + karel.CONDITIONAL_TOKENS:
        prog = prog.replace(t, f"---{t}---")

    transformer = {
        0: opposite1,
        1: random1,
        2: perm1,
        3: unsemantic1,
        4: unsemantic2,
        5: unsemantic3,
        6: unsemantic4,
        7: perm_nomarks_1,
        8: perm_nomarks_2,
        9: perm_nomarks_3,
        10: perm_nomarks_4,
        11: perm_nomarks_5,
    }[idx]
    prog = transformer(prog)

    assert "---" not in prog, f"--- in {prog}"
    return prog


def opposite1(prog):
    prog = prog.replace("---pick_marker()---", "put_marker()")
    prog = prog.replace("---put_marker()---", "pick_marker()")
    prog = prog.replace("---turn_right()---", "turn_left()")
    prog = prog.replace("---turn_left()---", "turn_right()")
    prog = prog.replace("---move()---", "move()")

    # prog = prog.replace("---front_is_clear()---", "front_is_clear()")
    # prog = prog.replace("---left_is_clear()---", "right_is_clear())")
    # prog = prog.replace("---right_is_clear()---", "left_is_clear()")
    # prog = prog.replace("---markers_present()---", "no_markers_present()")
    # prog = prog.replace("---no_markers_present()---", "markers_present()")

    return prog


def random1(prog):
    prog = prog.replace("---pick_marker()---", "turn_left()")
    prog = prog.replace("---put_marker()---", "move()")
    prog = prog.replace("---turn_right()---", "pick_marker()")
    prog = prog.replace("---turn_left()---", "put_marker()")
    prog = prog.replace("---move()---", "turn_right()")

    # prog = prog.replace("---front_is_clear()---", "no_markers_present()")
    # prog = prog.replace("---left_is_clear()---", "right_is_clear()")
    # prog = prog.replace("---right_is_clear()---", "front_is_clear")
    # prog = prog.replace("---markers_present()---", "left_is_clear()")
    # prog = prog.replace("---no_markers_present()---", "markers_present()")

    return prog


def perm1(prog):
    prog = prog.replace("---pick_marker()---", "put_marker()")
    prog = prog.replace("---put_marker()---", "turn_right()")
    prog = prog.replace("---turn_right()---", "turn_left()")
    prog = prog.replace("---turn_left()---", "move()")
    prog = prog.replace("---move()---", "pick_marker()")

    # prog = prog.replace("---front_is_clear()---", "left_is_clear()")
    # prog = prog.replace("---left_is_clear()---", "right_is_clear()")
    # prog = prog.replace("---right_is_clear()---", "markers_present")
    # prog = prog.replace("---markers_present()---", "no_markers_present()")
    # prog = prog.replace("---no_markers_present()---", "front_is_clear()")

    return prog


def unsemantic1(prog):
    prog = prog.replace("---pick_marker()---", "turn_right()")
    prog = prog.replace("---put_marker()---", "turn_left()")
    prog = prog.replace("---turn_right()---", "move()")
    prog = prog.replace("---turn_left()---", "turn_right()")
    prog = prog.replace("---move()---", "turn_left()")

    # prog = prog.replace("---front_is_clear()---", "front_is_clear()")
    # prog = prog.replace("---left_is_clear()---", "right_is_clear())")
    # prog = prog.replace("---right_is_clear()---", "left_is_clear()")
    # prog = prog.replace("---markers_present()---", "no_markers_present()")
    # prog = prog.replace("---no_markers_present()---", "markers_present()")

    return prog


def unsemantic2(prog):
    prog = prog.replace("---pick_marker()---", "turn_right()")
    prog = prog.replace("---put_marker()---", "move()")
    prog = prog.replace("---turn_right()---", "put_marker()")
    prog = prog.replace("---turn_left()---", "pick_marker()")
    prog = prog.replace("---move()---", "turn_left()")

    # prog = prog.replace("---front_is_clear()---", "front_is_clear()")
    # prog = prog.replace("---left_is_clear()---", "right_is_clear())")
    # prog = prog.replace("---right_is_clear()---", "left_is_clear()")
    # prog = prog.replace("---markers_present()---", "no_markers_present()")
    # prog = prog.replace("---no_markers_present()---", "markers_present()")

    return prog


def unsemantic3(prog):
    prog = prog.replace("---pick_marker()---", "put_marker()")
    prog = prog.replace("---put_marker()---", "pick_marker()")
    prog = prog.replace("---turn_right()---", "move()")
    prog = prog.replace("---turn_left()---", "turn_right()")
    prog = prog.replace("---move()---", "turn_left()")

    # prog = prog.replace("---front_is_clear()---", "front_is_clear()")
    # prog = prog.replace("---left_is_clear()---", "right_is_clear())")
    # prog = prog.replace("---right_is_clear()---", "left_is_clear()")
    # prog = prog.replace("---markers_present()---", "no_markers_present()")
    # prog = prog.replace("---no_markers_present()---", "markers_present()")

    return prog


def unsemantic4(prog):
    prog = prog.replace("---pick_marker()---", "put_marker()")
    prog = prog.replace("---put_marker()---", "pick_marker()")
    prog = prog.replace("---turn_right()---", "move()")
    prog = prog.replace("---turn_left()---", "turn_right()")
    prog = prog.replace("---move()---", "turn_left()")

    # prog = prog.replace("---front_is_clear()---", "front_is_clear()")
    # prog = prog.replace("---left_is_clear()---", "right_is_clear())")
    # prog = prog.replace("---right_is_clear()---", "left_is_clear()")
    # prog = prog.replace("---markers_present()---", "no_markers_present()")
    # prog = prog.replace("---no_markers_present()---", "markers_present()")

    return prog


def perm_nomarks_1(prog):
    prog = prog.replace("---pick_marker()---", "put_marker()")
    prog = prog.replace("---put_marker()---", "pick_marker()")
    prog = prog.replace("---turn_right()---", "turn_left()")
    prog = prog.replace("---turn_left()---", "turn_right()")
    prog = prog.replace("---move()---", "move()")
    return prog


def perm_nomarks_2(prog):
    prog = prog.replace("---pick_marker()---", "put_marker()")
    prog = prog.replace("---put_marker()---", "pick_marker()")
    prog = prog.replace("---turn_right()---", "turn_right()")
    prog = prog.replace("---turn_left()---", "move()")
    prog = prog.replace("---move()---", "turn_left()")
    return prog


def perm_nomarks_3(prog):
    prog = prog.replace("---pick_marker()---", "put_marker()")
    prog = prog.replace("---put_marker()---", "pick_marker()")
    prog = prog.replace("---turn_right()---", "move()")
    prog = prog.replace("---turn_left()---", "turn_left()")
    prog = prog.replace("---move()---", "turn_right()")
    return prog


def perm_nomarks_4(prog):
    prog = prog.replace("---pick_marker()---", "put_marker()")
    prog = prog.replace("---put_marker()---", "pick_marker()")
    prog = prog.replace("---turn_right()---", "move()")
    prog = prog.replace("---turn_left()---", "turn_right()")
    prog = prog.replace("---move()---", "turn_left()")
    return prog


def perm_nomarks_5(prog):
    prog = prog.replace("---pick_marker()---", "put_marker()")
    prog = prog.replace("---put_marker()---", "pick_marker()")
    prog = prog.replace("---turn_right()---", "turn_left()")
    prog = prog.replace("---turn_left()---", "move()")
    prog = prog.replace("---move()---", "turn_right()")
    return prog


def eval():
    args = parse_args()
    print()
    print(args)
    config = Config(args)

    return eval_with_config(config, skip_if_exists)


@torch.no_grad()
def eval_with_config(config, skip_if_exists):
    alt_idx = config.alt_idx

    semantic_ds_dir = config.semantic_dir
    os.makedirs(semantic_ds_dir, exist_ok=True)
    semantic_ds_path = config.semantic_ds_path
    semantic_alt_ds_path = config.alt_semantic_ds_path
    os.makedirs(os.path.dirname(semantic_alt_ds_path), exist_ok=True)

    eval_dataset = config.eval_dataset
    if not eval_dataset:
        eval_dataset = config.dataset

    if skip_if_exists:
        try:
            if CACHE.load(semantic_alt_ds_path) is not None:
                print(f"{semantic_alt_ds_path} already exists, skipping.")
                return
        except:
            pass

    semantic_ds = CACHE.load(semantic_ds_path)
    semantic_alt_ds = []

    state_dataset = karel.load_karel_raw(
        config.split, data_folder=eval_dataset, data_for="state"
    )["records"]

    try:
        os.remove(semantic_alt_ds_path)
    except OSError:
        pass

    print(
        f"Regenerating traces for {semantic_ds_path} using alternative "
        f"semantics {alt_idx} to {semantic_alt_ds_path}."
    )

    pbar = tqdm(semantic_ds, total=len(semantic_ds), leave=True)
    first = True
    for example in pbar:
        prog = example["text"]
        idx = example["idx"]
        active_examples = example.get("active", None)

        state_examples = state_dataset[idx]
        spec_examples = [
            (state_examples[f"input{i}"], state_examples[f"output{i}"])
            for i in range(TOTAL_EXAMPLES)
        ]
        if first:
            print("orig")
            print(prog)
            print(example["trace"][0])

        prog = semantics_transformer(prog, alt_idx)
        results = karel.semantic_eval(
            prog,
            spec_code=None,
            spec_examples=spec_examples,
            generate_trace=True,
            mode="synthesis",
        )
        if first:
            print("alt")
            print(prog)
            print(results["trace"][0])
        first = False

        semantic_alt_ds.append(
            {
                "idx": idx,
                "text": prog,
                "np_trace": results["np_trace"],
                "trace": results["trace"],
            }
        )

    torch.save(semantic_alt_ds, semantic_alt_ds_path)
    CACHE.insert(semantic_alt_ds_path, semantic_alt_ds)
    return semantic_alt_ds


if __name__ == "__main__":
    eval()
