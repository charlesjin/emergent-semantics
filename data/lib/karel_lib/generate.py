#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd

from sample_progs import generate_random_with_distribution

# from karel import KarelWithCurlyConfigurableParser, KarelForSynthesisConfigurableParser
from karel import KarelWithCurlyParser, KarelForSynthesisParser
from karel import str2bool, makedirs, pprint, beautify, TimeoutError


if __name__ == "__main__":
    data_arg = argparse.ArgumentParser()
    data_arg.add_argument("--num_train", type=int, default=1000000)
    data_arg.add_argument("--num_test", type=int, default=10000)
    data_arg.add_argument("--num_val", type=int, default=10000)
    data_arg.add_argument("--num_examples", type=int, default=10)
    data_arg.add_argument("--num_train_examples", type=int, default=5)
    data_arg.add_argument(
        "--parser_type", type=str, default="curly", choices=["curly", "synthesis"]
    )
    data_arg.add_argument("--no_loops", action="store_true")
    data_arg.add_argument("--no_conditionals", action="store_true")
    data_arg.add_argument("--no_markers", action="store_true")
    data_arg.add_argument("--no_noops", action="store_true")
    data_arg.add_argument("--uniform", action="store_true")
    data_arg.add_argument("--data_dir", type=str, default="karel")
    data_arg.add_argument("--max_length", type=int, default=10)
    data_arg.add_argument("--min_length", type=int, default=1)
    data_arg.add_argument("--beautify", type=str2bool, default=True)
    data_arg.add_argument(
        "--world_height", type=int, default=8, help="Height of square grid world"
    )
    data_arg.add_argument(
        "--world_width", type=int, default=8, help="Width of square grid world"
    )
    config = data_arg.parse_args()

    stmt_weights = {
        "pick_marker": 2,
        "put_marker": 2,
        "move": 5,
        "turn_right": 3,
        "turn_left": 3,
        "if": 1,
        "ifelse": 1,
    }

    # Make directories
    data_dir = os.path.join("data", config.data_dir)
    if config.uniform:
        data_dir += "_uniform"
        for key in stmt_weights:
            stmt_weights[key] = 1
    if config.no_loops:
        data_dir += "_noloops"
    else:
        assert False, "loops not yet supported"
    if config.no_conditionals:
        data_dir += "_nocond"
        stmt_weights["if"] = 0
        stmt_weights["ifelse"] = 0
    if config.no_markers:
        data_dir += "_nomarks"
        stmt_weights["put_marker"] = 0
        stmt_weights["pick_marker"] = 0
    if config.no_noops:
        data_dir += "_nonoops"

    makedirs(data_dir)
    datasets = ["train", "test", "val"]

    # Generate datasets
    _parser = {"curly": KarelWithCurlyParser, "synthesis": KarelForSynthesisParser}
    parser = _parser[config.parser_type]()

    def draw_string(parser):
        state = parser.draw(no_print=True)
        assert state is not None
        return "\n".join(state)

    world_size = (config.world_width, config.world_height)

    assert config.min_length <= config.max_length
    for name in datasets:
        data_num = getattr(config, "num_{}".format(name))
        distribution = [
            data_num // (config.max_length - config.min_length + 1)
        ] * config.max_length
        for idx in range(config.min_length - 1):
            distribution[idx] = 0

        text = ""
        records = []
        np_records = []

        progs = generate_random_with_distribution(
            distribution, no_noops=config.no_noops, stmt_weights=stmt_weights
        )
        for code, record, np_record in progs:
            if config.beautify:
                code = beautify(code)
            text += code + "\n"
            record["code"] = code
            records.append(record)
            np_records.append(np_record)

        save_path = os.path.join(data_dir, name)

        df = pd.DataFrame.from_records(records)
        df.to_json(f"{save_path}.jsonl", orient="records", lines=True)

        with open(f"{save_path}.txt", "w") as f:
            f.write(text)

        np.savez(f"{save_path}.npz", records=np_records)
