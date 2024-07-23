import argparse

from lm.eval import make_semantic_dataset, make_meta_dataset, eval_dataset
from probe.alt import eval_with_config as make_alt_dataset
from probe.train import train_with_config as probe_train
from utils.cache import CACHE
from utils.config import Config


# TODO read this from the model config?
NUM_LAYERS = 21


def parse_args():
    parser = argparse.ArgumentParser(description="Make semantic dataset and test.")
    Config.add_train_args(parser)
    Config.add_eval_args(parser)

    parser.add_argument(
        "--eval_mode",
        default="intervention",
        choices=[
            "intervention",
            "causal",
        ],
    )

    parser.add_argument("--intervention_alt", type=int, default=None)
    parser.add_argument("--intervention_spec", action="store_true")
    parser.add_argument(
        "--intervention_inactive_mode",
        default=None,
        choices=["wall", "space", "input"],
    )
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument(
        "--num_train_steps", type=int, default=None, help="Number of training steps."
    )
    parser.add_argument(
        "--semantic_train_batch_size",
        type=int,
        default=256,
        help="Batch size (per device) for the train dataloader.",
    )
    parser.add_argument(
        "--semantic_eval_batch_size",
        type=int,
        default=1024,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--make_semantic_ds_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="If the evaluation should try to resume first.",
    )
    parser.add_argument(
        "--mlp_icml_scheduler",
        action="store_true",
        help="Whether to use the ICML scheduler for the MLP probe.",
    )
    parser.add_argument(
        "--mlp_layernorm",
        action="store_true",
        help="Whether to use layer norm for the MLP probe.",
    )
    parser.add_argument(
        "--filter_lengths",
        nargs="+",
        type=int,
        default=None,
        help="Which lengths to keep when training MLP probe.",
    )
    parser.add_argument(
        "--all_labels",
        action="store_true",
        help="Whether to label all inputs (if false, label only first input).",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print()
    print(args)

    eval_mode = args.eval_mode
    num_train_steps = args.num_train_steps
    semantic_train_batch_size = args.semantic_train_batch_size
    semantic_eval_batch_size = args.semantic_eval_batch_size
    make_semantic_ds_batch_size = args.make_semantic_ds_batch_size
    resume = args.resume
    alt = args.intervention_alt
    alt_spec = args.intervention_spec
    inactive_mode = args.intervention_inactive_mode

    config = Config(args)
    del args

    hidden_state_layer = config.hidden_state_layer
    max_eval_samples = config.max_eval_samples

    eval_alt_actives = [None]
    if eval_mode in ["generate", "force_decode"]:
        if eval_mode == "force_decode":
            forced_decoding = True
            grammar_decoding = [False]
        else:
            forced_decoding = False
            grammar_decoding = [True, False]

        # for split in ["test", "train"]:
        for split in ["test"]:
            for gd in grammar_decoding:
                for eval_alt_active in eval_alt_actives:
                    config = config.update(
                        split=split,
                        grammar_decoding=gd,
                        forced_decoding=forced_decoding,
                        eval_alt_active=eval_alt_active,
                    )
                    print(
                        f"eval_dataset for "
                        f"{split=} forced_decoding=False "
                        f"{grammar_decoding=} {eval_alt_active=}"
                    )
                    if eval_mode == "generate":
                        eval_dataset(
                            config,
                            make_semantic_ds_batch_size,
                            max_eval_samples,
                            resume=resume,
                        )
                    else:
                        make_semantic_dataset(
                            config,
                            make_semantic_ds_batch_size,
                            max_eval_samples,
                            resume=resume,
                            generate_trace=True,
                        )
                    print()
        return

    eval_alt_actives = [None]

    mlp_icml_scheduler = False
    mlp_layernorm = False
    dropout_p = None
    
    filter_lengths = [config.filter_lengths]
    hidden_state_layers = [hidden_state_layer]
    seeds = [0, 1, 2, 3, 4]

    if eval_mode == "intervention":
        # ICML'24: semantic probing interventions
        forced_decoding = False
        if config.use_alt is not None:
            eval_alt_actives = ["all"]
        offsets = [0, 3, 6, -3, -6]
        eval_inactive_modes = [None]

        alt_idxs = [None, 0, 3, 5, 8, 11]

        assert not alt_spec
        eval_alt_spec = False
        eval_num_active_examples = 5
        drop_last = True
    elif eval_mode == "causal":
        # COLM'24: latent causal probing
        forced_decoding = True
        assert config.use_alt is None
        offsets = [-1, -2, -3, -6, -9]
        eval_inactive_modes = ["input", "space"]

        alt_idxs = [None, 8, 11]

        assert config.filter_lengths is None
        filter_lengths = [
            [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
        ]
        eval_alt_spec = alt_spec
        eval_num_active_examples = 0
        drop_last = False

    config = config.update(
        forced_decoding=forced_decoding,
        grammar_decoding=not forced_decoding,
        eval_num_active_examples=eval_num_active_examples,
        eval_alt_spec=eval_alt_spec,
    )

    if hidden_state_layers is None:
        if hidden_state_layer == "full":
            hidden_state_layers = list(range(NUM_LAYERS))
            hidden_state_layers.append(None)
        elif hidden_state_layer is None:
            hidden_state_layers = [None]
        else:
            hidden_state_layers = [int(hidden_state_layer)]

    def make_semantic_ds(config):
        for split in ["test", "train", "val"]:
            config = config.update(
                split=split,
                eval_alt_active=eval_alt_active,
            )
            print(
                f"make_semantic_dataset for "
                f"{split=} {config.forced_decoding=} "
                f"{config.hidden_state_layer=} "
                f"{config.eval_alt_active=} {config.eval_inactive_mode=}"
            )
            make_semantic_dataset(
                config,
                make_semantic_ds_batch_size,
                max_eval_samples,
                resume=resume,
                generate_trace=True,
            )
            print()
            print(
                f"make_meta_dataset for "
                f"{split=} {config.forced_decoding=} "
                f"{config.hidden_state_layer=} "
                f"{config.eval_alt_active=} {config.eval_inactive_mode=}"
            )
            make_meta_dataset(config)
            print()

    def make_alt_ds(config):
        for split in ["test", "train", "val"]:
            config = config.update(split=split)
            print(
                f"make_alt_dataset for "
                f"{config.alt_idx=} {split=} "
                f"{config.hidden_state_layer=} "
                f"{config.eval_alt_active=} {config.eval_inactive_mode=}"
            )
            make_alt_dataset(
                config,
                skip_if_exists=resume,
            )
            print()

    def semantic_train_all(config):
        config = config.update(split=None)
        for offset in offsets:
            for seed in seeds:
                config = config.update(
                    offset=offset,
                    drop_last=drop_last,
                    mlp_layers=None,
                    mlp_icml_scheduler=mlp_icml_scheduler,
                    mlp_layernorm=mlp_layernorm,
                    seed=seed,
                )
                print(
                    f"semantic_train for "
                    f"{config.alt_idx=} {offset=} "
                    f"{drop_last=} {mlp_icml_scheduler=} "
                    f"{config.hidden_state_layer=} {seed=} "
                    f"{config.eval_alt_active=} "
                    f"{config.eval_inactive_mode=}"
                )
                probe_train(
                    config,
                    train_split="train",
                    # eval_split="train",  # Eval dataset is split from train
                    eval_split="test",
                    train_batch_size=semantic_train_batch_size,
                    eval_batch_size=semantic_eval_batch_size,
                    num_train_steps=num_train_steps,
                    skip_if_exists=resume,
                    dropout_p=dropout_p,
                    # train_samples=None,  # 100000,
                    # eval_samples=None,  # 10000,
                    # val_samples=None,  # 10000,
                )
                print()

    for eval_inactive_mode in eval_inactive_modes:
        config = config.update(eval_inactive_mode=eval_inactive_mode)
        if eval_inactive_mode == None:
            config = config.update(
                eval_num_active_examples=5,
            )
        else:
            config = config.update(
                eval_num_active_examples=eval_num_active_examples,
            )
        if eval_inactive_mode == "alt":
            config = config.update(eval_inactive_alt=alt)
            print(f"using eval_inactive_alt={alt}")

        for hidden_state_layer in hidden_state_layers:
            config = config.update(hidden_state_layer=hidden_state_layer)
            if eval_inactive_mode != "alt" or alt != None:
                for eval_alt_active in eval_alt_actives:
                    config = config.update(eval_alt_active=eval_alt_active)
                    make_semantic_ds(config)

            for alt_idx in alt_idxs:
                for eval_alt_active in eval_alt_actives:
                    for fl in filter_lengths:
                        config = config.update(
                            alt_idx=alt_idx,
                            eval_alt_active=eval_alt_active,
                            filter_lengths=fl,
                        )

                        if eval_inactive_mode == "alt" and alt == None:
                            config = config.update(eval_inactive_alt=alt_idx)
                            if alt_idx == None:
                                config = config.update(
                                    eval_num_active_examples=5,
                                )
                            else:
                                config = config.update(
                                    eval_num_active_examples=eval_num_active_examples,
                                )
                            make_semantic_ds(config)

                        if alt_idx is not None:
                            make_alt_ds(config)

                        semantic_train_all(config)

                if alt_idx is not None:
                    for split in ["train", "test", "val"]:
                        config = config.update(split=split)
                        path = config.alt_semantic_ds_path
                        CACHE.evict(path)

        for split in ["train", "test", "val"]:
            config = config.update(split=split)
            path = config.semantic_ds_path
            CACHE.evict(path)


if __name__ == "__main__":
    main()
