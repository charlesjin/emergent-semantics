import os
import argparse
import copy
import torch


class Config:
    def __init__(self, args):
        # train args
        parser = argparse.ArgumentParser()
        self.add_train_args(parser)
        default_args = parser.parse_args(
            ["--dataset_name", args.dataset_name, "--output_dir", args.output_dir]
        )
        for k, v in vars(default_args).items():
            if k not in args:
                setattr(args, k, v)
        if "config_name" in args:
            args.model_name = args.config_name

        self.output_dir = args.output_dir
        self.model_name = args.model_name
        self.from_scratch = True
        self.dataset = args.dataset_name
        self.num_examples = args.num_examples
        self.num_active_examples = args.num_active_examples
        self.randomize_active = args.randomize_active
        self.inactive_mode = args.inactive_mode
        self.inactive_alt = args.inactive_alt
        self.alt_spec = args.alt_spec
        self.use_alt = args.use_alt
        self.mode = args.mode
        self.debug = args.debug

        if self.use_alt is not None:
            if self.num_examples != 2:
                raise ValueError(
                    "If using alternative semantics, must set num_examples=2"
                )

        if self.alt_spec and self.inactive_alt is None:
            raise ValueError("Can't set alt_spec without inactive_alt")

        assert self.mode in ["synthesis", "interp"]

        if (
            self.inactive_alt is not None
            and self.inactive_mode != "alt"
            and not self.alt_spec
        ):
            raise ValueError(
                "Cannot provide inactive_alt without inactive_mode=alt or alt_spec=True"
            )
        elif self.inactive_mode == "alt" and self.inactive_alt is None:
            raise ValueError("Must provide eval_inactive_alt if eval_inactive_mode=alt")

        # eval args
        parser = argparse.ArgumentParser()
        self.add_eval_args(parser)
        default_args = parser.parse_args([])
        for k, v in vars(default_args).items():
            if k not in args:
                setattr(args, k, v)

        self.step = args.checkpoint_steps
        self.hidden_state_layer = args.hidden_state_layer
        self.max_eval_samples = args.max_eval_samples
        self.eval_dataset = args.eval_dataset_name
        self.eval_num_active_examples = args.eval_num_active_examples
        self.eval_randomize_active = args.eval_randomize_active
        self.eval_inactive_mode = args.eval_inactive_mode
        self.eval_inactive_alt = args.eval_inactive_alt
        self.eval_alt_spec = args.eval_alt_spec
        self.eval_alt_active = args.eval_alt_active
        self.split = args.split
        self.forced_decoding = args.forced_decode
        self.grammar_decoding = args.grammar_decode

        # if self.mode == "synthesis":
        assert not self.grammar_decoding or not self.forced_decoding
        if (
            self.eval_inactive_alt is not None
            and self.eval_inactive_mode != "alt"
            and self.eval_alt_spec is not None
        ):
            raise ValueError(
                "Cannot provide eval_inactive_alt without eval_inactive_mode=alt or eval_alt_spec=True"
            )

        if (
            self.eval_inactive_mode == "alt"
            and self.inactive_alt is None
            and self.eval_inactive_alt is None
        ):
            raise ValueError("Must provide eval_inactive_alt if eval_inactive_mode=alt")

        if (
            self.eval_alt_spec
            and self.eval_inactive_alt is None
            and self.inactive_alt is None
        ):
            raise ValueError(
                "Can't set eval_alt_spec without eval_inactive_alt (or inactive_alt)."
            )

        # semantic args
        parser = argparse.ArgumentParser()
        self.add_semantic_args(parser)
        default_args = parser.parse_args([])
        for k, v in vars(default_args).items():
            if k not in args:
                setattr(args, k, v)
        self.offset = args.offset
        self.alt_idx = args.alternative
        self.seed = args.seed
        self.mlp_layers = args.mlp_layers
        self.mlp_layernorm = args.mlp_layernorm
        self.mlp_icml_scheduler = args.mlp_icml_scheduler
        self.drop_last = args.drop_last
        self.filter_lengths = args.filter_lengths
        self.all_labels = args.all_labels

    @staticmethod
    def from_kwargs(dataset_name, output_dir, **kwargs):
        parser = argparse.ArgumentParser()
        Config.add_train_args(parser)
        Config.add_eval_args(parser)
        Config.add_semantic_args(parser)
        args = parser.parse_args(
            ["--dataset_name", dataset_name, "--output_dir", output_dir]
        )
        for k, v in kwargs.items():
            setattr(args, k, v)
        return Config(args)

    def update(self, **kwargs):
        config = copy.deepcopy(self)
        for k, v in kwargs.items():
            if not hasattr(config, k):
                raise ValueError(f"Cannot update config with k={k}.")
            setattr(config, k, v)
        return config

    def _semantic_fn(self):
        fn = []
        if self.hidden_state_layer is not None:
            fn.append(f"hs{self.hidden_state_layer}")
        if self.alt_idx is not None:
            fn.append(f"alt{self.alt_idx}")
        if self.inactive_mode == "alt":
            if self.eval_inactive_alt is not None:
                fn.append(f"inactive{self.eval_inactive_alt}")
        if self.use_alt is not None:
            if self.eval_alt_active is not None:
                fn.append(f"active{self.eval_alt_active}")
        if self.drop_last:
            fn.append("droplast")
        if self.forced_decoding:
            fn.append("forced")
        if self.grammar_decoding:
            fn.append("grammar")
        if self.offset:
            if self.offset < 0:
                fn.append(f"n{-self.offset}")
            else:
                fn.append(f"p{self.offset}")
        if self.mlp_layers > 1:
            fn.append(f"layers{self.mlp_layers}")
        if not self.mlp_layernorm:
            fn.append(f"nolayernorm")
        if self.mlp_icml_scheduler:
            fn.append(f"oldsched")
        if self.filter_lengths:
            fl = ",".join(map(str, self.filter_lengths))
            fn.append(f"filter{fl}")
        if self.all_labels:
            fn.append(f"alllabels")
        if self.split is not None:
            fn.append(self.split)
        if self.seed is not None:
            fn.append(f"seed{self.seed}")
        if self.debug:
            fn.append("debug")
        return "_".join(fn)

    def _data_fn(self, alt=False):
        fn = ["data"]
        if self.hidden_state_layer is not None:
            fn.append(f"hs{self.hidden_state_layer}")
        if alt and self.alt_idx is not None:
            fn.append(f"alt{self.alt_idx}")
        if self.use_alt is not None:
            if self.eval_alt_active is not None:
                fn.append(f"active{self.eval_alt_active}")
        if self.forced_decoding:
            fn.append("forced")
        if self.grammar_decoding:
            fn.append("grammar")
        if self.split is not None:
            fn.append(self.split)
        return "_".join(fn)

    @staticmethod
    def _ds_name(
        ds,
        num_examples,
        num_active_examples,
        randomize_active,
        inactive_mode,
        inactive_alt,
        alt_spec,
        use_alt,
        mode,
    ):
        if inactive_mode == "alt":
            if inactive_alt is None:
                num_active_examples = None
            else:
                inactive_mode = str(inactive_alt) + "alt"
        ds_name = ds
        if num_active_examples is not None and num_active_examples != num_examples:
            ds_name += (
                "_"
                + ("rand" if randomize_active else "first")
                + str(num_active_examples)
            )
            # wall => noise is for backward compatibility
            ds_name += (
                "_"
                + (inactive_mode if inactive_mode != "wall" else "noise")
                + str(num_examples - num_active_examples)
            )
        else:
            ds_name += f"_{num_examples}"

        if alt_spec:
            ds_name += f"_altspec{inactive_alt}"
        if use_alt is not None:
            assert num_examples == 2
            ds_name += f"_alt{use_alt}"
        if mode != "synthesis":
            ds_name += f"_{mode}"
        return ds_name

    @property
    def train_ds_name(self):
        ds_name = self._ds_name(
            self.dataset,
            self.num_examples,
            self.num_active_examples,
            self.randomize_active,
            self.inactive_mode,
            self.inactive_alt,
            self.alt_spec,
            self.use_alt,
            self.mode,
        )
        return ds_name

    @property
    def train_base_dir(self):
        model_name = self.model_name
        if self.from_scratch:
            model_name += "/fromscratch"
        model_dir = model_name.replace("/", "_")

        ds_name = self.train_ds_name

        base_dir = os.path.join("outputs", self.output_dir, ds_name, model_dir)
        return base_dir

    @property
    def checkpoint_dir(self):
        if self.step is None:
            raise ValueError("No step provided.")

        base_dir = self.train_base_dir
        base_dir = os.path.join(base_dir, f"step_{self.step}")
        return base_dir

    @property
    def base_dir(self):
        if self.step is not None:
            base_dir = self.checkpoint_dir
        else:
            base_dir = self.train_base_dir

        ed = self.eval_dataset if self.eval_dataset is not None else self.dataset
        enae = (
            self.eval_num_active_examples
            if self.eval_num_active_examples is not None
            else self.num_active_examples
        )
        era = self.eval_randomize_active
        eim = (
            self.eval_inactive_mode
            if self.eval_inactive_mode is not None
            else self.inactive_mode
        )
        if eim == "alt":
            eia = (
                self.eval_inactive_alt
                if self.eval_inactive_alt is not None
                else self.inactive_alt
            )
        else:
            eia = None
        eval_name = self._ds_name(
            ed,
            self.num_examples,
            enae,
            era,
            eim,
            eia,
            self.eval_alt_spec,
            self.use_alt,
            self.mode,
        )
        if self.train_ds_name != eval_name:
            base_dir = os.path.join(base_dir, eval_name)
        return base_dir

    @property
    def semantic_dir(self):
        return os.path.join(self.base_dir, "dataset_semantic")

    @property
    def results_path(self):
        fn = ["results"]
        if self.use_alt is not None:
            if self.eval_alt_active is not None:
                fn.append(f"active{self.eval_alt_active}")
        if self.forced_decoding:
            fn.append("forced")
        if self.grammar_decoding:
            fn.append("grammar")
        assert self.split is not None
        fn.append(self.split)
        fn = "_".join(fn)
        return os.path.join(self.base_dir, f"{fn}.pt")

    def _semantic_results_path(self, meta=False):
        fn = self._semantic_fn()
        if fn:
            fn = "_" + fn
        fn = "results" + fn
        if meta:
            fn = f"{fn}_meta"
        return os.path.join(self.semantic_dir, f"{fn}.pt")

    @property
    def semantic_results_path(self):
        return self._semantic_results_path(meta=False)

    @property
    def semantic_results_meta_path(self):
        return self._semantic_results_path(meta=True)

    @property
    def semantic_probe_path(self):
        fn = self._semantic_fn()
        if fn:
            fn = "_" + fn
        fn = "probe" + fn
        return os.path.join(self.semantic_dir, f"{fn}.bin")

    @property
    def semantic_probe_results_path(self):
        fn = "results_probe_" + self._semantic_fn()
        return os.path.join(self.semantic_dir, f"{fn}.bin")

    def _semantic_ds_path(self, alt=False, meta=False):
        assert not alt or not meta
        fn = self._data_fn(alt)
        if meta:
            fn = f"{fn}_meta"
        ds_dir = os.path.join(self.semantic_dir, f"{fn}.pt")
        return ds_dir

    @property
    def semantic_ds_path(self):
        return self._semantic_ds_path()

    @property
    def semantic_ds_meta_path(self):
        return self._semantic_ds_path(meta=True)

    @property
    def alt_semantic_ds_path(self):
        if self.forced_decoding and self.step is not None:
            return self.update(step=None).alt_semantic_ds_path
        return self._semantic_ds_path(alt=True)

    @staticmethod
    def add_train_args(parser):
        parser.add_argument(
            "--output_dir",
            type=str,
            help="Where to store outputs.",
            required=True,
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            help="The name of the dataset to use.",
            required=True,
        )

        parser.add_argument(
            "--model_name",
            type=str,
            default="Salesforce/codegen-350M-mono",
            help="The model name of architecture.",
        )
        parser.add_argument(
            "--num_examples",
            type=int,
            default=5,
            help="Number of synthesis I/O examples to provide.",
        )
        parser.add_argument(
            "--num_active_examples",
            type=int,
            default=None,
            help="Number of active examples in spec.",
        )
        parser.add_argument(
            "--randomize_active",
            action="store_true",
            help="Whether to randomize which examples are selected to be active (or select the first n to be active).",
        )
        parser.add_argument(
            "--inactive_mode",
            type=str,
            default="wall",
            choices=["wall", "space", "input", "permute", "random", "alt"],
            help="What to do with the inactive examples.",
        )
        parser.add_argument(
            "--inactive_alt",
            type=int,
            default=None,
            help="Which semantics should be used for the inactive samples (must pass --inactive_mode=alt).",
        )
        parser.add_argument(
            "--alt_spec",
            action="store_true",
            help="Whether to change the spec with alt tokens (must pass --inactive_mode=alt and --inactive_alt).",
        )
        parser.add_argument(
            "--use_alt",
            type=int,
            default=None,
            help="Which alternative semantics to use for the second state (num_examples must be set to 2).",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="synthesis",
            choices=["synthesis", "interp"],
            help="Which spec mode to use.",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Debug run.",
        )

    @staticmethod
    def add_eval_args(parser):
        parser.add_argument(
            "--checkpoint_steps",
            type=int,
            default=None,
            help="Checkpoint to use in steps. If empty, eval final model.",
        )
        parser.add_argument(
            "--hidden_state_layer",
            type=str,
            default=None,
            help="Which hidden state to use. Defaults to None (mean reduction across layers). full runs all hidden states. Otherwise, pass 0-21 to pick a specific hidden state.",
        )
        parser.add_argument(
            "--max_eval_samples",
            type=int,
            default=100000,
            help="How many samples to use in the eval.",
        )
        parser.add_argument(
            "--eval_dataset_name",
            type=str,
            default=None,
            help="The name of the eval dataset to use, if different from dataset_name.",
        )
        parser.add_argument(
            "--eval_num_active_examples",
            type=int,
            default=None,
            help="Number of active examples in eval spec.",
        )
        parser.add_argument(
            "--eval_randomize_active",
            action="store_true",
            help="Whether to randomize which examples are selected to be active (or select the first n to be active).",
        )
        parser.add_argument(
            "--eval_inactive_mode",
            type=str,
            default=None,
            choices=["wall", "space", "input", "permute", "random", "alt"],
            help="What to do with the inactive examples.",
        )
        parser.add_argument(
            "--eval_inactive_alt",
            type=int,
            default=None,
            help="Which semantics should be used for the inactive samples (must pass --eval_inactive_mode=alt).",
        )
        parser.add_argument(
            "--eval_alt_spec",
            action="store_true",
            help="Whether to change the spec with alt tokens (must pass --eval_inactive_mode=alt and --eval_inactive_alt).",
        )
        parser.add_argument(
            "--eval_alt_active",
            type=str,
            default=None,
            choices=["0", "1", "all", "random"],
            help="Which semantics should be active (must pass --use_alt).",
        )

        parser.add_argument(
            "--split", type=str, default="test", choices=["test", "train"]
        )

        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--forced_decode",
            action="store_true",
            help="Whether to use forced decoding.",
        )
        group.add_argument(
            "--grammar_decode",
            action="store_true",
            help="Whether to use grammar decoding.",
        )

    @staticmethod
    def add_semantic_args(parser):
        parser.add_argument(
            "--offset", type=int, default=None, help="Offset of alignment."
        )
        parser.add_argument(
            "--alternative",
            type=int,
            default=None,
            help="Whether to use alternative semantics.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Seed to use for training.",
        )
        parser.add_argument(
            "--mlp_layers",
            type=int,
            default=1,
            help="How many layers to use in the MLP probe.",
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
            "--drop_last",
            action="store_true",
            help="Whether to drop the last state.",
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
