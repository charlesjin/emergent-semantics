# Emergent semantics 

This repo contains code for the following papers
-  Emergent Representations of Program Semantics in Language Models Trained on Programs (ICML'24, [arXiv](https://arxiv.org/abs/2305.11169))
- Latent Causal Probing: A Formal Perspective on Probing with Causal Models of Data (COLM'24, [arXiv](https://arxiv.org/abs/2407.13765))

## General usage

- Creating a conda env

   `conda create --prefix=./env --file environment.yml`
   `conda activate ./env`

- Then generate the Karel dataset (see `data/lib/karel_lib/README.md`)

- Training an LM

   `./scripts/train_lm.sh base karel`

- Training probes for one checkpoint

   `./scripts/train_probe.sh karel 76000`

## ICML'24

- Training the LM

   `./scripts/train_lm.sh karel_noloops_nocond "--output_dir filtered --learning_rate 5e-6 --num_warmup_steps 6000 --max_train_steps 80000 --lengths_to_filter 1 2 3 4 5"`

- Training probes for one checkpoint

   `./scripts/train_probe.sh karel_noloops_nocond 76000 "--eval_mode intervention --output_dir filtered --max_eval_samples 50000"`

## COLM'24

- We reuse the checkpoints from ICML'24

- Training probes for one checkpoint

   `./scripts/train_probe.sh karel 76000 "--eval_mode causal --output_dir filtered --eval_dataset karel_15only_uniform_noloops_nocond_nomarks --max_eval_samples 50000"`
