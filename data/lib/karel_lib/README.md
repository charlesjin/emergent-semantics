Forked from https://github.com/carpedm20/karel

To generate the datasets used to train the LM, run
`python generate.py --no_loops --no_conditionals`

To generate the probe dataset used in the COLM paper, run
`python generate.py --no_loops --no_conditionals --no_markers --uniform --min_length 15 --max_length 15 --num_train 0 --data_dir karel_15only`
