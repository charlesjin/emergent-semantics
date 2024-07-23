DATASET=$1
STEPS=$2
ARGS="$3"

set -x

python -m probe.run \
        --dataset_name $DATASET \
        --model_name Salesforce/codegen-350M-mono \
        --num_examples 5 \
        --output_dir outputs \
        --checkpoint_steps $STEPS \
        --make_semantic_ds_batch_size 24 \
        --semantic_train_batch_size 1024 \
        --semantic_eval_batch_size 4096 \
        --num_train_steps 10000000 $ARGS

