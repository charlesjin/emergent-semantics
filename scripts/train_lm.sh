DATASET=$1
ARGS="$2"

set -x

python -m lm.train \
    --dataset_name $DATASET \
    --model_name Salesforce/codegen-350M-mono \
    --num_examples 5 \
    --num_active_examples 5 \
    --block_size 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --overwrite_cache \
    --gradient_accumulation_steps 8 \
    --checkpointing_steps 2000 \
    --preprocessing_num_workers 8 \
    --seed 8 \
    --resume_from_checkpoint $ARGS


