#!/bin/bash

# CyIN unified paper-style training on MOSI.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP/bin/python}"

nohup "$PYTHON" -u train.py \
    --dataset mosi \
    --max_seq_length 50 \
    --train_batch_size 128 \
    --dev_batch_size 128 \
    --test_batch_size 128 \
    --gradient_accumulation_step 1 \
    --n_epochs 50 \
    --encoder_learning_rate 4e-5 \
    --cyin_learning_rate 1e-3 \
    --unified_dim 256 \
    --ib_dim 256 \
    --bottleneck_dim 128 \
    --beta 32 \
    --gamma 10 \
    --cra_layers 8 \
    --cra_dims 64,32,16 \
    --attention_layers 2 \
    --attention_heads 2 \
    --stage1_epochs 5 \
    --stage2_epochs 45 \
    --stage2_missing_rate 0.5 \
    --checkpoint_prefix cyIN_mosi \
    --seed 128 \
    > train_mosi.log 2>&1 &

echo "Training started in background. PID: $!"
echo "tail -f train_mosi.log"

# MOSEI dataset (Table 6 - uncomment to use)
# python train.py \
#     --dataset mosei \
#     --max_seq_length 50 \
#     --train_batch_size 128 \
#     --dev_batch_size 128 \
#     --test_batch_size 128 \
#     --n_epochs 50 \
#     --encoder_learning_rate 1e-5 \
#     --cyin_learning_rate 1e-3 \
#     --unified_dim 64 \
#     --ib_dim 64 \
#     --bottleneck_dim 256 \
#     --beta 4 \
#     --gamma 5 \
#     --cra_layers 4 \
#     --cra_dims 128,64,32 \
#     --attention_layers 8 \
#     --attention_heads 4 \
#     --stage1_epochs 15 \
#     --stage2_epochs 35 \
#     --seed 128
