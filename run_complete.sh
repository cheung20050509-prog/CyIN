#!/bin/bash

# CyIN complete-setting training on MOSI with the refined direct-predictor architecture.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python train.py \
    --dataset mosi \
    --max_seq_length 50 \
    --train_batch_size 32 \
    --dev_batch_size 64 \
    --test_batch_size 64 \
    --gradient_accumulation_step 4 \
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
    --stage2_missing_rate 0.0 \
    --checkpoint_prefix cyIN_complete_refined \
    --seed 128