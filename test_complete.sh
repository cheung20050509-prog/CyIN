#!/bin/bash

# Evaluate only the complete-modality setting for the refined complete-trained checkpoint.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python test.py \
    --dataset mosi \
    --checkpoint checkpoints/cyIN_complete_refined_best.pt \
    --test_batch_size 128 \
    --unified_dim 256 \
    --ib_dim 256 \
    --bottleneck_dim 128 \
    --beta 32 \
    --gamma 10 \
    --cra_layers 8 \
    --cra_dims 64,32,16 \
    --attention_layers 2 \
    --attention_heads 2 \
    --complete_only