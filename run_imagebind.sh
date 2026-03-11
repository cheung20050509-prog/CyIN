#!/bin/bash
# CyIN ImageBind training on MOSI (1024d audio+visual features).
# Paper: "For audio and vision modality, we leverage ImageBind as a feature extractor."
# Hyperparams follow Appendix D Table 6 (MOSI column).

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP/bin/python}"

nohup "$PYTHON" -u train.py \
    --dataset mosi_imagebind \
    --max_seq_length 50 \
    --train_batch_size 32 \
    --dev_batch_size 128 \
    --test_batch_size 128 \
    --gradient_accumulation_step 4 \
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
    --checkpoint_prefix cyIN_imagebind \
    --seed 128 \
    > train_imagebind.log 2>&1 &

echo "Training started. PID: $!"
echo "Monitor: tail -f train_imagebind.log"
