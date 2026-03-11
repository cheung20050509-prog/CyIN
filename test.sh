#!/bin/bash
# CyIN evaluation on MOSI: complete + fixed-missing + random-missing.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/cyIN_mosi_best.pt}"
COMMON="--dataset mosi --checkpoint $CHECKPOINT --test_batch_size 128
        --unified_dim 256 --ib_dim 256 --bottleneck_dim 128
        --beta 32 --gamma 10
        --cra_layers 8 --cra_dims 64,32,16
        --attention_layers 2 --attention_heads 2"

LOG_FILE="test_mosi.log"
exec > >(tee "$LOG_FILE") 2>&1
echo "=== CyIN MOSI Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Time: $(date)"
echo ""

echo "========== Complete Modality =========="
"$PYTHON" test.py $COMMON

echo ""
echo "========== Fixed Missing =========="
for MOD in acoustic visual text ta tv av; do
    echo "--- missing: $MOD ---"
    "$PYTHON" test.py $COMMON --missing_modality $MOD
done

echo ""
echo "========== Random Missing =========="
for MR in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    echo "--- MR=$MR ---"
    "$PYTHON" test.py $COMMON --missing_rate $MR
done

echo ""
echo "Done at $(date)"
