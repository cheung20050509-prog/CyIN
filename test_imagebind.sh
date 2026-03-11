#!/bin/bash
# Full evaluation of the ImageBind-trained CyIN checkpoint on MOSI.
# Covers complete, fixed-missing, and random-missing protocols (paper Appendix E).

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP/bin/python}"
CHECKPOINT="checkpoints/cyIN_imagebind_best.pt"
LOG_FILE="test_imagebind.log"

exec > >(tee "$LOG_FILE") 2>&1
echo "Evaluation started at $(date)"
echo "Checkpoint: $CHECKPOINT"
echo ""

COMMON="--dataset mosi_imagebind --checkpoint $CHECKPOINT --test_batch_size 128
        --unified_dim 256 --ib_dim 256 --bottleneck_dim 128
        --beta 32 --gamma 10
        --cra_layers 8 --cra_dims 64,32,16
        --attention_layers 2 --attention_heads 2"

# ── Complete modality ─────────────────────────────────────────────────────────
echo "========== Complete Modality =========="
"$PYTHON" test.py $COMMON

# ── Fixed missing ─────────────────────────────────────────────────────────────
echo ""
echo "========== Fixed Missing =========="
for MOD in acoustic visual text ta tv av; do
    echo "--- missing: $MOD ---"
    "$PYTHON" test.py $COMMON --missing_modality $MOD
done

# ── Random missing ────────────────────────────────────────────────────────────
echo ""
echo "========== Random Missing =========="
for MR in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    echo "--- MR=$MR ---"
    "$PYTHON" test.py $COMMON --missing_rate $MR
done

echo ""
echo "Done at $(date)"
