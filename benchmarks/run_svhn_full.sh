#!/bin/bash
set -e

# Benchmark Script for CIFAR-10 (Full Ablation)
DATASET="svhn"
CONFIG="conf/${DATASET}.json"

echo "============================================="
echo "Starting Full Benchmark for $DATASET"
echo "============================================="

# 0. Ensure Training (Fast check if checkpoint exists, otherwise train)
# For ablation, we assume model is trained. If not, run standard training once.
if [ ! -f "checkpoints/best_resnet18_${DATASET}.pth" ]; then
    echo "[Training] Backbone..."
    ./run.sh train "$CONFIG"
fi

echo "--- Phase 1: Component Ablation ---"

# 1. Baseline (Parametric Only)
echo "[1.1] Baseline (Parametric)"
python -m src.inference --config "$CONFIG" --baseline --task_note "Component: Baseline (Param)"

# 2. Baseline (Non-Param Only) -> Not supported natively as separate mode yet, 
# but we can infer from "Ours" logs (Acc OT column). 
# Or we can run standard inference and ignore fusion?
# Let's run Standard (Sinkhorn+DS) which gives us "Acc OT" (Row 2 in Table 1) and "Acc Fusion" (Row 6 in Table 1).
echo "[1.2 & 1.6] TVI (Ours) & OT-Only Baseline"
python -m src.inference --config "$CONFIG" --task_note "Component: TVI (Sinkhorn+DS)"

# 3. Dual-Stream (Euclidean + Average)
echo "[1.3] Variant A: Euclidean + Average"
python -m src.inference --config "$CONFIG" --metric_type euclidean --fusion_type average --task_note "Component: Euclidean + Avg"

# 4. Dual-Stream (Sinkhorn + Average)
echo "[1.4] Variant B: Sinkhorn + Average"
python -m src.inference --config "$CONFIG" --metric_type sinkhorn --fusion_type average --task_note "Component: Sinkhorn + Avg"

# 5. Dual-Stream (Euclidean + DS)
echo "[1.5] Variant C: Euclidean + DS"
python -m src.inference --config "$CONFIG" --metric_type euclidean --fusion_type dempster_shafer --task_note "Component: Euclidean + DS"


echo "--- Phase 2: Sensitivity Analysis ---"

# 6. K Neighbors
for K in 1 5 10 20 50; do
    echo "[Sensitivity] K=$K"
    python -m src.inference --config "$CONFIG" --k_neighbors $K --task_note "Sensitivity: K=$K"
done

# 7. Support Set Size (Memory)
# Note: Checkpoints cache specific support sets. This might need clearing if size changes dynamically?
# src/inference.py checks for "svhn_support.pt". 
# IF we change support size, we MUST force rebuild support set.
# Hack: delete support cache before running this loop.
rm -f checkpoints/${DATASET}_support.pt
for SZ in 100 500 1000 5000; do
    echo "[Sensitivity] Support Size=$SZ"
    # We must clear cache each time
    rm -f checkpoints/${DATASET}_support.pt
    python -m src.inference --config "$CONFIG" --support_size $SZ --task_note "Sensitivity: Support=$SZ"
done
# Restore default cache
rm -f checkpoints/${DATASET}_support.pt

# 8. RBF Gamma
for G in 0.1 0.5 1.0 2.0 5.0; do
    echo "[Sensitivity] Gamma=$G"
    python -m src.inference --config "$CONFIG" --rbf_gamma $G --task_note "Sensitivity: Gamma=$G"
done

# 9. Sinkhorn Epsilon
for EPS in 0.01 0.05 0.1 0.5 1.0; do
    echo "[Sensitivity] Eps=$EPS"
    python -m src.inference --config "$CONFIG" --sinkhorn_eps $EPS --task_note "Sensitivity: Eps=$EPS"
done

echo "============================================="
echo "Benchmark Complete for $DATASET"
echo "Generate summary with: python src/summarize_metrics.py"
