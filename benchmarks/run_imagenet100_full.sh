#!/bin/bash
set -e

# Benchmark Script for ImageNet-100 (Full Ablation)
# Usage: ./benchmarks/run_imagenet100_full.sh [GPU_ID]

DATASET="imagenet100"
CONFIG="conf/${DATASET}.json"

# GPU Support
if [ ! -z "$1" ]; then
    export CUDA_VISIBLE_DEVICES=$1
    echo "Using GPU: $1"
fi

echo "============================================="
echo "Starting Full Benchmark for $DATASET"
echo "============================================="

# 0. Ensure Training
if [ ! -f "checkpoints/best_resnet18_${DATASET}.pth" ]; then
    echo "[Training] Backbone..."
    python -m src.train_backbone --config "$CONFIG" --log_suffix "train"
fi

echo "--- Phase 1: Component Ablation ---"

# 1. Baseline (Parametric Only)
echo "[1.1] Baseline (Parametric)"
python -m src.inference --config "$CONFIG" --baseline --task_note "Component: Baseline (Param)" --log_suffix "baseline_param"

# 2. TVI (Ours)
echo "[1.2 & 1.6] TVI (Sinkhorn+DS)"
python -m src.inference --config "$CONFIG" --task_note "Component: TVI (Sinkhorn+DS)" --log_suffix "tvi_ours"

# 3. Dual-Stream (Euclidean + Average)
echo "[1.3] Variant A: Euclidean + Average"
python -m src.inference --config "$CONFIG" --metric_type euclidean --fusion_type average --task_note "Component: Euclidean + Avg" --log_suffix "ablation_euclidean_avg"

# 4. Dual-Stream (Sinkhorn + Average)
echo "[1.4] Variant B: Sinkhorn + Average"
python -m src.inference --config "$CONFIG" --metric_type sinkhorn --fusion_type average --task_note "Component: Sinkhorn + Avg" --log_suffix "ablation_sinkhorn_avg"

# 5. Dual-Stream (Euclidean + DS)
echo "[1.5] Variant C: Euclidean + DS"
python -m src.inference --config "$CONFIG" --metric_type euclidean --fusion_type dempster_shafer --task_note "Component: Euclidean + DS" --log_suffix "ablation_euclidean_ds"


echo "--- Phase 2: Sensitivity Analysis ---"

# 6. K Neighbors
for K in 1 5 10 20 50; do
    echo "[Sensitivity] K=$K"
    python -m src.inference --config "$CONFIG" --k_neighbors $K --task_note "Sensitivity: K=$K" --log_suffix "sensitivity_k_${K}"
done

# 7. Support Set Size (Memory)
rm -f checkpoints/${DATASET}_support.pt
for SZ in 100 500 1000 5000; do
    echo "[Sensitivity] Support Size=$SZ"
    rm -f checkpoints/${DATASET}_support.pt
    python -m src.inference --config "$CONFIG" --support_size $SZ --task_note "Sensitivity: Support=$SZ" --log_suffix "sensitivity_support_${SZ}"
done
rm -f checkpoints/${DATASET}_support.pt

# 8. RBF Gamma
for G in 0.1 0.5 1.0 2.0 5.0; do
    echo "[Sensitivity] Gamma=$G"
    python -m src.inference --config "$CONFIG" --rbf_gamma $G --task_note "Sensitivity: Gamma=$G" --log_suffix "sensitivity_gamma_${G}"
done

# 9. Sinkhorn Epsilon
for EPS in 0.01 0.05 0.1 0.5 1.0; do
    echo "[Sensitivity] Eps=$EPS"
    python -m src.inference --config "$CONFIG" --sinkhorn_eps $EPS --task_note "Sensitivity: Eps=$EPS" --log_suffix "sensitivity_eps_${EPS}"
done

echo "============================================="
echo "Benchmark Complete for $DATASET"
echo "Generate summary with: python src/summarize_metrics.py"
