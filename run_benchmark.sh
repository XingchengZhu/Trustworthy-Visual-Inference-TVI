#!/bin/bash

# run_benchmark.sh
# Sequentially trains and evaluates on all supported datasets.
# Usage: ./run_benchmark.sh

set -e # Exit on error

echo "========================================"
echo "Starting Comprehensive Benchmark"
echo "========================================"

# Datasets to benchmark
DATASETS=("cifar10" "cifar100" "svhn")

for DATASET in "${DATASETS[@]}"; do
    CONFIG="conf/${DATASET}.json"
    
    if [ ! -f "$CONFIG" ]; then
        echo "Warning: Config $CONFIG not found. Skipping."
        continue
    fi
    
    echo "----------------------------------------"
    echo "Processing Dataset: $DATASET"
    echo "----------------------------------------"
    
    # 1. Training (Complete Training)
    echo "[Training] $DATASET..."
    ./run.sh train "$CONFIG"
    
    # NEW: 2. Baseline Verification (ResNet-Only)
    echo "[Inference - Baseline] $DATASET (ResNet Only)..."
    python -m src.inference --config "$CONFIG" --baseline --task_note "ResNet Baseline"
    
    # 3. Inference - Base (Proposed Method)
    echo "[Inference - Base] $DATASET (Sinkhorn + DS)..."
    python -m src.inference --config "$CONFIG" --task_note "Proposed (Sinkhorn+DS)"
    
    # 4. Inference - Ablation: Metric (Euclidean)
    CONFIG_METRIC="conf/${DATASET}_metric_euclidean.json"
    if [ -f "$CONFIG_METRIC" ]; then
        echo "[Inference - Ablation] $DATASET (Euclidean)..."
        python -m src.inference --config "$CONFIG_METRIC" --task_note "Ablation: Euclidean"
    else
        echo "Warning: Ablation config $CONFIG_METRIC not found."
    fi
    
    # 5. Inference - Ablation: Fusion (Average)
    CONFIG_FUSION="conf/${DATASET}_fusion_average.json"
    if [ -f "$CONFIG_FUSION" ]; then
        echo "[Inference - Ablation] $DATASET (Average Fusion)..."
        python -m src.inference --config "$CONFIG_FUSION" --task_note "Ablation: Avg Fusion"
    else
        echo "Warning: Ablation config $CONFIG_FUSION not found."
    fi
    
    echo "Finished $DATASET"
    echo ""
done

echo "========================================"
echo "Generating Summary Report"
echo "========================================"

python src/summarize_metrics.py

echo "Benchmark Complete. Check all_metrics.txt"
