#!/bin/bash
# =============================================================================
# TVI Sensitivity Analysis
# Group E: Neighbors (K), Support Size (N), Gamma, Epsilon
# Usage: bash scripts/run_sensitivity_analysis.sh [cifar10|cifar100|imagenet100] [gpu_id]
# =============================================================================

set -e

DATASET=${1:-"cifar100"}
GPU_ID=${2:-"0"}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

CONFIG="conf/${DATASET}.json"
RESULTS_DIR="results/${DATASET}/resnet18/sensitivity"
PYTHONPATH_CMD="env PYTHONPATH=."
COMMON_ARGS="--config ${CONFIG} --extended_ood"

mkdir -p ${RESULTS_DIR}

echo "============================================"
echo " TVI Sensitivity Analysis: ${DATASET}"
echo " Started at: $(date)"
echo "============================================"

run_sens() {
    local note=$1
    local suffix=$2
    local extra_args=$3
    
    echo ">> Running Sensitivity [${note}] ..."
    ${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
        --task_note "${note}" \
        --log_suffix "${suffix}" \
        ${extra_args}
}

# E1: K Neighbors
for K in 1 5 10 20 50; do
    run_sens "Sens_K${K}" "sensitivity_k_${K}" "--k_neighbors ${K}"
done

# E2: Support Set Size (Requires rebuilding support)
rm -f checkpoints/${DATASET}_support.pt
for SZ in 100 500 1000 5000; do
    # Note: --rebuild_support forces reconstruction with new size
    # We remove cache file to be safe
    rm -f checkpoints/${DATASET}_support.pt
    run_sens "Sens_Size${SZ}" "sensitivity_size_${SZ}" "--support_size ${SZ} --rebuild_support"
done
# Restore default support cache (optional, simply deleting it forces regen next time)
rm -f checkpoints/${DATASET}_support.pt

# E3: RBF Gamma
for G in 0.1 0.5 1.0 2.0 5.0; do
    run_sens "Sens_Gamma${G}" "sensitivity_gamma_${G}" "--rbf_gamma ${G}"
done

# E4: Sinkhorn Epsilon
for EPS in 0.01 0.05 0.1 0.5 1.0; do
    run_sens "Sens_Eps${EPS}" "sensitivity_eps_${EPS}" "--sinkhorn_eps ${EPS}"
done

echo ""
echo "============================================"
echo " Sensitivity Analysis Complete!"
echo " Logs: ${RESULTS_DIR}/"
echo "============================================"
