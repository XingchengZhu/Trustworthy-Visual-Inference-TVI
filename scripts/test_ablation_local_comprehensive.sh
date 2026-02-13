#!/bin/bash
# =============================================================================
# TVI Local Test - Comprehensive
# Runs minimal samples (10) for ALL ablation groups to verify logic.
# Usage: bash scripts/test_ablation_local_comprehensive.sh [cifar100]
# =============================================================================

set -e

DATASET=${1:-"cifar100"}
CONFIG="conf/${DATASET}.json"
RESULTS_DIR="results/${DATASET}/test_local_full"
PYTHONPATH_CMD="env PYTHONPATH=."

# Use very few samples for speed
COMMON_ARGS="--config ${CONFIG} --max_samples 5"

mkdir -p ${RESULTS_DIR}

echo "============================================"
echo " TVI Local Comprehensive Test"
echo " Dataset: ${DATASET}"
echo "============================================"

run_test() {
    local note=$1
    local logname=$2
    local args=$3
    echo "[TEST] ${note} ..."
    ${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
        --task_note "[LOCAL] ${note}" \
        --log_suffix "${logname}" \
        ${args} || echo "!!! FAILED: ${note}"
}

# --- Group A ---
run_test "A1_Param" "test_A1" "--baseline"

# --- Group B ---
run_test "B1_Std_DS" "test_B1" ""
run_test "B5_Adaptive" "test_B5" "--adaptive_fusion --fusion_strategy adaptive"

# --- Group C ---
run_test "C0_Full" "test_C0" "--adaptive_fusion --fusion_strategy adaptive --pot"
run_test "C2_wo_VOS" "test_C2" "--adaptive_fusion --fusion_strategy adaptive --pot --no_vos"
run_test "C3_wo_React" "test_C3" "--adaptive_fusion --fusion_strategy adaptive --pot --no_react"
run_test "C6_wo_Conflict" "test_C6" "--adaptive_fusion --fusion_strategy adaptive --pot --no_conflict"

# --- Group D ---
run_test "D2_FixedEns" "test_D2" "--adaptive_fusion --fusion_strategy adaptive --pot --ensemble_weight 0.5"

# --- Group E (Sample) ---
run_test "E1_K1" "test_E1_K1" "--k_neighbors 1"
run_test "E3_Gamma0.1" "test_E3_G01" "--rbf_gamma 0.1"

echo ""
echo "============================================"
echo " Local Test Complete."
echo " Check ${RESULTS_DIR} for logs."
echo "============================================"
