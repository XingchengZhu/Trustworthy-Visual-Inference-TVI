#!/bin/bash
# =============================================================================
# TVI Comprehensive Ablation Study
# Groups: A (Baselines), B (Fusion), C (Components), D (Ensemble)
# Usage: bash scripts/run_comprehensive_ablation.sh [cifar10|cifar100|imagenet100] [gpu_id]
# =============================================================================

set -e

DATASET=${1:-"cifar100"}
GPU_ID=${2:-"0"}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

CONFIG="conf/${DATASET}.json"
RESULTS_DIR="results/${DATASET}/resnet18/ablation_full"
PYTHONPATH_CMD="env PYTHONPATH=."
COMMON_ARGS="--config ${CONFIG} --extended_ood"

mkdir -p ${RESULTS_DIR}

echo "============================================"
echo " TVI Comprehensive Ablation: ${DATASET}"
echo " GPU: ${GPU_ID}"
echo " Config: ${CONFIG}"
echo " Started at: $(date)"
echo "============================================"

# Helper function
run_exp() {
    local note=$1
    local suffix=$2
    local extra_args=$3
    
    echo ">> Running [${note}] ..."
    ${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
        --task_note "${note}" \
        --log_suffix "${suffix}" \
        ${extra_args}
}

# -----------------------------------------------------------------------------
# Group A: Single Branch Baselines
# -----------------------------------------------------------------------------
echo ""
echo ">>> [Group A] Baselines"
run_exp "A1_Parametric_Only" "ablation_A1_param_only" "--baseline"
# A2 (OT) & A3 (POT) are extracted from C0 logs

# -----------------------------------------------------------------------------
# Group B: Fusion Strategy
# -----------------------------------------------------------------------------
echo ""
echo ">>> [Group B] Fusion Strategy"
run_exp "B1_Standard_DS" "ablation_B1_standard_ds" ""
run_exp "B2_Fixed_w05"   "ablation_B2_fixed_w05"   "--adaptive_fusion --fusion_strategy fixed --fixed_weight 0.5"
run_exp "B3_Fixed_w03"   "ablation_B3_fixed_w03"   "--adaptive_fusion --fusion_strategy fixed --fixed_weight 0.3"
run_exp "B4_Fixed_w07"   "ablation_B4_fixed_w07"   "--adaptive_fusion --fusion_strategy fixed --fixed_weight 0.7"
run_exp "B5_Adaptive"    "ablation_B5_adaptive"    "--adaptive_fusion --fusion_strategy adaptive"

# -----------------------------------------------------------------------------
# Group C: Component Ablation
# -----------------------------------------------------------------------------
echo ""
echo ">>> [Group C] Component Ablation"

# C0: Full TVI (Adaptive + POT + VOS + React + Conflict)
run_exp "C0_TVI_Full"      "ablation_C0_full"      "--adaptive_fusion --fusion_strategy adaptive --pot"

# C1: w/o Adaptive Fusion (Uses Standard DS logic but keeps other components if applicable, usually same as B1 but with POT potentially enabled? 
# Note: In plan, C1 is 'w/o Adaptive', meaning Standard DS. If we want C1 to be comparable to C0 but just swapping fusion, we should enable POT too if C0 has it.)
# Let's align with scripts/run_ablation_cifar100.sh: C1 was just --pot (implying Standard DS implicitly).
run_exp "C1_wo_Adaptive"   "ablation_C1_wo_adaptive" "--pot"

# C2: w/o VOS
run_exp "C2_wo_VOS"        "ablation_C2_wo_vos"      "--adaptive_fusion --fusion_strategy adaptive --pot --no_vos"

# C3: w/o React
run_exp "C3_wo_React"      "ablation_C3_wo_react"    "--adaptive_fusion --fusion_strategy adaptive --pot --no_react"

# C4: w/o POT Branch (Adaptive Fusion Only) -> similar to B5 but we name it C4 for 'w/o POT' context
run_exp "C4_wo_POT"        "ablation_C4_wo_pot"      "--adaptive_fusion --fusion_strategy adaptive"

# C5: w/o Trust-Param-ID
run_exp "C5_wo_TrustParam" "ablation_C5_wo_trustparam" "--adaptive_fusion --fusion_strategy adaptive --pot --no_trust_param_id"

# C6: w/o Conflict Term
run_exp "C6_wo_Conflict"   "ablation_C6_wo_conflict"   "--adaptive_fusion --fusion_strategy adaptive --pot --no_conflict"

# -----------------------------------------------------------------------------
# Group D: Ensemble Strategy
# -----------------------------------------------------------------------------
echo ""
echo ">>> [Group D] Ensemble Strategy"
# D1: No Ensemble (=C4 w/o POT) -> Skipped, redundant
# D2: Fixed Ensemble
run_exp "D2_Fixed_Ensemble" "ablation_D2_fixed_ens" "--adaptive_fusion --fusion_strategy adaptive --pot --ensemble_weight 0.5"
# D3: Auto-Search (=C0) -> Skipped, redundant

echo ""
echo "============================================"
echo " Comprehensive Ablation Complete!"
echo " Logs: ${RESULTS_DIR}/"
echo " Run parser: python tools/parse_ablation_results.py --id_dataset ${DATASET}"
echo "============================================"
