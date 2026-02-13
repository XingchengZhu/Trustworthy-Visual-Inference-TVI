#!/bin/bash
# =============================================================================
# TVI Ablation Study — ID Dataset: CIFAR-10
# =============================================================================
# 用法: bash scripts/run_ablation_cifar10.sh
# 日志输出: results/cifar10/resnet18/experiment_ablation_*.log
# 运行完毕后执行: python tools/parse_ablation_results.py --id_dataset cifar10
# =============================================================================

set -e  # 遇到错误立即停止

CONFIG="conf/cifar10.json"
RESULTS_DIR="results/cifar10/resnet18"
PYTHONPATH_CMD="env PYTHONPATH=."
COMMON_ARGS="--config ${CONFIG} --extended_ood"

mkdir -p ${RESULTS_DIR}

echo "============================================"
echo " TVI Ablation Study — CIFAR-10"
echo " Started at: $(date)"
echo "============================================"

# ─────────────────────────────────────────────
# Group A: 单分支 Baselines
# ─────────────────────────────────────────────
echo ""
echo ">>> [Group A] Single Branch Baselines"

# A1: Parametric Only
echo "[A1] Parametric Only ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --baseline \
    --task_note "A1_Parametric_Only" \
    --log_suffix "ablation_A1_param_only"

# ─────────────────────────────────────────────
# Group B: 融合策略对比
# ─────────────────────────────────────────────
echo ""
echo ">>> [Group B] Fusion Strategy Comparison"

# B1: Standard Dempster-Shafer
echo "[B1] Standard DS ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --task_note "B1_Standard_DS" \
    --log_suffix "ablation_B1_standard_ds"

# B2: Fixed Fusion w=0.5
echo "[B2] Fixed Fusion w=0.5 ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy fixed --fixed_weight 0.5 \
    --task_note "B2_Fixed_w05" \
    --log_suffix "ablation_B2_fixed_w05"

# B3: Fixed Fusion w=0.3
echo "[B3] Fixed Fusion w=0.3 ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy fixed --fixed_weight 0.3 \
    --task_note "B3_Fixed_w03" \
    --log_suffix "ablation_B3_fixed_w03"

# B4: Fixed Fusion w=0.7
echo "[B4] Fixed Fusion w=0.7 ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy fixed --fixed_weight 0.7 \
    --task_note "B4_Fixed_w07" \
    --log_suffix "ablation_B4_fixed_w07"

# B5: Adaptive Fusion (TVI Standard)
echo "[B5] Adaptive Fusion ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive \
    --task_note "B5_Adaptive_Fusion" \
    --log_suffix "ablation_B5_adaptive"

# ─────────────────────────────────────────────
# Group C: 逐一移除组件 (w/o experiments)
# ─────────────────────────────────────────────
echo ""
echo ">>> [Group C] Component Removal (w/o)"

# C0: TVI Full
echo "[C0] TVI Full ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot \
    --task_note "C0_TVI_Full" \
    --log_suffix "ablation_C0_full"

# C1: w/o Adaptive Fusion
echo "[C1] w/o Adaptive Fusion ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --pot \
    --task_note "C1_wo_Adaptive" \
    --log_suffix "ablation_C1_wo_adaptive"

# C2: w/o VOS
echo "[C2] w/o VOS ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --no_vos \
    --task_note "C2_wo_VOS" \
    --log_suffix "ablation_C2_wo_vos"

# C3: w/o React
echo "[C3] w/o React ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --no_react \
    --task_note "C3_wo_React" \
    --log_suffix "ablation_C3_wo_react"

# C4: w/o POT
echo "[C4] w/o POT ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive \
    --task_note "C4_wo_POT" \
    --log_suffix "ablation_C4_wo_pot"

# C5: w/o Trust-Param-ID
echo "[C5] w/o Trust-Param-ID ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --no_trust_param_id \
    --task_note "C5_wo_TrustParam" \
    --log_suffix "ablation_C5_wo_trustparam"

# C6: w/o Conflict Term
echo "[C6] w/o Conflict ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --no_conflict \
    --task_note "C6_wo_Conflict" \
    --log_suffix "ablation_C6_wo_conflict"

# ─────────────────────────────────────────────
# Group D: Ensemble 策略对比
# ─────────────────────────────────────────────
echo ""
echo ">>> [Group D] Ensemble Strategy"

# D2: Fixed Ensemble
echo "[D2] Fixed Ensemble w=0.5 ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --ensemble_weight 0.5 \
    --task_note "D2_Fixed_Ensemble" \
    --log_suffix "ablation_D2_fixed_ens"

echo ""
echo "============================================"
echo " CIFAR-10 Ablation Complete at: $(date)"
echo " Logs saved to: ${RESULTS_DIR}/"
echo " Next: python tools/parse_ablation_results.py --id_dataset cifar10"
echo "============================================"
