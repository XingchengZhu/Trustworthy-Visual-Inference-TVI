#!/bin/bash
# =============================================================================
# TVI Ablation Study — ID Dataset: CIFAR-100
# =============================================================================
# 用法: bash scripts/run_ablation_cifar100.sh
# 日志输出: results/cifar100/resnet18/experiment_ablation_*.log
# 运行完毕后执行: python tools/parse_ablation_results.py --id_dataset cifar100
# =============================================================================

set -e  # 遇到错误立即停止

CONFIG="conf/cifar100.json"
RESULTS_DIR="results/cifar100/resnet18"
PYTHONPATH_CMD="env PYTHONPATH=."
COMMON_ARGS="--config ${CONFIG} --extended_ood"

mkdir -p ${RESULTS_DIR}

echo "============================================"
echo " TVI Ablation Study — CIFAR-100"
echo " Started at: $(date)"
echo "============================================"

# ─────────────────────────────────────────────
# Group A: 单分支 Baselines
# ─────────────────────────────────────────────
echo ""
echo ">>> [Group A] Single Branch Baselines"

# A1: Parametric Only (纯 Logit 分支, 跳过 OT/Fusion)
echo "[A1] Parametric Only ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --baseline \
    --task_note "A1_Parametric_Only" \
    --log_suffix "ablation_A1_param_only"

# A2+A3: Non-Parametric (OT) 和 POT 的 AUROC 会从完整运行 (C0) 日志中提取
# 无需单独运行

# ─────────────────────────────────────────────
# Group B: 融合策略对比
# ─────────────────────────────────────────────
echo ""
echo ">>> [Group B] Fusion Strategy Comparison"

# B1: Standard Dempster-Shafer (无 Adaptive Discount)
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

# B3: Fixed Fusion w=0.3 (偏向 OT)
echo "[B3] Fixed Fusion w=0.3 ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy fixed --fixed_weight 0.3 \
    --task_note "B3_Fixed_w03" \
    --log_suffix "ablation_B3_fixed_w03"

# B4: Fixed Fusion w=0.7 (偏向 Param)
echo "[B4] Fixed Fusion w=0.7 ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy fixed --fixed_weight 0.7 \
    --task_note "B4_Fixed_w07" \
    --log_suffix "ablation_B4_fixed_w07"

# B5: Adaptive Fusion (TVI Standard — Sigmoid 门控)
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

# C0: TVI Full (完整模型 = Adaptive + POT)
echo "[C0] TVI Full ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot \
    --task_note "C0_TVI_Full" \
    --log_suffix "ablation_C0_full"

# C1: w/o Adaptive Fusion → 用 Standard DS 替代
echo "[C1] w/o Adaptive Fusion ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --pot \
    --task_note "C1_wo_Adaptive" \
    --log_suffix "ablation_C1_wo_adaptive"

# C2: w/o VOS (跳过虚拟离群点折扣)
echo "[C2] w/o VOS ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --no_vos \
    --task_note "C2_wo_VOS" \
    --log_suffix "ablation_C2_wo_vos"

# C3: w/o React (跳过特征截断)
echo "[C3] w/o React ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --no_react \
    --task_note "C3_wo_React" \
    --log_suffix "ablation_C3_wo_react"

# C4: w/o POT Branch
echo "[C4] w/o POT ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive \
    --task_note "C4_wo_POT" \
    --log_suffix "ablation_C4_wo_pot"

# C5: w/o Trust-Param-ID (使用纯融合不确定度)
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

# D1: No Ensemble (仅 Fusion, 即 C4)
# → 与 C4 相同, 跳过

# D2: Fixed Ensemble (Fusion:POT = 0.5:0.5)
echo "[D2] Fixed Ensemble w=0.5 ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --ensemble_weight 0.5 \
    --task_note "D2_Fixed_Ensemble" \
    --log_suffix "ablation_D2_fixed_ens"

# D3: Auto-Search Ensemble (默认行为, 即 C0)
# → 与 C0 相同, 跳过

echo ""
echo "============================================"
echo " CIFAR-100 Ablation Complete at: $(date)"
echo " Logs saved to: ${RESULTS_DIR}/"
echo " Next: python tools/parse_ablation_results.py --id_dataset cifar100"
echo "============================================"
