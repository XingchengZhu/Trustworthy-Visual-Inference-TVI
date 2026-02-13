#!/bin/bash
# =============================================================================
# TVI Ablation Study — Local Test Script (Mini-Batch)
# =============================================================================
# 用于本地快速验证代码逻辑，仅运行 minimal samples
# =============================================================================

set -e

CONFIG="conf/cifar100.json"
RESULTS_DIR="results/cifar100/test_local"
PYTHONPATH_CMD="env PYTHONPATH=."
# 使用 10 个样本快速验证
COMMON_ARGS="--config ${CONFIG} --max_samples 10"

mkdir -p ${RESULTS_DIR}

echo "============================================"
echo " TVI Local Ablation Test"
echo " Started at: $(date)"
echo "============================================"

# Group A: A1 Parametric Only
echo "[TEST A1] Parametric Only ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --baseline \
    --task_note "A1_Parametric_Only" \
    --log_suffix "experiment_test_local_A1"

# Group B: B5 Adaptive Fusion
echo "[TEST B5] Adaptive Fusion ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive \
    --task_note "B5_Adaptive_Fusion" \
    --log_suffix "experiment_test_local_B5"

# Group C: C2 w/o VOS
echo "[TEST C2] w/o VOS ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --no_vos \
    --task_note "C2_wo_VOS" \
    --log_suffix "experiment_test_local_C2"

# Group D: D2 Fixed Ensemble
echo "[TEST D2] Fixed Ensemble ..."
${PYTHONPATH_CMD} python src/inference.py ${COMMON_ARGS} \
    --adaptive_fusion --fusion_strategy adaptive --pot --ensemble_weight 0.5 \
    --task_note "D2_Fixed_Ensemble" \
    --log_suffix "experiment_test_local_D2"

echo ""
echo "============================================"
echo " Local Test Complete!"
echo " Logs in: ${RESULTS_DIR}/"
echo "============================================"
