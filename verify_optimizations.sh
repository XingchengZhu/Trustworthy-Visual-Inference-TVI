#!/bin/bash

# Verification Script for ASH and LogitNorm (Phase 15)
# Tests:
# 1. Baseline (No ASH)
# 2. ASH @ 90%
# 3. ASH @ 95%

echo "=========================================="
echo "   Running Optimization Verification"
echo "=========================================="

# Ensure output directory exists
mkdir -p results/cifar10

# 1. Run Baseline (No ASH)
echo ""
echo "[1/3] Running Baseline (No ASH)..."
python -m src.inference --config conf/cifar10.json --extended_ood --max_samples 500 > results/verify_baseline.log 2>&1
echo "Done. Log: results/verify_baseline.log"

# Search for Metrics in log
echo "Baseline Metrics (Partial):"
grep "AUROC" results/verify_baseline.log | tail -n 5
grep "FPR" results/verify_baseline.log | tail -n 5

# 2. Run ASH 90
echo ""
echo "[2/3] Running ASH @ 90%..."
python -m src.inference --config conf/cifar10.json --extended_ood --ash --max_samples 500 > results/verify_ash90.log 2>&1
echo "Done. Log: results/verify_ash90.log"

echo "ASH 90 Metrics:"
grep "AUROC" results/verify_ash90.log | tail -n 5
grep "FPR" results/verify_ash90.log | tail -n 5

# 3. Quick Compare
echo ""
echo "=========================================="
echo "          Comparison Summary"
echo "=========================================="
echo "See full logs in results/verify_*.log"
