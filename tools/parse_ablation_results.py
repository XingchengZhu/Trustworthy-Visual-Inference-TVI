#!/usr/bin/env python3
"""
TVI Ablation Results Parser
===========================
解析消融实验日志文件，提取 AUROC / FPR@95 指标，生成汇总 Excel。

用法:
    PYTHONPATH=. python tools/parse_ablation_results.py --id_dataset cifar100
    PYTHONPATH=. python tools/parse_ablation_results.py --id_dataset cifar10
"""

import argparse
import os
import re
import glob
import pandas as pd

# ── 实验 ID → 日志后缀 映射 ──────────────────────────────────────────────────
EXPERIMENTS = {
    # Group A: Single Branch Baselines
    "A1_Parametric_Only":   "ablation_A1_param_only",
    # Group B: Fusion Strategy
    "B1_Standard_DS":       "ablation_B1_standard_ds",
    "B2_Fixed_w05":         "ablation_B2_fixed_w05",
    "B3_Fixed_w03":         "ablation_B3_fixed_w03",
    "B4_Fixed_w07":         "ablation_B4_fixed_w07",
    "B5_Adaptive_Fusion":   "ablation_B5_adaptive",
    # Group C: Component Removal
    "C0_TVI_Full":          "ablation_C0_full",
    "C1_wo_Adaptive":       "ablation_C1_wo_adaptive",
    "C2_wo_VOS":            "ablation_C2_wo_vos",
    "C3_wo_React":          "ablation_C3_wo_react",
    "C4_wo_POT":            "ablation_C4_wo_pot",
    "C5_wo_TrustParam":     "ablation_C5_wo_trustparam",
    "C6_wo_Conflict":       "ablation_C6_wo_conflict",
    # Group D: Ensemble
    "D2_Fixed_Ensemble":    "ablation_D2_fixed_ens",
}

# ── 正则表达式 ────────────────────────────────────────────────────────────────
RE_OOD_HEADER   = re.compile(r"OOD (\S+) Results:")
RE_AUROC_FPR    = re.compile(r"AUROC \((\w+)\):\s+([\d.]+)\s+\|\s+FPR@95:\s+([\d.]+)")
RE_ACC_PARAM    = re.compile(r"Parametric Acc:\s+([\d.]+)%")
RE_ACC_OT       = re.compile(r"Non-Parametric Acc:\s+([\d.]+)%")
RE_ACC_FUSE     = re.compile(r"Fused Acc:\s+([\d.]+)%")
RE_ECE          = re.compile(r"ECE Score:\s+([\d.]+)")
RE_LATENCY      = re.compile(r"Inference Latency:\s+([\d.]+)\s+ms")
RE_ENS_WEIGHTS  = re.compile(r"Best Ensemble Weights:\s+(\{.+\})")


def find_log_file(results_dir: str, suffix: str) -> str | None:
    """根据后缀找到最新的日志文件"""
    pattern = os.path.join(results_dir, f"experiment_{suffix}_*.log")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]  # 取最新


def parse_log(log_path: str) -> dict:
    """
    解析单个日志文件，返回结构化结果:
    {
        "acc_param": float, "acc_ot": float, "acc_fuse": float, "ece": float,
        "latency_ms": float,
        "ood": {
            "svhn": {"auroc_fusion": .., "fpr_fusion": .., "auroc_param": .., ...},
            "cifar10": {...},
            ...
        }
    }
    """
    result = {
        "acc_param": None, "acc_ot": None, "acc_fuse": None,
        "ece": None, "latency_ms": None,
        "ood": {}
    }

    current_ood = None

    with open(log_path, "r") as f:
        for line in f:
            # ID Accuracy
            m = RE_ACC_PARAM.search(line)
            if m:
                result["acc_param"] = float(m.group(1))
                continue
            m = RE_ACC_OT.search(line)
            if m:
                result["acc_ot"] = float(m.group(1))
                continue
            m = RE_ACC_FUSE.search(line)
            if m:
                result["acc_fuse"] = float(m.group(1))
                continue
            m = RE_ECE.search(line)
            if m:
                result["ece"] = float(m.group(1))
                continue
            m = RE_LATENCY.search(line)
            if m:
                result["latency_ms"] = float(m.group(1))
                continue

            # OOD Results Header
            m = RE_OOD_HEADER.search(line)
            if m:
                current_ood = m.group(1)
                if current_ood not in result["ood"]:
                    result["ood"][current_ood] = {}
                continue

            # AUROC / FPR lines
            m = RE_AUROC_FPR.search(line)
            if m and current_ood:
                branch = m.group(1)   # Fusion, Param, OT, Norm, POT, Ens3
                auroc  = float(m.group(2))
                fpr    = float(m.group(3))
                result["ood"][current_ood][f"auroc_{branch}"] = auroc
                result["ood"][current_ood][f"fpr_{branch}"]   = fpr
                continue

            # Ensemble Weights
            m = RE_ENS_WEIGHTS.search(line)
            if m and current_ood:
                result["ood"][current_ood]["ens_weights"] = m.group(1)

    return result


def build_dataframe(all_results: dict, ood_datasets: list) -> pd.DataFrame:
    """
    将多个实验的解析结果构造为 DataFrame。
    行 = 实验 ID, 列 = 各 OOD 数据集的各分支 AUROC/FPR。
    """
    rows = []
    for exp_id, data in all_results.items():
        row = {"Experiment": exp_id}
        row["Acc_Param (%)"] = data.get("acc_param")
        row["Acc_OT (%)"]    = data.get("acc_ot")
        row["Acc_Fuse (%)"]  = data.get("acc_fuse")
        row["ECE"]           = data.get("ece")

        for ood_name in ood_datasets:
            ood = data.get("ood", {}).get(ood_name, {})
            # 主要关注 Fusion 分支（该实验使用的融合策略的最终 AUROC）
            row[f"{ood_name} AUROC↑"]     = ood.get("auroc_Fusion")
            row[f"{ood_name} FPR@95↓"]    = ood.get("fpr_Fusion")
            # 同时记录其他分支供参考
            row[f"{ood_name} AUROC(Param)"]  = ood.get("auroc_Param")
            row[f"{ood_name} AUROC(OT)"]     = ood.get("auroc_OT")
            row[f"{ood_name} AUROC(POT)"]    = ood.get("auroc_POT")
            row[f"{ood_name} AUROC(Ens3)"]   = ood.get("auroc_Ens3")
            row[f"{ood_name} FPR(Ens3)"]     = ood.get("fpr_Ens3")

        rows.append(row)

    return pd.DataFrame(rows)


def add_highlight(writer, df, sheet_name):
    """对 AUROC 列标红最佳、标蓝次佳"""
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]

    fmt_best   = workbook.add_format({"bg_color": "#FF6666", "bold": True, "num_format": "0.0000"})
    fmt_second = workbook.add_format({"bg_color": "#6699FF", "bold": True, "num_format": "0.0000"})
    fmt_num    = workbook.add_format({"num_format": "0.0000"})

    for col_idx, col_name in enumerate(df.columns):
        if "AUROC" not in col_name:
            continue
        vals = pd.to_numeric(df[col_name], errors="coerce")
        if vals.isna().all():
            continue
        sorted_vals = vals.dropna().sort_values(ascending=False)
        best   = sorted_vals.iloc[0] if len(sorted_vals) > 0 else None
        second = sorted_vals.iloc[1] if len(sorted_vals) > 1 else None

        for row_idx in range(len(df)):
            val = vals.iloc[row_idx]
            if pd.isna(val):
                continue
            if val == best:
                worksheet.write(row_idx + 1, col_idx, val, fmt_best)
            elif second is not None and val == second:
                worksheet.write(row_idx + 1, col_idx, val, fmt_second)
            else:
                worksheet.write(row_idx + 1, col_idx, val, fmt_num)


def main():
    parser = argparse.ArgumentParser(description="Parse TVI ablation logs and generate Excel")
    parser.add_argument("--id_dataset", type=str, required=True, choices=["cifar100", "cifar10"],
                        help="ID dataset name")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Root results directory")
    args = parser.parse_args()

    results_dir = os.path.join(args.results_root, args.id_dataset, "resnet18")
    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    # OOD 数据集顺序
    if args.id_dataset == "cifar100":
        ood_datasets = ["cifar10", "svhn", "mnist", "textures", "tinyimagenet", "noise"]
    else:
        ood_datasets = ["svhn", "cifar100", "mnist", "textures", "tinyimagenet", "noise"]

    # ── 按 Group 分组解析 ──────────────────────────────────────────────────────
    groups = {
        "GroupA": ["A1_Parametric_Only"],
        "GroupB": ["B1_Standard_DS", "B2_Fixed_w05", "B3_Fixed_w03",
                   "B4_Fixed_w07", "B5_Adaptive_Fusion"],
        "GroupC": ["C0_TVI_Full", "C1_wo_Adaptive", "C2_wo_VOS",
                   "C3_wo_React", "C4_wo_POT", "C5_wo_TrustParam",
                   "C6_wo_Conflict"],
        "GroupD": ["D2_Fixed_Ensemble"],
    }

    # 补充: A2/A3 数据从 C0_TVI_Full 日志提取 (OT/POT 分支 AUROC)
    # D1 = C4, D3 = C0 (跳过重复)

    all_results = {}
    missing = []

    for exp_id, suffix in EXPERIMENTS.items():
        log_path = find_log_file(results_dir, suffix)
        if log_path is None:
            missing.append(exp_id)
            all_results[exp_id] = {"acc_param": None, "acc_ot": None, "acc_fuse": None,
                                   "ece": None, "latency_ms": None, "ood": {}}
            continue
        print(f"  Parsing [{exp_id}] → {os.path.basename(log_path)}")
        all_results[exp_id] = parse_log(log_path)

    if missing:
        print(f"\n  ⚠ Missing logs for: {', '.join(missing)}")

    # ── 从 C0 补充 A2 (OT Only) 和 A3 (POT Only) ───────────────────────────
    c0_data = all_results.get("C0_TVI_Full", {})
    if c0_data.get("ood"):
        # A2: Non-Parametric (OT) branch
        a2 = {"acc_param": None, "acc_ot": c0_data.get("acc_ot"),
              "acc_fuse": None, "ece": None, "latency_ms": None, "ood": {}}
        for ood_name, metrics in c0_data["ood"].items():
            a2["ood"][ood_name] = {
                "auroc_Fusion": metrics.get("auroc_OT"),  # OT 分支作为 "Fusion" 展示
                "fpr_Fusion":   metrics.get("fpr_OT"),
            }
        all_results["A2_OT_Only"] = a2
        groups["GroupA"].append("A2_OT_Only")

        # A3: POT branch
        a3 = {"acc_param": None, "acc_ot": None,
              "acc_fuse": None, "ece": None, "latency_ms": None, "ood": {}}
        for ood_name, metrics in c0_data["ood"].items():
            a3["ood"][ood_name] = {
                "auroc_Fusion": metrics.get("auroc_POT"),
                "fpr_Fusion":   metrics.get("fpr_POT"),
            }
        all_results["A3_POT_Only"] = a3
        groups["GroupA"].append("A3_POT_Only")

        # D1 = C4 (No Ensemble), D3 = C0 (Auto-Search Ensemble)
        if "C4_wo_POT" in all_results:
            all_results["D1_No_Ensemble"] = all_results["C4_wo_POT"]
            groups["GroupD"].insert(0, "D1_No_Ensemble")
        all_results["D3_Auto_Ensemble"] = c0_data
        groups["GroupD"].append("D3_Auto_Ensemble")

    # ── 生成 Excel ────────────────────────────────────────────────────────────
    output_path = os.path.join(args.results_root, args.id_dataset, "ablation_results.xlsx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n  Generating Excel → {output_path}")

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for group_name, exp_ids in groups.items():
            group_data = {eid: all_results[eid] for eid in exp_ids if eid in all_results}
            if not group_data:
                continue
            df = build_dataframe(group_data, ood_datasets)
            sheet_name = f"{args.id_dataset.upper()}_{group_name}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            add_highlight(writer, df, sheet_name)

        # Summary Sheet: 主要展示各实验在所有 OOD 数据集上的 Fusion AUROC
        summary_data = build_dataframe(all_results, ood_datasets)
        summary_data.to_excel(writer, sheet_name="Summary", index=False)
        add_highlight(writer, summary_data, "Summary")

    print(f"  ✅ Done! Saved to: {output_path}")
    print(f"     Sheets: {list(groups.keys()) + ['Summary']}")


if __name__ == "__main__":
    main()
