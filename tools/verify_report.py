import pandas as pd
import os
import sys

def verify_report(path):
    if not os.path.exists(path):
        print(f"[MISSING] Report file: {path}")
        return
        
    print(f"Reading {path}...")
    try:
        df = pd.read_excel(path, sheet_name='Summary Metrics')
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nRows:")
        print(df[['Dataset', 'Task Note', 'Acc Param', 'Acc Fusion']].to_string())
        
        # Check for Baseline
        baselines = df[df['Task Note'].str.contains("Baseline", na=False)]
        if not baselines.empty:
            print("\n[OK] Baseline row found.")
        else:
            print("\n[WARNING] No 'Baseline' task note found.")
            
        print("\n[SUCCESS] Report format verification complete.")
        
    except Exception as e:
        print(f"[ERROR] Failed to read report: {e}")

if __name__ == "__main__":
    report_path = "results/cifar10/analysis_report.xlsx"
    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    verify_report(report_path)
