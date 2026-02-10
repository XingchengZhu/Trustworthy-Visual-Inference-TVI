
import pandas as pd
import os

file_path = '/Users/zhuxingcheng/Projects/Trustworthy-Visual-Inference-TVI/results/cifar100/analysis_report.xlsx'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

try:
    xl = pd.ExcelFile(file_path)
    print(f"Sheet names: {xl.sheet_names}")
    
    for sheet in xl.sheet_names:
        print(f"\n--- Sheet: {sheet} Columns ---")
        df = xl.parse(sheet, nrows=0) # Read only header
        print(list(df.columns))
        print("-" * 30)
except Exception as e:
    print(f"Error reading excel: {e}")
