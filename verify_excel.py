import pandas as pd
import os
from src.config import Config

excel_path = "results/cifar10/analysis_report.xlsx"
if os.path.exists(excel_path):
    print("Excel found.")
    try:
        df = pd.read_excel(excel_path, sheet_name='Summary Metrics')
        print(df.to_string())
    except Exception as e:
        print(f"Error reading excel: {e}")
else:
    print("Excel NOT found.")
