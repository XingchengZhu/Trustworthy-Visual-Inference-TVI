import os
import json
import glob
from src.config import Config

def main():
    results_dir = "results"
    output_file = "all_metrics.txt"
    
    # Find all metrics.json files
    # Structure: results/<dataset_name>/metrics.json
    pattern = os.path.join(results_dir, "*", "metrics.json")
    files = glob.glob(pattern)
    
    if not files:
        print("No metrics.json files found.")
        return

    summary_lines = []
    summary_lines.append(f"Total metrics found: {len(files)}")
    summary_lines.append("="*50)
    
    for fpath in sorted(files):
        
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            dataset_name = data.get('dataset', 'Unknown')
            config = data.get('config', {})
            k = config.get('k_neighbors', '?')
            eps = config.get('sinkhorn_eps', '?')
            gamma = config.get('rbf_gamma', '?')
            sup = config.get('support_size', '?')
            
            summary_lines.append("-" * 30)
            summary_lines.append(f"File: {fpath}")
            summary_lines.append(f"Dataset: {dataset_name}")
            summary_lines.append(f"Config: Metric={config.get('metric')}, Fusion={config.get('fusion')}")
            summary_lines.append(f"Params: K={k}, Eps={eps}, Gamma={gamma}, Sup={sup}")
            summary_lines.append(f"Task Note: {data.get('Task Note', 'N/A')}")
            summary_lines.append(f"  Accuracy (Fusion): {data.get('accuracy_fusion', 'N/A')}%")
            summary_lines.append(f"  Accuracy (Param):  {data.get('accuracy_parametric', 'N/A')}%")
            summary_lines.append(f"  Accuracy (OT):     {data.get('accuracy_nonparametric', 'N/A')}%")
            summary_lines.append(f"  ECE:               {data.get('ece', 'N/A')}")
            summary_lines.append(f"  Latency:           {data.get('avg_latency_ms', 'N/A'):.2f} ms/sample")
            
            ood_data = data.get('auroc_ood', {})
            if ood_data:
                summary_lines.append("  OOD Metrics:")
                for ood_name, metrics in ood_data.items():
                    summary_lines.append(f"    [{ood_name}] AUROC (Fusion): {metrics.get('auroc_fusion', 'N/A')}")
                    summary_lines.append(f"    [{ood_name}] FPR@95 (Fusion): {metrics.get('fpr95_fusion', 'N/A')}")
            
            summary_lines.append("-" * 30)
            
        except Exception as e:
            summary_lines.append(f"Error reading {fpath}: {e}")
            
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(summary_lines))
        
    print(f"Summary written to {output_file}")
    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()
