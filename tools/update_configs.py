import os
import json
import glob
import sys

def update_epochs(new_epochs):
    print(f"Updating all configs in conf/ to epochs={new_epochs}...")
    files = glob.glob("conf/*.json")
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # Update epoch if present
            if 'epochs' in data:
                old = data['epochs']
                data['epochs'] = int(new_epochs)
                
                with open(fpath, 'w') as f:
                    json.dump(data, f, indent=4)
                    
                print(f"  [UPDATED] {os.path.basename(fpath)}: {old} -> {new_epochs}")
            else:
                print(f"  [SKIP] {os.path.basename(fpath)} (no 'epochs' key)")
                
        except Exception as e:
            print(f"  [ERROR] {fpath}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/update_configs.py <num_epochs>")
        sys.exit(1)
        
    update_epochs(sys.argv[1])
