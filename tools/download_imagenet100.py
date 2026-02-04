import os
import sys
import subprocess

def download_imagenet100(target_dir="./data/imagenet100"):
    print(f"Attempting to download ImageNet-100 to {target_dir}...")
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("[ERROR] 'kaggle' library not found. Please run 'pip install kaggle'.")
        return False

    # Check for credentials
    kaggle_config_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_config_dir, "kaggle.json")
    if not os.path.exists(kaggle_json):
        print(f"[ERROR] Kaggle credentials not found at {kaggle_json}.")
        print("Please download 'kaggle.json' from your Kaggle Account Settings and place it there.")
        print("chmod 600 ~/.kaggle/kaggle.json")
        return False
        
    # Create dir if not exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    # Command
    # kaggle datasets download -d ambityga/imagenet100 -p ./data/imagenet100 --unzip
    # Using subprocess to see output
    cmd = [
        "kaggle", "datasets", "download", 
        "-d", "ambityga/imagenet100", 
        "-p", target_dir, 
        "--unzip"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("[SUCCESS] ImageNet-100 downloaded and unzipped.")
        
        # Verify structure: The zip might unzip into a subfolder or directly
        # ambityga/imagenet100 usually has 'train' 'val' inside.
        if os.path.exists(os.path.join(target_dir, "train")):
            return True
        else:
            print("[WARNING] Download finished but 'train' folder not found directly.")
            print(f"Check {target_dir} to see if it's nested.")
            # Maybe list dir
            print(os.listdir(target_dir))
            return False # Let manual check handle it
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Download failed: {e}")
        return False
    except FileNotFoundError:
        print("[ERROR] 'kaggle' command line tool not found in PATH.")
        return False

if __name__ == "__main__":
    download_imagenet100()
