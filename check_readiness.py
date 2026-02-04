import os
import sys
import torchvision
from torchvision import datasets
from src.config import Config

def ensure_dataset_download(name, root):
    print(f"checking {name} in {root}...")
    try:
        if name == 'cifar10':
            datasets.CIFAR10(root=root, train=True, download=True)
            datasets.CIFAR10(root=root, train=False, download=True)
        elif name == 'cifar100':
            datasets.CIFAR100(root=root, train=True, download=True)
            datasets.CIFAR100(root=root, train=False, download=True)
        elif name == 'svhn':
            datasets.SVHN(root=root, split='train', download=True)
            datasets.SVHN(root=root, split='test', download=True)
        print(f"[OK] {name} is ready.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {name}: {e}")
        return False

def check_imagenet100(root):
    # Root should be ./data
    # ImageNet-100 path: ./data/imagenet100
    base_path = os.path.join(root, "imagenet100")
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        # Quick check class count
        classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
        if len(classes) > 0:
            print(f"[OK] ImageNet-100 found ({len(classes)} classes).")
            return True
            
    # Auto-handle: Try Real Download First
    download_tool = "tools/download_imagenet100.py"
    
    if os.path.exists(download_tool):
        print("  -> Attempting Real Download (via Kaggle)...")
        # Call the python script
        ret = os.system(f"python {download_tool}")
        if ret == 0:
            # Check again
             if os.path.exists(train_path):
                 print("[OK] Real ImageNet-100 downloaded successfully.")
                 return True
             else:
                 print("[WARNING] Download script ran but data still missing (Auth error?).")
        else:
             print("[WARNING] Download script failed.")

    return False

def main():
    print("Checking and Preparing Environment...")
    print("="*40)
    
    # 1. Configs
    datasets_list = ["cifar10", "cifar100", "svhn", "imagenet100"]
    missing_configs = []
    
    for ds in datasets_list:
        files = [f"conf/{ds}.json", f"conf/{ds}_metric_euclidean.json", f"conf/{ds}_fusion_average.json"]
        for f in files:
            if not os.path.exists(f):
                print(f"[MISSING] Config: {f}")
                missing_configs.append(f)
    
    if missing_configs:
        print("Error: Missing configs.")
        sys.exit(1)
    else:
        print("[OK] All configurations present.")
        
    # 2. Datasets
    data_root = "./data"
    if not os.path.exists(data_root):
        os.makedirs(data_root)
        
    # Standard Torchvision
    ensure_dataset_download('cifar10', data_root)
    ensure_dataset_download('cifar100', data_root)
    ensure_dataset_download('svhn', data_root)
    
    # ImageNet-100
    check_imagenet100(data_root)
    
    print("="*40)
    print("Readiness Check Complete.")

if __name__ == "__main__":
    main()
