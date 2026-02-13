
import torch
import sys
import os
from src.dataset import get_ood_loader
from src.config import Config

# Mock Config
Config.DATA_DIR = "data"
Config.BATCH_SIZE = 4
Config.DATASET_NAME = 'cifar100' # default

def test_loader(name, img_size):
    print(f"Testing {name} with img_size={img_size}...")
    try:
        loader = get_ood_loader(name, img_size=img_size)
        batch = next(iter(loader))
        images, labels = batch
        print(f"  Success! Image Shape: {images.shape}")
        if images.shape[2] != img_size or images.shape[3] != img_size:
            print(f"  ERROR: Expected size {img_size}, got {images.shape[2]}x{images.shape[3]}")
    except FileNotFoundError as e:
        print(f"  Skipped (Not Found): {e}")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    # Test SVHN (Downloadable)
    test_loader("svhn", 32)

    # Test Places365 (assuming downloaded)
    test_loader("places365", 32)
    
    # Test others (will likely skip)
    test_loader("inaturalist", 224)
    test_loader("openimage_o", 224)
    test_loader("places365", 224) # Test resize
