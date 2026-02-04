import torch
import json
import os

class Config:
    # Defaults
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    SEED = 42
    
    # Paths
    Checkpoints_DIR = "checkpoints"
    RESULTS_DIR = "results"
    DATA_DIR = "./data"
    
    # Dataset Params (Will be overwritten by load_config)
    DATASET_NAME = "cifar10"
    NUM_CLASSES = 10
    
    # Training
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    # OT Module
    NUM_SUPPORT_SAMPLES = 1000
    SINKHORN_EPS = 0.1
    SINKHORN_MAX_ITER = 100
    K_NEIGHBORS = 5
    
    # Evidence
    EVIDENCE_SCALE = 1.0

    @classmethod
    def load_config(cls, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        for key, value in config_dict.items():
            # Convert key to uppercase to match class attributes
            attr_name = key.upper()
            if hasattr(cls, attr_name):
                setattr(cls, attr_name, value)
            else:
                # Optionally set new attributes or warn
                print(f"Warning: Unknown config key {key}")
                setattr(cls, attr_name, value)
                
        print(f"Loaded configuration from {config_path}")
