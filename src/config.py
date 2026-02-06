import torch
import json
import os

class Config:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Model Selection
    # Options: "resnet18", "resnet34", "resnet50", "wide_resnet50_2"
    BACKBONE = "resnet18" 
    
    # Input Size (for first layer modification)
    # CIFAR: 32, ImageNet: 224
    IMAGE_SIZE = 32 
    
    # Data params   
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
    SINKHORN_EPS = 0.1
    SINKHORN_MAX_ITER = 100
    K_NEIGHBORS = 5
    METRIC_TYPE = "mahalanobis" # sinkhorn, euclidean, cosine, mahalanobis
    REACT_PERCENTILE = 90 # For feature clipping (React), clip at 90th percentile
    
    # Evidence
    EVIDENCE_SCALE = 1.0
    RBF_GAMMA = 1.0
    
    # Fusion
    FUSION_TYPE = "dempster_shafer" # dempster_shafer, average
    
    # Advanced / Metrics Learning
    CENTER_LOSS_WEIGHT = 0.005 # Default recommendation is 0.003-0.01 for ResNet
    VOS_TYPE = "radial" # "radial" (classic) or "boundary" (between classes)
    VO_BETA = 1.0 # 1.0 for Radial, 0.5 for Boundary

    @classmethod
    def load_config(cls, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        for key, value in config_dict.items():
            # Convert key to uppercase to match class attributes
            attr_name = key.strip().upper()
            if hasattr(cls, attr_name):
                setattr(cls, attr_name, value)
            else:
                # Optionally set new attributes or warn
                print(f"Warning: Unknown config key {key}")
                setattr(cls, attr_name, value)
                
        print(f"Loaded configuration from {config_path}")
