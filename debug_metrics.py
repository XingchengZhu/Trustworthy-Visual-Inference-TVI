
import torch
import numpy as np
import os
from src.config import Config
from src.dataset import get_dataloaders, get_ood_loader
from src.model import ResNetBackbone
from src.ot_module import OTMetric
from src.inference import apply_bn_spatial, build_support_set

class DummyLogger:
    def info(self, msg): print(msg)
    def warning(self, msg): print(f"WARNING: {msg}")

def debug_metrics():
    Config.DATASET_NAME = 'cifar100'
    Config.METRIC_TYPE = 'cosine'
    Config.BATCH_SIZE = 128
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    logger = DummyLogger()
    
    # Load Model
    model = ResNetBackbone(num_classes=100).to(device)
    ckpt_path = "checkpoints/best_resnet18_cifar100.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Load Data
    _, support_loader, test_loader = get_dataloaders()
    ood_loader = get_ood_loader('svhn')
    
    # Build Support (Subset for speed)
    print("Building Support Set...")
    # Use existing function but maybe we debug it
    support_features, support_labels, _, _, _, _ = build_support_set(model, support_loader, device, logger)
    
    print(f"Support Features: {support_features.shape}")
    
    # Get 1 Batch ID
    id_iter = iter(test_loader)
    img_id, lbl_id = next(id_iter)
    img_id = img_id.to(device)
    
    # Get 1 Batch OOD
    ood_iter = iter(ood_loader)
    img_ood, lbl_ood = next(ood_iter)
    img_ood = img_ood.to(device)
    
    # Extract Features
    with torch.no_grad():
        _, _, logits_id, _ = model(img_id)
        raw_feat_id, _, _, _ = model(img_id)
        feat_id = apply_bn_spatial(raw_feat_id, model.bn)
        
        _, _, logits_ood, _ = model(img_ood)
        raw_feat_ood, _, _, _ = model(img_ood)
        feat_ood = apply_bn_spatial(raw_feat_ood, model.bn)
        
    print(f"\nFeature Stats (After BN):")
    print(f"ID  Mean: {feat_id.mean().item():.4f}, Std: {feat_id.std().item():.4f}, Norm: {feat_id.norm(dim=1).mean().item():.4f}")
    print(f"OOD Mean: {feat_ood.mean().item():.4f}, Std: {feat_ood.std().item():.4f}, Norm: {feat_ood.norm(dim=1).mean().item():.4f}")

    # 1. Flattened Cosine (Concatenated Spatial)
    flat_id = feat_id.view(feat_id.size(0), -1)
    flat_ood = feat_ood.view(feat_ood.size(0), -1)
    flat_supp = support_features.view(support_features.size(0), -1)
    
    norm_id = torch.nn.functional.normalize(flat_id, dim=1)
    norm_ood = torch.nn.functional.normalize(flat_ood, dim=1)
    norm_supp = torch.nn.functional.normalize(flat_supp, dim=1)
    
    max_sim_id, _ = torch.mm(norm_id, norm_supp.T).max(dim=1)
    max_sim_ood, _ = torch.mm(norm_ood, norm_supp.T).max(dim=1)
    
    print(f"\n[Flattened] Cosine Similarity:")
    print(f"ID  Mean: {max_sim_id.mean().item():.4f}")
    print(f"OOD Mean: {max_sim_ood.mean().item():.4f}")
    print(f"Diff: {max_sim_id.mean().item() - max_sim_ood.mean().item():.4f}")
    
    # 2. GAP Cosine (Global Average Pooling)
    gap_id = feat_id.mean(dim=(2, 3))
    gap_ood = feat_ood.mean(dim=(2, 3))
    gap_supp = support_features.mean(dim=(2, 3))
    
    norm_id_g = torch.nn.functional.normalize(gap_id, dim=1)
    norm_ood_g = torch.nn.functional.normalize(gap_ood, dim=1)
    norm_supp_g = torch.nn.functional.normalize(gap_supp, dim=1)
    
    max_sim_id_g, _ = torch.mm(norm_id_g, norm_supp_g.T).max(dim=1)
    max_sim_ood_g, _ = torch.mm(norm_ood_g, norm_supp_g.T).max(dim=1)
    
    print(f"\n[GAP] Cosine Similarity:")
    print(f"ID  Mean: {max_sim_id_g.mean().item():.4f}")
    print(f"OOD Mean: {max_sim_ood_g.mean().item():.4f}")
    print(f"Diff: {max_sim_id_g.mean().item() - max_sim_ood_g.mean().item():.4f}")


    # Check Raw Features (No BN)
    print("\n--- Checking Raw Features (No BN) ---")
    flat_id_raw = raw_feat_id.view(raw_feat_id.size(0), -1)
    flat_ood_raw = raw_feat_ood.view(raw_feat_ood.size(0), -1)
    
    # Need to extract raw support features too. 
    # This assumes build_support_set used BN. We can't easily undo it.
    # But let's check norms.
    print(f"ID Raw Norm: {flat_id_raw.norm(dim=1).mean().item():.4f}")
    print(f"OOD Raw Norm: {flat_ood_raw.norm(dim=1).mean().item():.4f}")

if __name__ == "__main__":
    debug_metrics()
