
import torch
import numpy as np
from src.config import Config
from src.dataset import get_dataloaders
from src.model import ResNetBackbone
from src.inference import apply_bn_spatial, build_support_set
from src.ot_module import OTMetric
from src.evidence_module import EvidenceExtractor

class DummyLogger:
    def info(self, msg): print(msg)

def diagnose():
    # Force Config
    Config.METRIC_TYPE = 'cosine'
    Config.EVIDENCE_SCALE = 10000.0
    Config.RBF_GAMMA = 20.0
    Config.K_NEIGHBORS = 1
    Config.ODIN_EPS = 0.0
    Config.NUM_CLASSES = 100
    Config.DATASET_NAME = 'cifar100'
    Config.DATA_DIR = './data'

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load Model
    model = ResNetBackbone(num_classes=100).to(device)
    model.load_state_dict(torch.load("checkpoints/best_resnet18_cifar100.pth", map_location=device))
    model.eval()

    # Load Data
    _, support_loader, test_loader = get_dataloaders()
    logger = DummyLogger()
    
    # Support + VOS
    print("Building Support + VOS...")
    support_features, support_labels, virtual_outliers, _, _, _ = build_support_set(model, support_loader, device, logger)
    
    # Batch
    img, lbl = next(iter(test_loader))
    img = img.to(device)
    
    # Features
    with torch.no_grad():
        feat = apply_bn_spatial(model(img)[0], model.bn)
        
    print(f"Feature Shape: {feat.shape}")
    
    # OT Metric
    ot_metric = OTMetric()
    ev_extractor = EvidenceExtractor(num_classes=100)
    
    print("Running Compute Batch OT...")
    # Pass virtual_outliers
    dists, topk, vo_dists = ot_metric.compute_batch_ot(feat, support_features, None, virtual_outliers)
    
    print(f"Distances Mean: {dists.mean().item():.4f}")
    if Config.K_NEIGHBORS == 1:
        sim = 1.0 - dists.mean().item()
        print(f"Implied Sim: {sim:.4f}")
        
    # Evidence (Pass vo_distances)
    print("Computing Evidence...")
    # vo_distances is not None
    evidence = ev_extractor.get_non_parametric_evidence(dists, topk, support_labels, vo_distances=vo_dists)
    
    ev_sum = evidence.sum(dim=1).mean().item()
    print(f"Evidence Sum Mean: {ev_sum:.4f}")
    
    # Loop all batches
    print("Looping all batches...")
    all_u = []
    all_ev = []
    
    for i, (img, lbl) in enumerate(test_loader):
        img = img.to(device)
        with torch.no_grad():
            feat = apply_bn_spatial(model(img)[0], model.bn)
            
        dists, topk, vo_dists = ot_metric.compute_batch_ot(feat, support_features, None, virtual_outliers)
        evidence = ev_extractor.get_non_parametric_evidence(dists, topk, support_labels, vo_distances=vo_dists)
        
        alpha = evidence + 1
        S = alpha.sum(dim=1)
        u = Config.NUM_CLASSES / S
        
        all_u.extend(u.cpu().numpy())
        all_ev.extend(evidence.sum(dim=1).cpu().numpy())
        
        if i % 10 == 0:
            print(f"Batch {i}: U={u.mean().item():.4f}, Ev={evidence.sum(dim=1).mean().item():.4f}")
            
    print(f"Global U Mean: {np.mean(all_u):.4f}")
    print(f"Global Ev Mean: {np.mean(all_ev):.4f}")

if __name__ == "__main__":
    diagnose()
