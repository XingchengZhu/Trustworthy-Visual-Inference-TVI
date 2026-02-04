import torch
import numpy as np
import os
import time
from tqdm import tqdm
from src.config import Config
from src.dataset import get_dataloaders
from src.model import ResNet18Backbone
from src.ot_module import OTMetric
from src.evidence_module import EvidenceExtractor
from src.fusion_module import DempsterShaferFusion
from src.utils import setup_logger, save_results, compute_ece, compute_auroc
import matplotlib.pyplot as plt

import argparse

def load_backbone(device, logger):
    model = ResNet18Backbone(num_classes=Config.NUM_CLASSES).to(device)
    ckpt_name = f"best_resnet18_{Config.DATASET_NAME}.pth"
    checkpoint_path = os.path.join(Config.Checkpoints_DIR, ckpt_name)
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Using random weights.")
    model.eval()
    return model

def build_support_set(model, support_loader, device, logger):
    logger.info("Building Support Set Features...")
    support_features = []
    support_labels = []
    
    samples_per_class = Config.NUM_SUPPORT_SAMPLES // Config.NUM_CLASSES
    class_counts = {i: 0 for i in range(Config.NUM_CLASSES)}
    
    with torch.no_grad():
        for images, labels in tqdm(support_loader, desc="Extracting Support", leave=False):
            images = images.to(device)
            features, _ = model(images) 
            
            for i in range(images.size(0)):
                lbl = labels[i].item()
                if class_counts[lbl] < samples_per_class:
                    support_features.append(features[i].cpu()) 
                    support_labels.append(lbl)
                    class_counts[lbl] += 1
            
            if all(c >= samples_per_class for c in class_counts.values()):
                break
                
    support_features = torch.stack(support_features).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    logger.info(f"Support Set Created. Shape: {support_features.shape}")
    return support_features, support_labels

def plot_inference_metrics(uncertainties, correct_mask, ood_uncertainties=None):
    uncertainties = np.array(uncertainties)
    correct_mask = np.array(correct_mask)
    
    u_correct = uncertainties[correct_mask]
    u_incorrect = uncertainties[~correct_mask]
    
    plt.figure(figsize=(10, 6))
    plt.hist(u_correct, bins=30, alpha=0.5, label='ID Correct', density=True, color='green')
    plt.hist(u_incorrect, bins=30, alpha=0.5, label='ID Incorrect', density=True, color='red')
    
    if ood_uncertainties is not None:
         plt.hist(ood_uncertainties, bins=30, alpha=0.5, label='OOD (Noise)', density=True, color='blue')
         
    plt.title(f'Uncertainty Distribution ({Config.DATASET_NAME})')
    plt.xlabel('Uncertainty (Entropy / Mass)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Results dir with dataset name
    results_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    plt.savefig(os.path.join(results_dir, 'uncertainty_distribution.png'))
    plt.close()

def evaluate(model, test_loader, support_features, support_labels, device, logger):
    ot_metric = OTMetric(device)
    evidence_extractor = EvidenceExtractor(num_classes=Config.NUM_CLASSES)
    fusion_module = DempsterShaferFusion(num_classes=Config.NUM_CLASSES)
    
    correct_param = 0
    correct_nonparam = 0
    correct_fusion = 0
    total = 0
    total_uncertainty = 0.0
    
    # Storage for metrics
    all_uncertainties = []
    all_correct_fusion = []
    all_probs_fusion = []
    all_labels = []
    
    logger.info("Starting ID Inference...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images, labels = images.to(device), labels.to(device)
            
            # 1. Backbone
            features, logits = model(images)
            
            # 2. Parametric
            evidence_param = evidence_extractor.get_parametric_evidence(logits)
            alpha_param = evidence_param + 1
            
            # 3. Non-Parametric
            ot_dists, topk_indices = ot_metric.compute_batch_ot(features, support_features, support_labels)
            evidence_nonparam = evidence_extractor.get_non_parametric_evidence(ot_dists, topk_indices, support_labels)
            alpha_nonparam = evidence_nonparam + 1
            
            # 4. Fusion
            alpha_fuse, u_fuse = fusion_module.ds_combination(alpha_param, alpha_nonparam)
            
            # Predictions
            _, pred_param = torch.max(alpha_param, 1)
            _, pred_nonparam = torch.max(alpha_nonparam, 1)
            _, pred_fuse = torch.max(alpha_fuse, 1)
            
            # Uncertainty & Probs
            # Prob = alpha / sum(alpha)
            probs_fuse = alpha_fuse / torch.sum(alpha_fuse, dim=1, keepdim=True)
            
            total += labels.size(0)
            correct_param += (pred_param == labels).sum().item()
            correct_nonparam += (pred_nonparam == labels).sum().item()
            correct_fusion += (pred_fuse == labels).sum().item()
            total_uncertainty += u_fuse.sum().item()
            
            all_uncertainties.extend(u_fuse.flatten().cpu().numpy())
            all_correct_fusion.extend((pred_fuse == labels).cpu().numpy())
            all_probs_fusion.append(probs_fuse)
            all_labels.append(labels)
            
            # Limit for demo speed if needed
            # if total >= 500: break
            
    # Concatenate Probs
    all_probs_fusion = torch.cat(all_probs_fusion)
    all_labels = torch.cat(all_labels)
    
    # Compute Metrics
    acc_param = 100 * correct_param / total
    acc_nonparam = 100 * correct_nonparam / total
    acc_fuse = 100 * correct_fusion / total
    ece_score = compute_ece(all_probs_fusion, all_labels)
    
    logger.info(f"Parametric Acc: {acc_param:.2f}%")
    logger.info(f"Non-Parametric Acc: {acc_nonparam:.2f}%")
    logger.info(f"Fused Acc: {acc_fuse:.2f}%")
    logger.info(f"ECE Score: {ece_score:.4f}")
    
    # -----------------------
    # OOD Simulation (Noise)
    # -----------------------
    logger.info("Starting OOD (Noise) Inference...")
    ood_uncertainties = []
    num_ood = 500  # Number of noise samples
    
    with torch.no_grad():
        # Generate Gaussian noise
        noise_images = torch.randn(num_ood, 3, 32, 32).to(device)
        
        # Backbone
        features, logits = model(noise_images)
        
        # Parametric
        evidence_param = evidence_extractor.get_parametric_evidence(logits)
        alpha_param = evidence_param + 1
        
        # Non-Parametric (OT)
        ot_dists, topk_indices = ot_metric.compute_batch_ot(features, support_features, support_labels)
        evidence_nonparam = evidence_extractor.get_non_parametric_evidence(ot_dists, topk_indices, support_labels)
        alpha_nonparam = evidence_nonparam + 1
        
        # Fusion
        _, u_fuse_ood = fusion_module.ds_combination(alpha_param, alpha_nonparam)
        ood_uncertainties.extend(u_fuse_ood.flatten().cpu().numpy())
        
    auroc = compute_auroc(np.array(all_uncertainties), np.array(ood_uncertainties))
    logger.info(f"OOD AUROC (vs Gaussian Noise): {auroc:.4f}")
    
    # Save Results
    results = {
        "dataset": Config.DATASET_NAME,
        "accuracy_parametric": acc_param,
        "accuracy_nonparametric": acc_nonparam,
        "accuracy_fusion": acc_fuse,
        "ece": ece_score,
        "auroc_ood": auroc,
        "avg_uncertainty_id": total_uncertainty / total,
        "avg_uncertainty_ood": np.mean(ood_uncertainties)
    }
    results_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME)
    if not os.path.exists(results_dir):
         os.makedirs(results_dir)
         
    save_results(results, results_dir, filename="metrics.json")
    
    # Plot
    plot_inference_metrics(all_uncertainties, all_correct_fusion, ood_uncertainties)

def main():
    parser = argparse.ArgumentParser(description="Run TVI Inference")
    parser.add_argument("--config", type=str, default="conf/cifar10.json", help="Path to config file")
    args = parser.parse_args()
    
    # Load Config
    Config.load_config(args.config)
    
    results_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME)
    if not os.path.exists(results_dir):
         os.makedirs(results_dir)

    logger = setup_logger(results_dir, name="experiment")
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")
    
    # Load Data
    _, support_loader, test_loader = get_dataloaders()
    
    # Load Model
    model = load_backbone(device, logger)
    
    # Build Support Set
    support_features, support_labels = build_support_set(model, support_loader, device, logger)
    
    # Run Evaluation
    evaluate(model, test_loader, support_features, support_labels, device, logger)

if __name__ == "__main__":
    main()
