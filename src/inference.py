import torch
import numpy as np
import os
import time
from tqdm import tqdm
from src.config import Config
from src.dataset import get_dataloaders, get_ood_loader
from src.model import ResNet18Backbone
from src.ot_module import OTMetric
from src.evidence_module import EvidenceExtractor
from src.fusion_module import DempsterShaferFusion
from src.utils import setup_logger, save_results, compute_ece, compute_auroc
import matplotlib.pyplot as plt
import pandas as pd
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
    support_path = os.path.join(Config.Checkpoints_DIR, f"{Config.DATASET_NAME}_support.pt")
    
    if os.path.exists(support_path):
        logger.info(f"Loading cached support set from {support_path}")
        data = torch.load(support_path, map_location=device)
        return data['features'], data['labels']

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
    
    # Cache support set
    torch.save({'features': support_features, 'labels': support_labels}, support_path)
    logger.info(f"Saved support set to {support_path}")
    
    return support_features, support_labels

def plot_inference_metrics(uncertainties, correct_mask, ood_uncertainties=None):
    uncertainties = np.array(uncertainties).flatten()
    correct_mask = np.array(correct_mask)
    
    u_correct = uncertainties[correct_mask]
    u_incorrect = uncertainties[~correct_mask]
    
    plt.figure(figsize=(10, 6))
    plt.hist(u_correct, bins=30, alpha=0.5, label='ID Correct', density=True, color='green')
    plt.hist(u_incorrect, bins=30, alpha=0.5, label='ID Incorrect', density=True, color='red')
    
    if ood_uncertainties is not None:
         ood_arr = np.array(ood_uncertainties).flatten()
         plt.hist(ood_arr, bins=30, alpha=0.5, label='OOD (Noise)', density=True, color='blue')
         
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
    
    # ID Metric Storage
    id_uncertainties_fuse = []
    id_uncertainties_param = []
    id_uncertainties_nonparam = []
    
    # Detailed Logs container
    detailed_logs = []
    
    # ID Probs for ECE
    all_probs_fusion = []
    all_labels_tensor = []
    
    # Metrics for Plots
    all_correct_fusion = [] # Boolean mask for ID
    
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
            # Now fusion returns (alpha, u, C)
            alpha_fuse, u_fuse, C_fuse = fusion_module.ds_combination(alpha_param, alpha_nonparam)
            
            # Calculate uncertainties for branches (Entroy or similar)
            # Parametric U: num_classes / sum(alpha)
            S_param = torch.sum(alpha_param, dim=1)
            u_param = Config.NUM_CLASSES / S_param
            
            # NonParam U: num_classes / sum(alpha)
            S_nonparam = torch.sum(alpha_nonparam, dim=1)
            u_nonparam = Config.NUM_CLASSES / S_nonparam
            
            # Predictions
            _, pred_param = torch.max(alpha_param, 1)
            _, pred_nonparam = torch.max(alpha_nonparam, 1)
            _, pred_fuse = torch.max(alpha_fuse, 1)
            probs_fuse = alpha_fuse / torch.sum(alpha_fuse, dim=1, keepdim=True)
            confidence, _ = torch.max(probs_fuse, 1)
            
            # Min OT Distance (First neighbor)
            min_ot_dist = ot_dists[:, 0]
            
            # Update Counters
            total += labels.size(0)
            correct_param += (pred_param == labels).sum().item()
            correct_nonparam += (pred_nonparam == labels).sum().item()
            correct_fusion += (pred_fuse == labels).sum().item()
            total_uncertainty += u_fuse.sum().item()
            
            # Store ID Metrics
            id_uncertainties_fuse.extend(u_fuse.cpu().numpy())
            id_uncertainties_param.extend(u_param.cpu().numpy())
            id_uncertainties_nonparam.extend(u_nonparam.cpu().numpy())
            
            all_probs_fusion.append(probs_fuse)
            all_labels_tensor.append(labels)
            all_correct_fusion.extend((pred_fuse == labels).cpu().numpy())
            
            # Log Sample Details
            for i in range(images.size(0)):
                detailed_logs.append({
                    "dataset_source": "ID",
                    "true_label": labels[i].item(),
                    "pred_label": pred_fuse[i].item(),
                    "is_correct": bool(pred_fuse[i].item() == labels[i].item()),
                    "uncertainty_fuse": u_fuse[i].item(),
                    "uncertainty_param": u_param[i].item(),
                    "uncertainty_ot": u_nonparam[i].item(),
                    "conflict": C_fuse[i].item(),
                    "confidence": confidence[i].item(),
                    "min_ot_dist": min_ot_dist[i].item()
                })

    # Concatenate Probs
    all_probs_fusion = torch.cat(all_probs_fusion)
    all_labels_tensor = torch.cat(all_labels_tensor)
    
    # Compute Metrics
    acc_param = 100 * correct_param / total
    acc_nonparam = 100 * correct_nonparam / total
    acc_fuse = 100 * correct_fusion / total
    ece_score = compute_ece(all_probs_fusion, all_labels_tensor)
    
    logger.info(f"Parametric Acc: {acc_param:.2f}%")
    logger.info(f"Non-Parametric Acc: {acc_nonparam:.2f}%")
    logger.info(f"Fused Acc: {acc_fuse:.2f}%")
    logger.info(f"ECE Score: {ece_score:.4f}")
    
    # -----------------------
    # OOD Evaluation
    # -----------------------
    # Define OOD datasets to test based on current dataset
    ood_datasets = []
    if Config.DATASET_NAME == 'cifar10':
        ood_datasets = ['svhn', 'cifar100']
    elif Config.DATASET_NAME == 'cifar100':
        ood_datasets = ['svhn', 'cifar10']
    
    # Always keeping Noise for sanity check
    ood_datasets.append('noise')
    
    ood_results = {}
    
    # Store last OOD uncertainties for plotting (Noise usually)
    last_ood_uncertainties = None

    for ood_name in ood_datasets:
        logger.info(f"Starting OOD ({ood_name}) Inference...")
        
        ood_u_fuse = []
        ood_u_param = []
        ood_u_nonparam = []
        
        current_ood_loop_func = None
        
        if ood_name == 'noise':
            # Generator wrapper
            def noise_gen():
                num_ood = 500
                with torch.no_grad():
                    noise_images = torch.randn(num_ood, 3, 32, 32).to(device)
                    # Fake labels -1 for OOD
                    fake_labels = torch.full((num_ood,), -1, dtype=torch.long).to(device)
                    yield noise_images, fake_labels
            current_ood_loop_func = noise_gen
        else:
            def loader_gen():
                loader = get_ood_loader(ood_name)
                max_samples = 500
                cnt = 0
                for img, lbl in loader:
                    if cnt >= max_samples: break
                    # Force labels to -1 for OOD
                    lbl = torch.full_like(lbl, -1)
                    yield img.to(device), lbl.to(device)
                    cnt += img.size(0)
            current_ood_loop_func = loader_gen

        try:
            with torch.no_grad():
                for images, labels in current_ood_loop_func():
                    features, logits = model(images)
                    
                    evidence_param = evidence_extractor.get_parametric_evidence(logits)
                    alpha_param = evidence_param + 1
                    
                    ot_dists, topk_indices = ot_metric.compute_batch_ot(features, support_features, support_labels)
                    evidence_nonparam = evidence_extractor.get_non_parametric_evidence(ot_dists, topk_indices, support_labels)
                    alpha_nonparam = evidence_nonparam + 1
                    
                    alpha_fuse, u_fuse, C_fuse = fusion_module.ds_combination(alpha_param, alpha_nonparam)
                    
                    # Branch Uncertainties
                    S_param = torch.sum(alpha_param, dim=1)
                    u_param = Config.NUM_CLASSES / S_param
                    S_nonparam = torch.sum(alpha_nonparam, dim=1)
                    u_nonparam = Config.NUM_CLASSES / S_nonparam
                    
                    probs = alpha_fuse / torch.sum(alpha_fuse, dim=1, keepdim=True)
                    conf, preds = torch.max(probs, 1)
                    
                    min_ot_dist = ot_dists[:, 0]
                    
                    # Collect
                    ood_u_fuse.extend(u_fuse.cpu().numpy())
                    ood_u_param.extend(u_param.cpu().numpy())
                    ood_u_nonparam.extend(u_nonparam.cpu().numpy())
                    
                    # Log Sample Details
                    for i in range(images.size(0)):
                        detailed_logs.append({
                            "dataset_source": ood_name,
                            "true_label": -1, # OOD
                            "pred_label": preds[i].item(),
                            "is_correct": False,
                            "uncertainty_fuse": u_fuse[i].item(),
                            "uncertainty_param": u_param[i].item(),
                            "uncertainty_ot": u_nonparam[i].item(),
                            "conflict": C_fuse[i].item(),
                            "confidence": conf[i].item(),
                            "min_ot_dist": min_ot_dist[i].item()
                        })

            # Compute AUROC for all branches (Ablation)
            auroc_fuse = compute_auroc(np.array(id_uncertainties_fuse), np.array(ood_u_fuse))
            auroc_param = compute_auroc(np.array(id_uncertainties_param), np.array(ood_u_param))
            auroc_nonparam = compute_auroc(np.array(id_uncertainties_nonparam), np.array(ood_u_nonparam))
            
            logger.info(f"OOD {ood_name} Results:")
            logger.info(f"  AUROC (Fusion): {auroc_fuse:.4f}")
            logger.info(f"  AUROC (Param): {auroc_param:.4f}")
            logger.info(f"  AUROC (OT):    {auroc_nonparam:.4f}")

            ood_results[ood_name] = {
                "auroc_fusion": float(auroc_fuse),
                "auroc_parametric": float(auroc_param),
                "auroc_nonparametric": float(auroc_nonparam)
            }
            
            last_ood_uncertainties = ood_u_fuse

        except Exception as e:
            logger.error(f"Failed OOD {ood_name}: {e}")

    # Save Results
    results = {
        "dataset": Config.DATASET_NAME,
        "accuracy_parametric": float(acc_param),
        "accuracy_nonparametric": float(acc_nonparam),
        "accuracy_fusion": float(acc_fuse),
        "ece": float(ece_score),
        "auroc_ood": ood_results, # Nested dict
        "avg_uncertainty_id": float(total_uncertainty / total),
    }
    
    results_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME)
    if not os.path.exists(results_dir):
         os.makedirs(results_dir)
         
    save_results(results, results_dir, filename="metrics.json")
    
    # Save Detailed Logs
    df = pd.DataFrame(detailed_logs)
    df.to_csv(os.path.join(results_dir, "detailed_analysis.csv"), index=False)
    logger.info(f"Saved detailed analysis to {os.path.join(results_dir, 'detailed_analysis.csv')}")
    
    # Plot (Use last OOD, likely Noise)
    plot_inference_metrics(id_uncertainties_fuse, all_correct_fusion, last_ood_uncertainties)

def main():
    parser = argparse.ArgumentParser(description="Run TVI Inference")
    parser.add_argument("--config", type=str, default="conf/cifar10.json", help="Path to config file")
    args = parser.parse_args()
    
    # Load Config
    Config.load_config(args.config)
    
    # Set Seeds for Reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
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
