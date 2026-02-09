import torch
import numpy as np
import os
import time
from tqdm import tqdm
from src.config import Config
from src.dataset import get_dataloaders, get_ood_loader
from src.model import ResNetBackbone
from src.ot_module import OTMetric
from src.evidence_module import EvidenceExtractor
from src.fusion_module import DempsterShaferFusion
from src.utils import setup_logger, save_results, compute_ece, compute_auroc, compute_fpr95
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def load_backbone(device, logger):
    model = ResNetBackbone(num_classes=Config.NUM_CLASSES).to(device)
    ckpt_name = f"best_resnet18_{Config.DATASET_NAME}.pth"
    checkpoint_path = os.path.join(Config.Checkpoints_DIR, ckpt_name)
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Using random weights.")
    model.eval()
    return model

    model.eval()
    return model

def apply_bn_spatial(x, bn_layer):
    """
    Apply parameters of a BatchNorm1d layer to a spatial feature map (B, C, H, W).
    This ensures spatial features are in the same distribution space as the flattened (BN'd) features.
    """
    # x: (B, C, H, W)
    # bn_layer: nn.BatchNorm1d
    
    mean = bn_layer.running_mean.view(1, -1, 1, 1).to(x.device)
    var = bn_layer.running_var.view(1, -1, 1, 1).to(x.device)
    weight = bn_layer.weight.view(1, -1, 1, 1).to(x.device)
    bias = bn_layer.bias.view(1, -1, 1, 1).to(x.device)
    eps = bn_layer.eps
    
    # Standard BN: y = (x - mean) / sqrt(var + eps) * weight + bias
    out = (x - mean) / torch.sqrt(var + eps) * weight + bias
    return out

def kmeans_select(features, k, max_iter=20, device='cpu'):
    """
    Select k representative centroids using K-Means (PyTorch).
    features: (N, D)
    """
    N, D = features.shape
    if N <= k: return features
    
    # Init centroids (Random)
    indices = torch.randperm(N)[:k]
    centroids = features[indices].clone()
    
    for _ in range(max_iter):
        # Assign to nearest centroid
        # dists: (N, k)
        dists = torch.cdist(features, centroids)
        labels = torch.argmin(dists, dim=1)
        
        # Update centroids
        new_centroids = []
        for i in range(k):
            mask = (labels == i)
            if mask.sum() > 0:
                new_centroids.append(features[mask].mean(dim=0))
            else:
                # Re-init empty cluster
                random_idx = torch.randint(0, N, (1,)).item()
                new_centroids.append(features[random_idx])
        centroids = torch.stack(new_centroids)
        
    return centroids

def build_support_set(model, support_loader, device, logger, rebuild_support=False):
    support_path = os.path.join(Config.Checkpoints_DIR, f"{Config.DATASET_NAME}_support.pt")
    
    if os.path.exists(support_path) and not rebuild_support:
        logger.info(f"Loading cached support set from {support_path}")
        data = torch.load(support_path, map_location=device)
        # Check if new cache format
        if 'virtual_outliers' in data and 'precision_matrix' in data:
            # Handle backward compatibility with default react=1e9 (no clip)
            react = data.get('react_threshold', 1e9)
            return data['features'], data['labels'], data['virtual_outliers'], data['gamma_scale'], data['precision_matrix'], react
        else:
            logger.warning("Old cache format found. Rebuilding...")


    logger.info("Building Support Set Features...")
    
    samples_per_class = Config.NUM_SUPPORT_SAMPLES // Config.NUM_CLASSES
    # Temporary storage for ALL candidates
    candidate_features_per_class = {i: [] for i in range(Config.NUM_CLASSES)}
    
    # Determine max candidates to fetch (e.g. 5x desired size) to have good cluster pool
    # Or just fetch all training data? That might be too slow.
    logger.info("Building Support Set Features (Using Projected Features for Method 16)...")
    model.eval()
    
    candidate_features_per_class = [[] for _ in range(Config.NUM_CLASSES)]
    
    with torch.no_grad():
        for images, labels in tqdm(support_loader, desc="Extracting Candidates"):
            images, labels = images.to(device), labels.to(device)
            # Method 16: Extract Projected Features (128d)
            # x, features, logits, projected = model(images)
            _, _, _, projected = model(images)
            
            # Normalize Projected features (SupCon requirement)
            projected = torch.nn.functional.normalize(projected, dim=1)
            
            for i in range(images.size(0)):
                label = labels[i].item()
                # Store (128,)
                candidate_features_per_class[label].append(projected[i])
                
    support_features = []
    support_labels = []
    samples_per_class = Config.NUM_SUPPORT_SAMPLES // Config.NUM_CLASSES
    
    logger.info(f"Selecting {samples_per_class} prototypes per class using K-Means (Projected Space)...")
    
    for i in range(Config.NUM_CLASSES):
        candidates = torch.stack(candidate_features_per_class[i]) # (M, 128)
        
        # Method 16: K-Means on Projected Features
        centroids = kmeans_select(candidates, samples_per_class, device=device) # (K, 128)
        
        # Find nearest real candidates to centroids
        dists = torch.cdist(centroids, candidates)
        _, nearest_indices = torch.min(dists, dim=1)
        
        selected = candidates[nearest_indices] # (K, 128)
        
        # Reshape to (K, 128, 1, 1) for compatibility with spatial OT module
        selected = selected.unsqueeze(2).unsqueeze(3)
        
        support_features.append(selected)
        support_labels.extend([i] * samples_per_class)
        
    support_features = torch.cat(support_features, dim=0).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    
    # --- React Implementation (Clip High Activations) ---
    logger.info(f"Applying React (Rectified Activation) w/ p={Config.REACT_PERCENTILE}")
    # Flatten strictly for stats: (N, C*H*W) or (N*H*W, C)?
    # We clip activations element-wise regardless of spatial position.
    # Move to CPU and use Numpy to avoid 'quantile() input tensor is too large' error
    # PyTorch quantile has element limit even on CPU, Numpy is robust.
    flat_support_all = support_features.view(-1).cpu().numpy()
    # Find global percentile threshold
    val = np.percentile(flat_support_all, Config.REACT_PERCENTILE)
    clip_threshold = torch.tensor(val, device=device, dtype=support_features.dtype)
    logger.info(f"React Threshold: {clip_threshold:.4f}")
    
    # Clip Support Features
    support_features = torch.clamp(support_features, max=clip_threshold)
    
    # --- Robust Mahalanobis (Tied statistics) ---
    # We use FLATTENED (GAP) features for Covariance Calculation to match Global semantics
    # Global Average Pooling manually:
    support_flat_gap = torch.mean(support_features, dim=(2, 3)) # (N, C)
    
    prototypes = []
    centered_features = []
    
    for i in range(Config.NUM_CLASSES):
        mask = (support_labels == i)
        # GAP features for class i
        class_feats = support_flat_gap[mask]
        
        # Mean
        mu = class_feats.mean(dim=0)
        prototypes.append(mu)
        
        # Center
        centered_features.append(class_feats - mu)
        
    prototypes = torch.stack(prototypes) # (K, C)
    centered_features = torch.cat(centered_features, dim=0) # (N, C)
    
    # Covariance: (X.T @ X) / (N - 1)
    N_samples = centered_features.size(0)
    cov = torch.matmul(centered_features.T, centered_features) / (N_samples - 1)
    
    # Robust Shrinkage
    alpha = 1e-4
    cov = (1 - alpha) * cov + alpha * torch.eye(cov.size(0)).to(device)
    
    # Precision Matrix
    try:
        precision_matrix = torch.inverse(cov)
    except RuntimeError:
        logger.warning("Covariance matrix is singular! Using pseudo-inverse.")
        precision_matrix = torch.pinverse(cov)
        
    logger.info("Computed Robust Precision Matrix (on GAP features).")
    
    # Re-shape Prototypes for VOS (K, C, 1, 1)
    # Ensure they match spatial resolution of support set for OT
    H, W = support_features.shape[2], support_features.shape[3]
    prototypes = prototypes.view(Config.NUM_CLASSES, -1, 1, 1).expand(-1, -1, H, W).clone()
    
    # 2. Global Center (Mean across classes)
    global_center = torch.mean(prototypes, dim=0) # (C, H, W)
    
    # 3. Virtual Outliers
    vos_type = getattr(Config, 'VOS_TYPE', 'radial') # Default to radial if not set
    beta = Config.VO_BETA
    num_vos_per_class = 5 
    virtual_outliers_list = []
    
    if vos_type == 'boundary':
        # Boundary Sampling: Target nearest other class
        K_classes = prototypes.size(0)
        flat_protos = prototypes.view(K_classes, -1)
        dists_proto = torch.cdist(flat_protos, flat_protos)
        dists_proto.fill_diagonal_(float('inf'))
        _, nearest_indices = torch.min(dists_proto, dim=1)
        
        for i in range(K_classes):
            p_k = prototypes[i]
            p_near = prototypes[nearest_indices[i]]
            direction = p_near - p_k
            
            for _ in range(num_vos_per_class):
                # Use beta + N(0, 0.1)
                lambda_sample = beta + torch.randn(1).item() * 0.1
                epsilon = torch.randn_like(p_k) * 0.1 
                vo = p_k + lambda_sample * direction + epsilon
                virtual_outliers_list.append(vo)
                
    else: # radial
        # Classic Radial Sampling: p_k + beta * (p_k - global_center)
        # Pushes outliers outward from center
        for proto in prototypes:
            direction = proto - global_center
            
            for _ in range(num_vos_per_class):
                epsilon = torch.randn_like(proto) * 0.1 
                vo = proto + beta * direction + epsilon
                virtual_outliers_list.append(vo)
            
    virtual_outliers = torch.stack(virtual_outliers_list)
    
    # Clip VOS too? Yes, consistent with feature space.
    virtual_outliers = torch.clamp(virtual_outliers, max=clip_threshold)
    
    # 4. Adaptive Gamma (using Mahalanobis/Euclidean logic)
    # Recalculate based on clipped features
    # Let's compute average distance of samples to their class prototype
    total_dist = 0.0
    count = 0
    
    # Pre-compute Cholesky of Precision for Mahalanobis Distance check
    L_inv = torch.linalg.cholesky(precision_matrix) # P = L @ L.T ... wait.
    # We want dist = (x-u).T P (x-u) = || L.T(x-u) ||^2 where P = L @ L.T
    # So we multiply by L.T. 
    # Actually if P = U S U.T, transform is U sqrt(S).
    
    for i in range(Config.NUM_CLASSES):
        # Take first 10 samples (clipped)
        mask = (support_labels == i)
        # Use GAP for Gamma Calc (matches Precision Matrix)
        # support_features: (N, C, H, W) -> (N, C)
        samples = support_features[mask][:10].mean(dim=(2, 3)) 
        
        # Get dynamic channel dim (e.g. 512 or 128)
        C = samples.size(1)
        
        # prototypes: (K, C, H, W) -> (C) -> (1, C)
        proto = prototypes[i].mean(dim=(1, 2)).view(1, C)
        
        # Mahalanobis Distance
        diff = samples - proto
        # diff: (N, 512)
        # Dist = sum (diff @ P) * diff
        # (N, D) @ (D, D) -> (N, D)
        term1 = torch.matmul(diff, precision_matrix)
        dists = torch.sum(term1 * diff, dim=1)
        
        total_dist += torch.sum(dists).item()
        count += len(samples)
    
    avg_dist = total_dist / count
    gamma_scale = 1.0 / (avg_dist + 1e-6)
    
    logger.info(f"Support Set Created. Shape: {support_features.shape}")
    logger.info(f"Generated {virtual_outliers.shape[0]} Virtual Outliers.")
    logger.info(f"Adaptive Gamma: {gamma_scale:.4f}")
    
    # Cache (including new items)
    torch.save({
        'features': support_features, 
        'labels': support_labels,
        'virtual_outliers': virtual_outliers,
        'gamma_scale': gamma_scale,
        'precision_matrix': precision_matrix,
        'react_threshold': clip_threshold
    }, support_path)
    logger.info(f"Saved support set to {support_path}")
    
    return support_features, support_labels, virtual_outliers, gamma_scale, precision_matrix, clip_threshold

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

def evaluate(model, test_loader, support_features, support_labels, virtual_outliers, gamma_scale, precision_matrix, clip_threshold, device, logger, args):
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
    
    # Time measurement
    start_time = time.time()
    
    logger.info(f"Starting ID Inference... (Baseline Mode: {args.baseline})")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images, labels = images.to(device), labels.to(device)
            # ... process one batch ...
            
            # 1. Backbone
            # Return: spatial, flat, logits, projected
            # Method 16: Use Projected Features (128d)
            _, _, logits, projected = model(images)
            
            # Normalize (SupCon Space)
            projected = torch.nn.functional.normalize(projected, dim=1)
            
            # Reshape for OT Module (B, 128, 1, 1). 
            # This turns Sinkhorn into simple Element-wise Distance between vectors (since 1x1 grid).
            features = projected.unsqueeze(2).unsqueeze(3)
            
            # Note: We skip React Clipping for Projected Features as they are already normalized/squashed?
            # SupCon features are unit vectors. React makes no sense here.
            # However, if clip_threshold > 0, we might clamp? No, let's skip for SupCon.
            
            # 2. Parametric
            evidence_param = evidence_extractor.get_parametric_evidence(logits)
            alpha_param = evidence_param + 1
            
            # Branch Uncertainties
            S_param = torch.sum(alpha_param, dim=1)
            u_param = Config.NUM_CLASSES / S_param
            _, pred_param = torch.max(alpha_param, 1)

            # Initialize other variables with defaults
            alpha_nonparam = torch.ones_like(alpha_param) # Dummy
            alpha_fuse = alpha_param # In baseline, fusion = param
            u_nonparam = torch.zeros_like(u_param)
            u_fuse = u_param
            C_fuse = torch.zeros_like(u_param)
            pred_nonparam = pred_param # Dummy
            pred_fuse = pred_param
            min_ot_dist = torch.zeros_like(u_param)
            confidence = torch.max(alpha_param / S_param.unsqueeze(1), 1)[0]
            
            if not args.baseline:
                # 3. Non-Parametric with Virtual Outliers
                ot_dists, topk_indices, vo_dists = ot_metric.compute_batch_ot(features, support_features, support_labels, virtual_outliers=virtual_outliers, precision_matrix=precision_matrix)
                
                # Use Adaptive Gamma & Virtual Outlier Discounting
                evidence_nonparam = evidence_extractor.get_non_parametric_evidence(
                    ot_dists, topk_indices, support_labels, 
                    gamma_scale=gamma_scale, vo_distances=vo_dists
                )
                alpha_nonparam = evidence_nonparam + 1
                
                # 4. Fusion
                # Disable Entropy Discounting to preserve "Unknown" signal from Parametric branch
                # entropy_param = fusion_module.compute_entropy(alpha_param) 
                
                # Fusion
                alpha_fuse, u_fuse_ds, C_fuse = fusion_module.ds_combination(alpha_param, alpha_nonparam, discount_factor=None)
                
                # Update vars
                S_nonparam = torch.sum(alpha_nonparam, dim=1)
                u_nonparam = Config.NUM_CLASSES / S_nonparam
                
                _, pred_nonparam = torch.max(alpha_nonparam, 1)
                _, pred_fuse = torch.max(alpha_fuse, 1)
                probs_fuse = alpha_fuse / torch.sum(alpha_fuse, dim=1, keepdim=True)
                confidence, _ = torch.max(probs_fuse, 1)
                
                min_ot_dist = ot_dists[:, 0]
                
                # INTEGRATE CONFLICT INTO UNCERTAINTY (OOD Score)
                # Method 13: Boost Conflict Weight to 5.0
                # High conflict -> High OOD probability
                if len(u_param.shape) > 1: u_param = u_param.squeeze()
                if len(u_nonparam.shape) > 1: u_nonparam = u_nonparam.squeeze()
                if len(C_fuse.shape) > 1: C_fuse = C_fuse.squeeze()
                
                u_fuse = torch.max(u_param, u_nonparam) + C_fuse * 5.0
            
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
            if not args.baseline:
                min_ot_dist = ot_dists[:, 0]
            else:
                min_ot_dist = torch.zeros(images.size(0)).to(device)
            
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
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_latency_ms = (total_time / total) * 1000 # ms per sample
    
    logger.info(f"Parametric Acc: {acc_param:.2f}%")
    logger.info(f"Non-Parametric Acc: {acc_nonparam:.2f}%")
    logger.info(f"Fused Acc: {acc_fuse:.2f}%")
    logger.info(f"ECE Score: {ece_score:.4f}")
    logger.info(f"Inference Latency: {avg_latency_ms:.2f} ms/sample")
    
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
                    # Method 16: Use Projected Features for OOD
                    _, _, logits, projected = model(images)
                    
                    # Normalize (SupCon Space)
                    projected = torch.nn.functional.normalize(projected, dim=1)
                    
                    # Reshape for OT (1x1)
                    features = projected.unsqueeze(2).unsqueeze(3)
                    
                    # Skip React Clipping (SupCon features are normalized)
                    
                    evidence_param = evidence_extractor.get_parametric_evidence(logits)
                    alpha_param = evidence_param + 1
                    
                    # Fix: Calculate u_param for current batch!
                    S_param_init = torch.sum(alpha_param, dim=1)
                    u_param = Config.NUM_CLASSES / S_param_init
                    
                    # Defaults
                    alpha_nonparam = torch.ones_like(alpha_param)
                    alpha_fuse = alpha_param
                    u_nonparam = torch.zeros(images.size(0)).to(device)
                    u_fuse = Config.NUM_CLASSES / torch.sum(alpha_param, dim=1) # Default to param U
                    C_fuse = torch.zeros_like(u_fuse)
                    min_ot_dist = torch.zeros_like(u_fuse)
                    
                    if not args.baseline:
                        ot_dists, topk_indices, vo_dists = ot_metric.compute_batch_ot(features, support_features, support_labels, virtual_outliers=virtual_outliers, precision_matrix=precision_matrix)
                        
                        evidence_nonparam = evidence_extractor.get_non_parametric_evidence(
                            ot_dists, topk_indices, support_labels, 
                            gamma_scale=gamma_scale, vo_distances=vo_dists
                        )
                        alpha_nonparam = evidence_nonparam + 1
                        
                        # Conservative Fusion (Method 10)
                        # entropy_param = fusion_module.compute_entropy(alpha_param)
                        alpha_fuse, u_fuse_ds, C_fuse = fusion_module.ds_combination(alpha_param, alpha_nonparam, discount_factor=None)
                        
                        # Recalculate u_nonparam locally if needed for max
                        S_nonparam = torch.sum(alpha_nonparam, dim=1)
                        u_nonparam = Config.NUM_CLASSES / S_nonparam

                        # Integrate Conflict (Conservative)
                        if len(u_param.shape) > 1: u_param = u_param.squeeze()
                        if len(u_nonparam.shape) > 1: u_nonparam = u_nonparam.squeeze()
                        if len(C_fuse.shape) > 1: C_fuse = C_fuse.squeeze()
                        
                        u_fuse = torch.max(u_param, u_nonparam) + C_fuse * 5.0
                        
                        min_ot_dist = ot_dists[:, 0]
                    
                    # Branch Uncertainties
                    S_param = torch.sum(alpha_param, dim=1)
                    u_param = Config.NUM_CLASSES / S_param
                    if not args.baseline:
                         S_nonparam = torch.sum(alpha_nonparam, dim=1)
                         u_nonparam = Config.NUM_CLASSES / S_nonparam
                    
                    probs = alpha_fuse / torch.sum(alpha_fuse, dim=1, keepdim=True)
                    conf, preds = torch.max(probs, 1)
                    
                    if not args.baseline:
                        min_ot_dist = ot_dists[:, 0]
                    else:
                        min_ot_dist = torch.zeros(images.size(0)).to(device)

                    
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
            
            # Compute FPR@95 for all branches
            fpr_fuse = compute_fpr95(np.array(id_uncertainties_fuse), np.array(ood_u_fuse))
            fpr_param = compute_fpr95(np.array(id_uncertainties_param), np.array(ood_u_param))
            fpr_nonparam = compute_fpr95(np.array(id_uncertainties_nonparam), np.array(ood_u_nonparam))
            
            logger.info(f"OOD {ood_name} Results:")
            logger.info(f"  AUROC (Fusion): {auroc_fuse:.4f} | FPR@95: {fpr_fuse:.4f}")
            logger.info(f"  AUROC (Param): {auroc_param:.4f} | FPR@95: {fpr_param:.4f}")
            logger.info(f"  AUROC (OT):    {auroc_nonparam:.4f} | FPR@95: {fpr_nonparam:.4f}")

            ood_results[ood_name] = {
                "auroc_fusion": float(auroc_fuse),
                "auroc_parametric": float(auroc_param),
                "auroc_nonparametric": float(auroc_nonparam),
                "fpr95_fusion": float(fpr_fuse),
                "fpr95_parametric": float(fpr_param),
                "fpr95_nonparametric": float(fpr_nonparam)
            }
            
            last_ood_uncertainties = ood_u_fuse

        except Exception as e:
            logger.error(f"Failed OOD {ood_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save Results
    results = {
        "dataset": Config.DATASET_NAME,
        "config": {
             "metric": Config.METRIC_TYPE,
             "fusion": Config.FUSION_TYPE,
             "k_neighbors": Config.K_NEIGHBORS,
             "sinkhorn_eps": Config.SINKHORN_EPS,
             "rbf_gamma": Config.RBF_GAMMA,
             "support_size": Config.NUM_SUPPORT_SAMPLES
        },
        "accuracy_parametric": float(acc_param),
        "accuracy_nonparametric": float(acc_nonparam),
        "accuracy_fusion": float(acc_fuse),
        "ece": ece_score,
        "avg_latency_ms": avg_latency_ms,
        "auroc_ood": ood_results, # Nested dict
        "avg_uncertainty_id": float(total_uncertainty / total),
    }
    
    results_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME)
    if not os.path.exists(results_dir):
         os.makedirs(results_dir)
         
    save_results(results, results_dir, filename="metrics.json")
    
    # Save Detailed Logs and Summary to Excel
    df_detailed = pd.DataFrame(detailed_logs)
    
    # Prepare Summary DataFrame (Sheet 2)
    summary_data = {
        "Dataset": [Config.DATASET_NAME],
        "Task Note": [args.task_note],
        "Acc Param": [acc_param],
        "Acc OT": [acc_nonparam],
        "Acc Fusion": [acc_fuse],
        "ECE": [ece_score],
        "Metric": [Config.METRIC_TYPE],
        "Fusion": [Config.FUSION_TYPE]
    }
    
    # Add OOD metrics dynamically
    for ood_name, metrics in ood_results.items():
        summary_data[f"AUROC Fusion ({ood_name})"] = [metrics['auroc_fusion']]
        summary_data[f"FPR95 Fusion ({ood_name})"] = [metrics['fpr95_fusion']]
        summary_data[f"AUROC Param ({ood_name})"] = [metrics['auroc_parametric']]
        summary_data[f"AUROC OT ({ood_name})"] = [metrics['auroc_nonparametric']]
        
    df_summary = pd.DataFrame(summary_data)
    
    excel_path = os.path.join(results_dir, "analysis_report.xlsx")
    
    # Robust Append Strategy: Load -> Concat -> Write
    existing_summary = pd.DataFrame()
    if os.path.exists(excel_path):
        try:
            # Read existing summary
            existing_summary = pd.read_excel(excel_path, sheet_name='Summary Metrics')
        except:
            # Maybe sheet doesn't exist
            pass
            
    if not existing_summary.empty:
        # Check if this config already exists? Or just append.
        df_summary = pd.concat([existing_summary, df_summary], ignore_index=True)
        
    with pd.ExcelWriter(excel_path, mode='w') as writer:
        # If detailed analysis is huge, maybe skip it if user said it's redundant? 
        # User said "is detailed_analysis.csv a subset... if so remove". 
        # But here we are saving to Excel Sheet 1. 
        # The user said "detailed_analysis.csv ... redundancy... remove".
        # So I will KEEP the Excel Sheet 1 (Detailed Samples) but NOT save the CSV file.
        df_detailed.to_excel(writer, sheet_name='Detailed Samples', index=False)
        df_summary.to_excel(writer, sheet_name='Summary Metrics', index=False)
        
    logger.info(f"Saved analysis report to {excel_path}")
    
    # Removed separate CSV export as per user request
    # df_detailed.to_csv(...) 

    # Plot (Use last OOD, likely Noise)
    if not args.baseline:
         plot_inference_metrics(id_uncertainties_fuse, all_correct_fusion, last_ood_uncertainties)

def main():
    parser = argparse.ArgumentParser(description="Run TVI Inference")
    parser.add_argument("--config", type=str, default="conf/cifar10.json", help="Path to config file")
    parser.add_argument("--task_note", type=str, default="Base", help="Note for the experiment (e.g. Ablation, Baseline)")
    parser.add_argument("--baseline", action="store_true", help="Run in baseline mode (only Parametric branch, skip OT/Fusion)")
    parser.add_argument("--rebuild_support", action="store_true", help="Force rebuilding of the support set cache")
    parser.add_argument("--log_suffix", type=str, default=None, help="Suffix for the log file name (e.g. experiment_baseline.log)")
    
    # Ablation / Sensitivity Overrides
    parser.add_argument("--k_neighbors", type=int, default=None, help="Override K Neighbors")
    parser.add_argument("--rbf_gamma", type=float, default=None, help="Override RBF Gamma")
    parser.add_argument("--sinkhorn_eps", type=float, default=None, help="Override Sinkhorn Epsilon")
    parser.add_argument("--fusion_type", type=str, default=None, help="Override Fusion Type")
    parser.add_argument("--metric_type", type=str, default=None, help="Override Metric Type")
    parser.add_argument("--support_size", type=int, default=None, help="Override Support Set Size")
    
    args = parser.parse_args()
    
    # Load Config
    Config.load_config(args.config)
    
    # Apply Overrides
    if args.k_neighbors is not None: Config.K_NEIGHBORS = args.k_neighbors
    if args.rbf_gamma is not None: Config.RBF_GAMMA = args.rbf_gamma
    if args.sinkhorn_eps is not None: Config.SINKHORN_EPS = args.sinkhorn_eps
    if args.fusion_type is not None: Config.FUSION_TYPE = args.fusion_type
    if args.metric_type is not None: Config.METRIC_TYPE = args.metric_type
    if args.support_size is not None: Config.NUM_SUPPORT_SAMPLES = args.support_size
    
    # Set Seeds for Reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    results_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME)
    if not os.path.exists(results_dir):
         os.makedirs(results_dir)
    from src.utils import setup_logger
    log_name = f"experiment_{args.log_suffix}" if args.log_suffix else "experiment"
    logger = setup_logger(results_dir, name=log_name)
    
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")
    logger.info("="*30)
    logger.info("Inference Configuration:")
    logger.info(f"Dataset: {Config.DATASET_NAME}")
    logger.info(f"Metric: {Config.METRIC_TYPE}")
    logger.info(f"Fusion: {Config.FUSION_TYPE}")
    logger.info(f"K Neighbors: {Config.K_NEIGHBORS}")
    logger.info(f"Sinkhorn Eps: {Config.SINKHORN_EPS}")
    logger.info(f"RBF Gamma: {Config.RBF_GAMMA}")
    logger.info(f"Support Size: {Config.NUM_SUPPORT_SAMPLES}")
    logger.info(f"Seed: {Config.SEED}")
    logger.info("="*30)
    
    # Load Data
    _, support_loader, test_loader = get_dataloaders()
    
    # Load Model
    model = load_backbone(device, logger)
    
    # Build Support Set
    support_features, support_labels, virtual_outliers, gamma_scale, precision_matrix, clip_threshold = build_support_set(model, support_loader, device, logger, rebuild_support=args.rebuild_support)
    
    # Run Evaluation
    # Run Evaluation
    evaluate(model, test_loader, support_features, support_labels, virtual_outliers, gamma_scale, precision_matrix, clip_threshold, device, logger, args)

if __name__ == "__main__":
    main()
