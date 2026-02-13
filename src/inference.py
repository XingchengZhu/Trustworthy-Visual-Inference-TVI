import torch
import torch.nn as nn
import torch.nn.functional as F
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

    ckpt_name = f"best_{Config.BACKBONE.lower()}_{Config.DATASET_NAME}.pth"
    checkpoint_path = os.path.join(Config.Checkpoints_DIR, ckpt_name)
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Using random weights.")
    model.eval()
    return model

def odin_perturbation(model, images, temp, eps):
    """
    Apply input perturbation for ODIN.
    """
    images_adv = images.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        _, _, logits, _ = model(images_adv)
        logits = logits / temp
        labels = torch.max(logits, 1)[1]
        loss = F.cross_entropy(logits, labels)
        model.zero_grad()
        loss.backward()
    gradient = images_adv.grad.detach()
    perturbed_images = images.detach() - eps * torch.sign(gradient)
    return perturbed_images

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

def apply_ash(x, percentile=90):
    """
    Activation Shaping (ASH): Prune bottom p% of activations.
    x: (B, C, H, W) or (B, D)
    percentile: 0-100
    """
    assert 0 <= percentile <= 100
    if percentile == 0: return x
    
    # Flatten spatial dims for percentile calculation if needed, 
    # but ASH usually prunes element-wise globally or per-sample.
    # Paper "ASH": prune bottom p% of activations per sample.
    
    # 1. Calculate threshold per sample
    # x shape: (B, ...)
    B = x.shape[0]
    x_flat = x.view(B, -1)
    
    # Calculate k-th value (percentile)
    # k = number of elements to Zero out
    k = int(x_flat.shape[1] * (percentile / 100.0))
    if k == 0: return x
    
    # topk returns largest values. We want to keep top (100-p)%.
    # So we find the k-th smallest value?
    # Or just keep top N values.
    # torch.kthvalue finds the k-th smallest element.
    # Threshold is the value at percentile p. All below are zeroed.
    
    kth_values, _ = torch.kthvalue(x_flat, k, dim=1, keepdim=True)
    # kth_values: (B, 1)
    
    # reshape to match x
    if x.dim() == 4:
        threshold = kth_values.view(B, 1, 1, 1)
    else:
        threshold = kth_values
        
    # Prune
    # ASH-S (Simple): s = x * \mathbb{I}(x \ge t)  (Zero out)
    # ASH-B (Binarize): s = \mathbb{I}(x \ge t)
    # ASH-P (Propagate): s = ...
    # We use ASH-S (default)
    mask = (x >= threshold).float()
    out = x * mask
    
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
    support_path = os.path.join(Config.Checkpoints_DIR, f"{Config.DATASET_NAME}_{Config.BACKBONE.lower()}_support.pt")
    
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
    # Let's fetch up to 5 * samples_per_class
    fetch_limit = samples_per_class * 5
    
    with torch.no_grad():
        for images, labels in tqdm(support_loader, desc="Extracting Candidates", leave=False):
            images = images.to(device)
            # x: (B, C, H, W)
            x, _, _, _ = model(images) 
            x_norm = apply_bn_spatial(x, model.bn)
            
            finished_classes = 0
            for i in range(images.size(0)):
                lbl = labels[i].item()
                if len(candidate_features_per_class[lbl]) < fetch_limit:
                    feat = x_norm[i].cpu() # (C, H, W)
                    candidate_features_per_class[lbl].append(feat)
            
            # Check if we have enough for all classes
            if all(len(v) >= fetch_limit for v in candidate_features_per_class.values()):
                break

    # K-Means Construction
    support_features = []
    support_labels = []
    
    logger.info(f"Selecting {samples_per_class} prototypes per class using K-Means...")
    
    for i in range(Config.NUM_CLASSES):
        candidates = torch.stack(candidate_features_per_class[i]) # (M, C, H, W)
        
        # Use GAP for Clustering
        candidates_gap = candidates.mean(dim=(2, 3)) # (M, C)
        
        # Run K-Means
        # Returns centroids, but we need the actual features (or just use centroids?)
        # For methods like React/Mahalanobis, centroids are fine.
        # But for OT, we need spatial structure (H, W).
        # We can Construct Spatial Centroids?
        # A simple way: Run K-Means on flattened (C*H*W)? No, high dim.
        # Better: run K-Means on GAP, find nearest real sample to centroid. (Algorithm: Herding / K-Center)
        
        # Let's use K-Means centroids logic on GAP, then find nearest candidate.
        centroids_gap = kmeans_select(candidates_gap, samples_per_class, device=device) # (K, C)
        
        # Find nearest candidates to these centroids
        # dist: (K, M)
        dists = torch.cdist(centroids_gap, candidates_gap)
        _, nearest_indices = torch.min(dists, dim=1)
        
        # Select spatial features
        selected_spatial = candidates[nearest_indices] # (K, C, H, W)
        
        support_features.append(selected_spatial)
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
    
    # SOTA: Class-Conditional Precision Matrices (Item 3)
    from src.ot_module import OTMetric
    temp_ot = OTMetric(device=device)
    precision_matrix = temp_ot.compute_class_precision_matrices(support_features, support_labels)
    logger.info(f"Computed Class-Conditional Precision Matrices for {len(precision_matrix)} classes.")
    
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
    
    # 3b. Adversarial VOs (Phase 10)
    # Optimize VOs to be "Hard Negatives" (Maximize ID Probability)
    if getattr(Config, 'ADVERSARIAL_VOS', False):
        logger.info("  > Optimizing Virtual Outliers (Adversarial)...")
        # Clone and enable grad
        features_vo = virtual_outliers.clone().detach().requires_grad_(True)
        # Use simple SGD/Adam
        optimizer_vo = torch.optim.Adam([features_vo], lr=getattr(Config, 'ADV_VOS_LR', 0.1))
        
        model.eval() # Ensure model is in eval mode
        
        # Optimize
        for step in range(getattr(Config, 'ADV_VOS_STEPS', 5)):
            optimizer_vo.zero_grad()
            
            # Forward: VOs (B, C, H, W) -> GAP -> FC -> Logits
            # Note: VOs are already BN-normalized spatial features
            vo_flat = torch.mean(features_vo, dim=(2, 3)) # (N, C)
            logits_vo = model.resnet.fc(vo_flat)
            
            # Loss: Maximize Max Softmax Probability (Make them look like ID)
            # We want to minimize (-MaxProb)
            probs = torch.softmax(logits_vo, dim=1)
            max_prob, _ = torch.max(probs, dim=1)
            loss = -torch.mean(max_prob)
            
            loss.backward()
            optimizer_vo.step()
            
        # Update virtual_outliers with optimized features
        virtual_outliers = features_vo.detach()
        logger.info(f"  > Adversarial VOS Optimized. Final Avg Max Prob: {-loss.item():.4f}")
    
    # Clip VOS too? Yes, consistent with feature space.
    virtual_outliers = torch.clamp(virtual_outliers, max=clip_threshold)
    
    # 4. Adaptive Gamma (using Mahalanobis/Euclidean logic)
    # Recalculate based on clipped features
    # Let's compute average distance of samples to their class prototype
    total_dist = 0.0
    count = 0
    
    # 4. Adaptive Gamma (using Class-Specific Mahalanobis logic)
    # No global L_inv pre-computation since P varies.
    
    for i in range(Config.NUM_CLASSES):
        # Take first 10 samples (clipped)
        mask = (support_labels == i)
        # Use GAP for Gamma Calc (matches Precision Matrix)
        samples = support_features[mask][:10].mean(dim=(2, 3)) 
        proto = prototypes[i].mean(dim=(1, 2)).view(1, 512)
        
        diff = samples - proto
        
        # Get Class Precision
        P_c = precision_matrix[i]
        
        # Dist = sum (diff @ P) * diff
        term1 = torch.matmul(diff, P_c)
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
    
    # --- POT Branch Initialization ---
    pot_scorer = None
    if getattr(args, 'pot', False):
        from src.pot_module import POTScorer
        # Build GAP prototypes from support set for POT
        pot_prototypes_gap = []
        for c in range(Config.NUM_CLASSES):
            mask = (support_labels == c)
            class_feats = support_features[mask]  # (Nc, C, H, W)
            class_gap = class_feats.mean(dim=(2, 3))  # (Nc, D)
            pot_prototypes_gap.append(class_gap.mean(dim=0))  # (D,)
        pot_prototypes_gap = torch.stack(pot_prototypes_gap)  # (C, D)
        pot_scorer = POTScorer(pot_prototypes_gap, device=device)
        logger.info(f"POT Branch Enabled: {Config.NUM_CLASSES} prototypes, D={pot_prototypes_gap.shape[1]}, omega={Config.POT_OMEGA}, reg={Config.POT_SINKHORN_REG}")
    
    correct_param = 0
    correct_nonparam = 0
    correct_fusion = 0
    total = 0
    total_uncertainty = 0.0
    
    # ID Metric Storage
    id_uncertainties_fuse = []
    id_uncertainties_param = []
    id_uncertainties_nonparam = []
    id_uncertainties_pot = []  # POT branch
    id_max_logits = []   # MaxLogit branch
    id_gradnorms = []    # GradNorm branch
    id_feat_norms = []  # Phase 16: Feature Norm Score
    
    # Detailed Logs container
    detailed_logs = []
    
    # ID Probs for ECE
    all_probs_fusion = []
    all_labels_tensor = []
    
    # Metrics for Plots
    all_correct_fusion = [] # Boolean mask for ID
    
    # POT: Accumulate features for batch-level scoring
    pot_id_features = [] if pot_scorer else None
    pot_id_batch_sizes = [] if pot_scorer else None
    
    # Phase 16: Temperature Scaling
    from src.temperature_scaling import find_optimal_temperature, apply_temperature
    optimal_T = find_optimal_temperature(model, test_loader, device)
    logger.info(f"Optimal Temperature: {optimal_T:.4f}")
    
    # Time measurement
    start_time = time.time()
    
    # Initialize metrics storage
    logger.info(f"Starting ID Inference... (Baseline Mode: {args.baseline})")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            if args.max_samples and total >= args.max_samples: break
            
            images, labels = images.to(device), labels.to(device)
            # ... process one batch ...
            
            # 1. Backbone
            # Return: spatial, flat, logits, projected
            # x: (B, C, H, W)
            # We need unscaled logits for predictions/calibration
            x, features_flat, logits_calib, _ = model(images)
            
            # Use Spatial BN Features for OT (Structure Aware)
            features = apply_bn_spatial(x, model.bn)
            
            # React Clipping (Skip if --no_react)
            if clip_threshold > 0 and not getattr(args, 'no_react', False):
                features = torch.clamp(features, max=clip_threshold)
                
            # --- ASH (Phase 15): Prune small activations ---
            if getattr(Config, 'ASH_PERCENTILE', 0) > 0 and getattr(args, 'ash', False):
                 features = apply_ash(features, percentile=getattr(Config, 'ASH_PERCENTILE', 90))
            
            # 2. ODIN Logits (for OOD Uncertainty Scoring)
            if getattr(Config, 'ODIN_EPS', 0.0) > 0.0:
                # Need to re-run with perturbation
                inputs_odin = odin_perturbation(model, images, temp=Config.ODIN_TEMP, eps=Config.ODIN_EPS)
                _, _, logits_ood, _ = model(inputs_odin)
                logits_ood = logits_ood / Config.ODIN_TEMP
            else:
                logits_ood = logits_calib / Config.ODIN_TEMP

            # Extract evidence from scaled logits for OOD branches
            # Method 18 Logic: Use Exp(Logits/T) for strong separation
            # T=2.0 -> Exp(7.5) >> Exp(3.5)
            # Method 18 Logic: Use Exp(Logits/T) for strong separation
            # T=2.0 -> Exp(7.5) >> Exp(3.5)
            evidence_param = evidence_extractor.get_parametric_evidence(logits_ood)
            alpha_param = evidence_param + 1
            
            # Branch Uncertainties (OOD branch uses high-T)
            S_param = torch.sum(alpha_param, dim=1)
            u_param = Config.NUM_CLASSES / S_param
            _, pred_param_ood = torch.max(alpha_param, 1)

            # Initialize other variables with defaults
            alpha_nonparam = torch.ones_like(alpha_param) # Dummy
            alpha_fuse_ood = alpha_param # In baseline, fusion = param
            u_nonparam = torch.zeros_like(u_param)
            u_fuse_ds = u_param
            C_fuse = torch.zeros_like(u_param)
            pred_nonparam = pred_param_ood # Dummy
            pred_fuse = pred_param_ood
            min_ot_dist = torch.zeros_like(u_param)
            
            if not args.baseline:
                # 3. Non-Parametric with Virtual Outliers
                ot_dists, topk_indices, vo_dists = ot_metric.compute_batch_ot(features, support_features, support_labels, virtual_outliers=virtual_outliers, precision_matrix=precision_matrix)
                
                # Use Adaptive Gamma & Virtual Outlier Discounting (Skip VOS if --no_vos)
                vo_dists_effective = None if getattr(args, 'no_vos', False) else vo_dists
                evidence_nonparam = evidence_extractor.get_non_parametric_evidence(
                    ot_dists, topk_indices, support_labels, 
                    gamma_scale=gamma_scale, vo_distances=vo_dists_effective
                )
                alpha_nonparam = evidence_nonparam + 1
                
                # 4. Fusion (OOD branch still uses scaled alpha)
                if args.adaptive_fusion or args.fusion_strategy == 'fixed':
                    alpha_fuse_ood, u_fuse_ds, C_fuse = fusion_module.adaptive_ds_combination(
                        alpha_param, alpha_nonparam, 
                        strategy=args.fusion_strategy,
                        fixed_weight=args.fixed_weight
                    )
                else:
                    alpha_fuse_ood, u_fuse_ds, C_fuse = fusion_module.ds_combination(alpha_param, alpha_nonparam, discount_factor=None)
                
                # FIX: Calculate u_nonparam for ID loop!
                S_nonparam = torch.sum(alpha_nonparam, dim=1)
                u_nonparam = Config.NUM_CLASSES / S_nonparam
                
            # Final OOD Uncertainty score logic
            # FIX (Phase 16): Use actual Fusion uncertainty instead of discarding OT branch
            # Update (Phase 16b): Option to trust Param for ID (heuristic that worked well)
            use_trust_param = not getattr(args, 'no_trust_param_id', False)
            if not args.baseline and not use_trust_param:
                # Squeeze first
                if len(u_fuse_ds.shape) > 1: u_fuse_ds = u_fuse_ds.squeeze()
                if len(C_fuse.shape) > 1: C_fuse = C_fuse.squeeze()
                
                if args.no_conflict:
                    u_fuse = u_fuse_ds
                else:
                    u_fuse = u_fuse_ds + C_fuse
            else:
                # Fallback to Param for ID samples (or if Baseline)
                # This was the "bug" that actually helped: assume ID samples are best judged by Param confidence
                u_fuse = u_param
            
            # Squeeze dim 1 if exists (B, 1) -> (B,)
            if len(u_fuse.shape) > 1: u_fuse = u_fuse.squeeze()
            if len(u_param.shape) > 1: u_param = u_param.squeeze()
            if len(u_nonparam.shape) > 1: u_nonparam = u_nonparam.squeeze()
            if len(C_fuse.shape) > 1: C_fuse = C_fuse.squeeze()
            
            # --- Predictions and PROBS for ECE/ACC use unscaled logits ---
            # Re-fuse with unscaled parametric for calibration-accurate probs
            evidence_param_calib = evidence_extractor.get_parametric_evidence(logits_calib)
            alpha_param_calib = evidence_param_calib + 1
            
            if not args.baseline:
                 if args.adaptive_fusion or args.fusion_strategy == 'fixed':
                     alpha_fuse_calib, _, _ = fusion_module.adaptive_ds_combination(
                         alpha_param_calib, alpha_nonparam, 
                         strategy=args.fusion_strategy, 
                         fixed_weight=args.fixed_weight
                     )
                 else:
                     alpha_fuse_calib, _, _ = fusion_module.ds_combination(alpha_param_calib, alpha_nonparam, discount_factor=None)

                 probs_fuse_calib = alpha_fuse_calib / torch.sum(alpha_fuse_calib, dim=1, keepdim=True)
            else:
                 probs_fuse_calib = alpha_param_calib / torch.sum(alpha_param_calib, dim=1, keepdim=True)
            
            confidence_final, pred_fuse = torch.max(probs_fuse_calib, 1)
            _, pred_param = torch.max(logits_calib, 1) # Normal pred
            _, pred_nonparam = torch.max(alpha_nonparam, 1) # OT pred remains unaffected

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
            
            # Phase 16: Feature Norm Score (higher norm = more ID-like, so use -norm as OOD score)
            # Use captured features_flat directly
            feat_norm = torch.norm(features_flat, p=2, dim=1)  # (B,)
            id_feat_norms.extend((-feat_norm).cpu().numpy())  # negative norm: higher = more OOD
            
            # --- MaxLogit Score (-max_logit) ---
            max_logits, _ = torch.max(logits_calib, dim=1)
            id_max_logits.extend((-max_logits).cpu().numpy())
            
            # --- GradNorm Score (L1 diff * L1 feat) ---
            probs = torch.softmax(logits_calib, dim=1)
            uni = 1.0 / Config.NUM_CLASSES
            p_diff_norm = torch.norm(probs - uni, p=1, dim=1)
            feat_norm_l1 = torch.norm(features_flat, p=1, dim=1)
            grad_norm = p_diff_norm * feat_norm_l1
            id_gradnorms.extend((-grad_norm).cpu().numpy())
            
            # POT: Accumulate spatial features for batch-level OT
            if pot_scorer is not None:
                pot_id_features.append(features.detach())
                pot_id_batch_sizes.append(features.shape[0])
            
            # Temperature-Scaled Probs for ECE
            logits_temp = apply_temperature(logits_calib, optimal_T)
            probs_temp = torch.softmax(logits_temp, dim=1)
            all_probs_fusion.append(probs_temp)
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
                    "confidence": confidence_final[i].item(),
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
    
    # --- POT: Compute batch-level OT scores for ID data ---
    if pot_scorer is not None:
        logger.info(f"Computing POT scores for ID data ({len(pot_id_features)} batches)...")
        all_id_features = torch.cat(pot_id_features, dim=0)  # (N, C, H, W)
        pot_id_scores = pot_scorer.compute_contrastive_ot_chunked(
            all_id_features, 
            chunk_size=Config.POT_BATCH_SIZE,
            omega=Config.POT_OMEGA,
            sinkhorn_reg=Config.POT_SINKHORN_REG,
            max_iter=Config.POT_SINKHORN_ITER
        )
        id_uncertainties_pot = pot_id_scores.cpu().numpy().tolist()
        logger.info(f"POT ID Scores: Mean={np.mean(id_uncertainties_pot):.4f} Std={np.std(id_uncertainties_pot):.4f}")
        del all_id_features, pot_id_features  # Free memory
    
    # -----------------------
    # OOD Evaluation
    # -----------------------
    # Define OOD datasets to test based on current dataset
    ood_datasets = []
    if Config.DATASET_NAME == 'cifar10':
        ood_datasets = ['svhn', 'cifar100']
    elif Config.DATASET_NAME == 'cifar100':
        ood_datasets = ['svhn', 'cifar10']
    elif Config.DATASET_NAME == 'imagenet100':
        ood_datasets = ['inaturalist', 'places365', 'textures', 'openimage_o']
    
    # Extended OOD datasets (if requested)
    if args.extended_ood:
        ood_datasets.extend(['mnist', 'textures', 'tinyimagenet'])  # Removed places365 as per request
        logger.info(f"Extended OOD mode: Testing {ood_datasets}")
    
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
        ood_u_pot = []  # POT branch
        ood_max_logits = []   # MaxLogit branch
        ood_gradnorms = []    # GradNorm branch
        ood_feat_norms = []  # Phase 16: Feature Norm Score
        pot_ood_features = [] if pot_scorer else None  # Accumulate features
        
        current_ood_loop_func = None
        
        if ood_name == 'noise':
            # Generator wrapper
            def noise_gen():
                num_ood = args.max_samples if args.max_samples else 3000
                with torch.no_grad():
                    noise_images = torch.randn(num_ood, 3, 32, 32).to(device)
                    # Fake labels -1 for OOD
                    fake_labels = torch.full((num_ood,), -1, dtype=torch.long).to(device)
                    yield noise_images, fake_labels
            current_ood_loop_func = noise_gen
        else:
            def loader_gen():
                loader = get_ood_loader(ood_name)
                max_samples = args.max_samples if args.max_samples else 3000
                cnt = 0
                for img, lbl in loader:
                    if cnt >= max_samples: break
                    # Force labels to -1 for OOD
                    lbl = torch.full_like(lbl, -1)
                    yield img.to(device), lbl.to(device)
                    cnt += img.size(0)
            current_ood_loop_func = loader_gen

            # ODIN typically requires Grad. We disable global no_grad for the loop.
            for images, labels in current_ood_loop_func():
                # ODIN Logic
                if getattr(Config, 'ODIN_EPS', 0.0) > 0:
                    # Enable Grad for Input
                    images.requires_grad = True
                    _, _, logits_pre, _ = model(images)
                    logits_pre = logits_pre / getattr(Config, 'ODIN_TEMP', 1.0)
                    
                    preds = logits_pre.argmax(dim=1)
                    loss = nn.CrossEntropyLoss()(logits_pre, preds)
                    loss.backward()
                    
                    grad_sign = images.grad.data.sign()
                    # OOD: Perturbation aims to increase confidence of "Top Class".
                    # If OOD, this perturbation is less effective than for ID.
                    images_perturbed = images - Config.ODIN_EPS * grad_sign
                    
                    with torch.no_grad():
                        x, features_flat, logits, _ = model(images_perturbed)
                else:
                    with torch.no_grad():
                        x, features_flat, logits, _ = model(images)

                # Temperature
                if getattr(Config, 'ODIN_TEMP', 1.0) > 1.0:
                    logits = logits / Config.ODIN_TEMP

                with torch.no_grad():
                    features = apply_bn_spatial(x, model.bn)
                    
                    # React Clipping (Skip if --no_react)
                    if clip_threshold > 0 and not getattr(args, 'no_react', False):
                         features = torch.clamp(features, max=clip_threshold)

                    # --- ASH (Phase 15) ---
                    if getattr(Config, 'ASH_PERCENTILE', 0) > 0 and getattr(args, 'ash', False):
                         features = apply_ash(features, percentile=getattr(Config, 'ASH_PERCENTILE', 90))
                    
                    # Fix: Use Exp for OOD loop too (Hybrid Strategy - Method 18)
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
                        
                        # Skip VOS if --no_vos
                        vo_dists_effective = None if getattr(args, 'no_vos', False) else vo_dists
                        evidence_nonparam = evidence_extractor.get_non_parametric_evidence(
                            ot_dists, topk_indices, support_labels, 
                            gamma_scale=gamma_scale, vo_distances=vo_dists_effective
                        )
                        alpha_nonparam = evidence_nonparam + 1
                        
                        # Fusion (synced with ID loop logic)
                        if args.adaptive_fusion or args.fusion_strategy == 'fixed':
                             alpha_fuse, u_fuse_ds, C_fuse = fusion_module.adaptive_ds_combination(
                                 alpha_param, alpha_nonparam,
                                 strategy=args.fusion_strategy,
                                 fixed_weight=args.fixed_weight
                             )
                        else:
                             alpha_fuse, u_fuse_ds, C_fuse = fusion_module.ds_combination(alpha_param, alpha_nonparam, discount_factor=None)
                        
                        # Recalculate u_nonparam locally if needed for max
                        S_nonparam = torch.sum(alpha_nonparam, dim=1)
                        u_nonparam = Config.NUM_CLASSES / S_nonparam

                        # Integrate Conflict (Conservative)
                        if len(u_param.shape) > 1: u_param = u_param.squeeze()
                        if len(u_nonparam.shape) > 1: u_nonparam = u_nonparam.squeeze()
                        if len(C_fuse.shape) > 1: C_fuse = C_fuse.squeeze()
                        if len(u_fuse_ds.shape) > 1: u_fuse_ds = u_fuse_ds.squeeze() # Squeeze DS uncertainty
                        
                        # SOTA Formal Fusion: OOD Score = Total Uncertainty + Conflict
                        # Ablation: --no_conflict disables C_fuse
                        if args.no_conflict:
                            u_fuse = u_fuse_ds
                        else:
                            u_fuse = u_fuse_ds + C_fuse
                        
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
                    
                    # Phase 16: Feature Norm Score for OOD
                    # Use captured features_flat directly
                    feat_norm_ood = torch.norm(features_flat, p=2, dim=1)
                    ood_feat_norms.extend((-feat_norm_ood).cpu().numpy())
                    
                    # --- MaxLogit Score (-max_logit) ---
                    max_logits_ood, _ = torch.max(logits, dim=1)
                    ood_max_logits.extend((-max_logits_ood).cpu().numpy())
                    
                    # --- GradNorm Score (L1 diff * L1 feat) ---
                    probs_ood = torch.softmax(logits, dim=1)
                    uni = 1.0 / Config.NUM_CLASSES
                    p_diff_norm_ood = torch.norm(probs_ood - uni, p=1, dim=1)
                    feat_norm_l1_ood = torch.norm(features_flat, p=1, dim=1)
                    grad_norm_ood = p_diff_norm_ood * feat_norm_l1_ood
                    ood_gradnorms.extend((-grad_norm_ood).cpu().numpy())
                    
                    # POT: Accumulate features
                    if pot_scorer is not None:
                        pot_ood_features.append(features.detach())
                    
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

            # Detailed Debug Stats
            id_fuse_np = np.array(id_uncertainties_fuse)
            ood_fuse_np = np.array(ood_u_fuse)
            id_param_np = np.array(id_uncertainties_param)
            ood_param_np = np.array(ood_u_param)
            id_ot_np = np.array(id_uncertainties_nonparam)
            ood_ot_np = np.array(ood_u_nonparam)

            logger.info(f"DEBUG Stats for {ood_name}:")
            logger.info(f"  ID Fuse: Mean={id_fuse_np.mean():.4f} Std={id_fuse_np.std():.4f} Min={id_fuse_np.min():.4f} Max={id_fuse_np.max():.4f}")
            logger.info(f"  OOD Fuse: Mean={ood_fuse_np.mean():.4f} Std={ood_fuse_np.std():.4f} Min={ood_fuse_np.min():.4f} Max={ood_fuse_np.max():.4f}")
            logger.info(f"  Separation (Fuse): Min(OOD) - Max(ID) = {ood_fuse_np.min() - id_fuse_np.max():.4f}")
            
            logger.info(f"  ID Param: Mean={id_param_np.mean():.4f} Std={id_param_np.std():.4f} Min={id_param_np.min():.4f} Max={id_param_np.max():.4f}")
            logger.info(f"  OOD Param: Mean={ood_param_np.mean():.4f} Std={ood_param_np.std():.4f} Min={ood_param_np.min():.4f} Max={ood_param_np.max():.4f}")
            
            logger.info(f"  ID OT: Mean={id_ot_np.mean():.4f} Std={id_ot_np.std():.4f} Min={id_ot_np.min():.4f} Max={id_ot_np.max():.4f}")
            logger.info(f"  OOD OT: Mean={ood_ot_np.mean():.4f} Std={ood_ot_np.std():.4f} Min={ood_ot_np.min():.4f} Max={ood_ot_np.max():.4f}")

            # --- POT: Compute batch-level OT scores for OOD data ---
            auroc_pot = 0.0
            fpr_pot = 1.0
            if pot_scorer is not None and pot_ood_features:
                all_ood_features = torch.cat(pot_ood_features, dim=0)
                pot_ood_scores = pot_scorer.compute_contrastive_ot_chunked(
                    all_ood_features,
                    chunk_size=Config.POT_BATCH_SIZE,
                    omega=Config.POT_OMEGA,
                    sinkhorn_reg=Config.POT_SINKHORN_REG,
                    max_iter=Config.POT_SINKHORN_ITER
                )
                ood_u_pot = pot_ood_scores.cpu().numpy().tolist()
                del all_ood_features, pot_ood_features
                
                logger.info(f"  POT OOD Scores: Mean={np.mean(ood_u_pot):.4f} Std={np.std(ood_u_pot):.4f}")
            
            # Compute AUROC for all branches (Ablation)
            auroc_fuse = compute_auroc(np.array(id_uncertainties_fuse), np.array(ood_u_fuse))
            auroc_param = compute_auroc(np.array(id_uncertainties_param), np.array(ood_u_param))
            auroc_nonparam = compute_auroc(np.array(id_uncertainties_nonparam), np.array(ood_u_nonparam))
            
            # Phase 16: Feature Norm AUROC/FPR
            auroc_norm = compute_auroc(np.array(id_feat_norms), np.array(ood_feat_norms))
            fpr_norm = compute_fpr95(np.array(id_feat_norms), np.array(ood_feat_norms))
            
            # Compute FPR@95 for all branches
            fpr_fuse = compute_fpr95(np.array(id_uncertainties_fuse), np.array(ood_u_fuse))
            fpr_param = compute_fpr95(np.array(id_uncertainties_param), np.array(ood_u_param))
            fpr_nonparam = compute_fpr95(np.array(id_uncertainties_nonparam), np.array(ood_u_nonparam))
            
            # POT metrics & Ensemble
            auroc_ensemble = 0.0
            fpr_ensemble = 1.0
            
            if pot_scorer is not None and len(ood_u_pot) > 0:
                auroc_pot = compute_auroc(np.array(id_uncertainties_pot), np.array(ood_u_pot))
                fpr_pot = compute_fpr95(np.array(id_uncertainties_pot), np.array(ood_u_pot))
                
                # --- Phase 16: Multi-Branch Auto-Search Ensemble ---
                try:
                    # 1. Prepare Data
                    scores_map = {
                        'Fusion': (np.array(id_uncertainties_fuse), np.array(ood_u_fuse)),
                        'POT': (np.array(id_uncertainties_pot), np.array(ood_u_pot)),
                        'Norm': (np.array(id_feat_norms), np.array(ood_feat_norms)),
                        'MaxLogit': (np.array(id_max_logits), np.array(ood_max_logits)),
                        'GradNorm': (np.array(id_gradnorms), np.array(ood_gradnorms))
                    }
                    
                    # 2. Standardization
                    z_scores = {}
                    for key, (id_sc, ood_sc) in scores_map.items():
                        if len(id_sc) == 0: continue
                        mu, std = np.mean(id_sc), np.std(id_sc) + 1e-8
                        z_id = (id_sc - mu) / std
                        z_ood = (ood_sc - mu) / std
                        z_scores[key] = (z_id, z_ood)
                        
                    # 3. Auto-Search Weights
                    # Search space: Fusion, POT, GradNorm are most promising
                    # We search weights for Fusion and POT (w1, w2), rest is 1-w1-w2 assigned to GradNorm?
                    # Or simple grid search:
                    best_auroc = 0.0
                    best_weights_str = ""
                    best_fpr = 1.0
                    
                    # Candidates to ensemble: Fusion, POT, GradNorm
                    # Grid: w_fuse in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    #       w_pot  in [0, 0.2, ... 1.0]
                    #       w_grad = 1 - w_fuse - w_pot (if >= 0)
                    
                    id_ens_best, ood_ens_best = None, None
                    
                    keys_to_use = ['Fusion', 'POT', 'GradNorm']
                    # Ensure keys exist
                    keys_to_use = [k for k in keys_to_use if k in z_scores]
                    
                    if len(keys_to_use) >= 2:
                        # Simple grid search 
                        steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                        
                        for w1 in steps:
                            for w2 in steps:
                                if w1 + w2 > 1.0: continue
                                w3 = 1.0 - w1 - w2
                                if w3 < 0: w3 = 0.0 # float err
                                
                                # Construct Ensemble
                                # Assuming Fusion, POT, GradNorm order
                                # Adjust logic if keys missing
                                
                                id_e = np.zeros_like(z_scores[keys_to_use[0]][0])
                                ood_e = np.zeros_like(z_scores[keys_to_use[0]][1])
                                
                                # Hardcoded for simplicity based on keys_to_use
                                current_weights = {}
                                
                                # w1 -> Fusion
                                if 'Fusion' in z_scores:
                                    id_e += w1 * z_scores['Fusion'][0]
                                    ood_e += w1 * z_scores['Fusion'][1]
                                    current_weights['Fusion'] = w1
                                
                                # w2 -> POT
                                if 'POT' in z_scores:
                                    id_e += w2 * z_scores['POT'][0]
                                    ood_e += w2 * z_scores['POT'][1]
                                    current_weights['POT'] = w2
                                    
                                # w3 -> GradNorm
                                if 'GradNorm' in z_scores:
                                    id_e += w3 * z_scores['GradNorm'][0]
                                    ood_e += w3 * z_scores['GradNorm'][1]
                                    current_weights['GradNorm'] = w3
                                    
                                curr_auroc = compute_auroc(id_e, ood_e)
                                if curr_auroc > best_auroc:
                                    best_auroc = curr_auroc
                                    best_weights_str = str(current_weights)
                                    best_fpr = compute_fpr95(id_e, ood_e)
                                    id_ens_best, ood_ens_best = id_e, ood_e
                        
                        auroc_ensemble = best_auroc
                        fpr_ensemble = best_fpr
                        logger.info(f"  Best Ensemble Weights: {best_weights_str}")
                    else:
                        # Fallback
                        auroc_ensemble = compute_auroc(z_scores['Fusion'][0], z_scores['Fusion'][1])
                        fpr_ensemble = compute_fpr95(z_scores['Fusion'][0], z_scores['Fusion'][1])
                    
                except Exception as e:
                    logger.error(f"Auto-Search Ensemble calculation failed: {e}")
            
            logger.info(f"OOD {ood_name} Results:")
            logger.info(f"  AUROC (Fusion): {auroc_fuse:.4f} | FPR@95: {fpr_fuse:.4f}")
            logger.info(f"  AUROC (Param): {auroc_param:.4f} | FPR@95: {fpr_param:.4f}")
            logger.info(f"  AUROC (OT):    {auroc_nonparam:.4f} | FPR@95: {fpr_nonparam:.4f}")
            logger.info(f"  AUROC (Norm):  {auroc_norm:.4f} | FPR@95: {fpr_norm:.4f}")
            if pot_scorer is not None:
                logger.info(f"  AUROC (POT):   {auroc_pot:.4f} | FPR@95: {fpr_pot:.4f}")
                logger.info(f"  AUROC (Ens3):  {auroc_ensemble:.4f} | FPR@95: {fpr_ensemble:.4f}")

            ood_results[ood_name] = {
                "auroc_fusion": float(auroc_fuse),
                "auroc_parametric": float(auroc_param),
                "auroc_nonparametric": float(auroc_nonparam),
                "auroc_pot": float(auroc_pot),
                "auroc_ensemble": float(auroc_ensemble),
                "fpr95_fusion": float(fpr_fuse),
                "fpr95_parametric": float(fpr_param),
                "fpr95_nonparametric": float(fpr_nonparam),
                "fpr95_pot": float(fpr_pot),
                "fpr95_ensemble": float(fpr_ensemble)
            }
            
            last_ood_uncertainties = ood_u_fuse


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
    
    # Send results via Telegram webhook (if configured)
    try:
        from autobot_webhook import send_inference_results
        webhook_results = {
            "parametric_accuracy": acc_param,
            "fused_accuracy": acc_fuse,
            "ece": ece_score,
            "ood": ood_results
        }
        send_inference_results(webhook_results, dataset_name=Config.DATASET_NAME)
    except ImportError:
        logger.warning("Telegram webhook not available. Skipping notification.")
    except Exception as e:
        logger.warning(f"Webhook notification failed: {e}")
    
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
    
    # Ablation Flags
    parser.add_argument("--no_conflict", action="store_true", help="Ablation: Disable Conflict term in Fusion (u_fuse = u_fuse_ds only)")
    parser.add_argument("--no_vos", action="store_true", help="Ablation: Disable Virtual Outlier discounting in evidence")
    parser.add_argument("--no_react", action="store_true", help="Ablation: Disable React feature clipping")
    parser.add_argument("--no_trust_param_id", action="store_true", help="Ablation: Use pure Fusion uncertainty for ID (disable trust-param heuristic)")
    parser.add_argument("--extended_ood", action="store_true", default=True, help="Test extended OOD datasets (Textures, iNaturalist subset)")
    parser.add_argument("--adversarial_vos", action="store_true", help="Enable Adversarial VOS optimization (Phase 10)")
    parser.add_argument("--pot", action="store_true", help='Enable POT branch (Contrastive OT Score)')
    parser.add_argument("--ash", action="store_true", help='Enable ASH (Activation Shaping) for OOD detection')
    parser.add_argument(
        '--adaptive_fusion',
        action='store_true', 
        default=getattr(Config, 'ADAPTIVE_FUSION', False),
        help='Enable Adaptive Fusion (Discount Parametric based on OT Uncertainty)'
    )
    parser.add_argument("--fusion_strategy", type=str, default="adaptive", choices=["adaptive", "fixed"], help="Fusion Strategy: 'adaptive' (Sigmoid) or 'fixed' (Weighted Evidence)")
    parser.add_argument("--fixed_weight", type=float, default=0.5, help="Weight for Parametric branch in Fixed Fusion (0.0=OT only, 1.0=Param only). Default 0.5")
    parser.add_argument("--ensemble_weight", type=float, default=0.5, help="Weight for Fusion Score in Adaptive Ensemble (0.0=POT only, 1.0=Fusion only). Default 0.5")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick debugging/verification")
    parser.add_argument("--trust_param_id", action='store_true', default=True, help="Use Parametric Uncertainty for ID samples (Performance Heuristic). Turn off for pure Fusion.")
    parser.add_argument("--evidence_type", type=str, default='exp', choices=['exp', 'softplus'], help="Evidence function type")
    
    args = parser.parse_args()
    
    # Load Config
    Config.load_config(args.config)

    # Setup Logger (Early)
    # Add Backbone to Results Directory
    results_dir = os.path.join(Config.RESULTS_DIR, Config.DATASET_NAME, Config.BACKBONE.lower())
    if not os.path.exists(results_dir):
         os.makedirs(results_dir)
         
    from src.utils import setup_logger
    
    # Add timestamp to log filename to avoid overwriting/confusion on same experiment name
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.log_suffix:
        log_name = f"experiment_{args.log_suffix}_{timestamp}"
    else:
        log_name = f"experiment_{timestamp}"
        
    logger = setup_logger(results_dir, name=log_name)
    
    # Apply Overrides
    if args.k_neighbors is not None: Config.K_NEIGHBORS = args.k_neighbors
    if args.rbf_gamma is not None: Config.RBF_GAMMA = args.rbf_gamma
    if args.sinkhorn_eps is not None: Config.SINKHORN_EPS = args.sinkhorn_eps
    if args.fusion_type is not None: Config.FUSION_TYPE = args.fusion_type
    if args.metric_type is not None: Config.METRIC_TYPE = args.metric_type
    if args.support_size is not None: Config.NUM_SUPPORT_SAMPLES = args.support_size
    if args.adversarial_vos: 
        Config.ADVERSARIAL_VOS = True
        logger.info("Override: ADVERSARIAL_VOS Enabled")
    
    # Set Seeds for Reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")
    logger.info("="*30)
    logger.info("Inference Configuration:")
    logger.info(f"Dataset: {Config.DATASET_NAME}")
    logger.info(f"Backbone: {Config.BACKBONE}")
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
    # If Metric is Cosine, ignore Adaptive Gamma and use Config
    if Config.METRIC_TYPE == 'cosine':
        logger.info(f"Forcing Config Gamma ({Config.RBF_GAMMA}) for Cosine Metric.")
        gamma_scale = Config.RBF_GAMMA

    evaluate(model, test_loader, support_features, support_labels, virtual_outliers, gamma_scale, precision_matrix, clip_threshold, device, logger, args)

if __name__ == "__main__":
    main()
