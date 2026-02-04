import logging
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def setup_logger(save_dir, name="experiment"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(save_dir, f"{name}.log"))
        fh.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(sh)
        
    return logger

def save_results(results, save_dir, filename="metrics.json"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, filename), 'w') as f:
        json.dump(results, f, indent=4)

def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error
    probs: (N, C) or (N,) max probabilities
    labels: (N,)
    """
    if probs.dim() == 2:
        confidences, predictions = torch.max(probs, 1)
    else:
        confidences = probs
        predictions = torch.argmax(probs, dim=1) # This line assumes probs was 2D. 
        # If probs is 1D (max conf), we need predictions separately.
        # Let's assume input is (N, C) Probs
        
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

def compute_auroc(id_uncertainties, ood_uncertainties):
    """
    Compute AUROC for OOD Detection.
    Label 0: In-Distribution
    Label 1: Out-of-Distribution
    Discriminator: Uncertainty Score (Higher -> OOD)
    """
    y_true = np.concatenate([np.zeros(len(id_uncertainties)), np.ones(len(ood_uncertainties))])
    y_scores = np.concatenate([id_uncertainties, ood_uncertainties])
    
    return roc_auc_score(y_true, y_scores)

def compute_fpr95(id_uncertainties, ood_uncertainties):
    """
    Compute FPR at 95% TPR.
    TPR = TP / (TP + FN) = 0.95
    FPR = FP / (FP + TN)
    
    Here 'Positive' (Class 1) is OOD. 'Negative' (Class 0) is ID.
    We want to detect OOD (High Uncertainty).
    
    Threshold such that 95% of OOD samples are detected.
    Check how many ID samples are incorrectly detected as OOD (False Positives).
    """
    # Ensure numpy
    id_uncertainties = np.array(id_uncertainties)
    ood_uncertainties = np.array(ood_uncertainties)
    
    # Concatenate
    scores = np.concatenate([id_uncertainties, ood_uncertainties])
    labels = np.concatenate([np.zeros(len(id_uncertainties)), np.ones(len(ood_uncertainties))])
    
    # Sort scores
    # We assume higher score = OOD
    # Find threshold where TPR >= 0.95
    
    # Use sklearn simply? 
    # Or manual:
    # Sort OOD scores descending
    ood_sorted = np.sort(ood_uncertainties)
    # Threshold is the value at 5% percentile (since we want Top 95% to be above threshold)
    # percentile 5 means 5% are below, 95% are above.
    threshold = np.percentile(ood_sorted, 5)
    
    # FP: ID samples > threshold
    fp = np.sum(id_uncertainties > threshold)
    tn = np.sum(id_uncertainties <= threshold)
    
    fpr = fp / (fp + tn + 1e-8)
    return fpr
