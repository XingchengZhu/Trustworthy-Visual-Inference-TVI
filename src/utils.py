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
    probs: (N, C) predicted probabilities
    labels: (N,) true labels
    """
    if probs.dim() != 2:
        raise ValueError("compute_ece expects probs of shape (N, C)")
        
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

def compute_fpr95(id_scores, ood_scores):
    """
    Compute FPR at 95% TPR (Standard Academic Definition).
    
    Definition: The probability that a negative example (OOD) is misclassified as positive (ID)
    when the True Positive Rate (ID) is 95%.
    
    Args:
        id_scores: Uncertainty scores for In-Distribution data (Higher = More Uncertain/OOD).
        ood_scores: Uncertainty scores for Out-of-Distribution data.
    """
    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)
    
    # 1. Determine Threshold to ensure 95% ID TPR
    # We classify sample as ID if score <= threshold.
    # To keep 95% of ID samples, threshold must be the 95th percentile of ID scores.
    # (i.e., 95% of ID scores are below this value).
    threshold = np.percentile(id_scores, 95)
    
    # 2. Compute FPR (OOD samples misclassified as ID)
    # Misclassified if ood_score <= threshold
    num_ood = len(ood_scores)
    num_fp = np.sum(ood_scores <= threshold)
    
    fpr = num_fp / (num_ood + 1e-10)
    
    return fpr
