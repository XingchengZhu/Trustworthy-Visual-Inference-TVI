"""
Temperature Scaling: Post-hoc calibration for ECE improvement.

Usage:
    from src.temperature_scaling import find_optimal_temperature, apply_temperature

    # Find optimal T on validation set
    T = find_optimal_temperature(model, val_loader, device)

    # Apply at inference
    calibrated_logits = apply_temperature(logits, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
import numpy as np


class TemperatureScaler(nn.Module):
    """
    Learnable temperature parameter for post-hoc calibration.
    Optimizes T to minimize NLL on a validation set.
    """
    def __init__(self):
        super().__init__()
        # Initialize T = 1.0 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, logits):
        return logits / self.temperature


def find_optimal_temperature(model, val_loader, device, max_iter=50):
    """
    Find the optimal temperature T that minimizes NLL on the validation set.
    
    Args:
        model: trained model
        val_loader: validation data loader
        device: torch device
        max_iter: max LBFGS iterations
        
    Returns:
        optimal_temperature: float
    """
    model.eval()
    
    # Collect all logits and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            _, _, logits, _ = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)  # (N, C)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)
    
    # Optimize Temperature
    scaler = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)
    
    def eval_closure():
        optimizer.zero_grad()
        scaled_logits = scaler(all_logits)
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_closure)
    
    optimal_T = scaler.temperature.item()
    
    return optimal_T


def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits."""
    return logits / temperature
