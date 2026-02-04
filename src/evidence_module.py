import torch
import torch.nn.functional as F
import numpy as np
from src.config import Config

class EvidenceExtractor:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def get_parametric_evidence(self, logits):
        """
        Convert logits to evidence.
        Common approaches: Relu(logits), Softplus(logits), Exp(logits).
        Plan says: alpha = e_k + 1. 
        """
        # Using Softplus to ensure positivity
        evidence = F.softplus(logits)
        return evidence

    def get_non_parametric_evidence(self, ot_distances, topk_indices, support_labels):
        """
        Convert OT distances to evidence.
        ot_distances: (B, K)
        topk_indices: (B, K)
        support_labels: (S,) - tensor
        """
        B, K = ot_distances.shape
        evidence = torch.zeros((B, self.num_classes)).to(support_labels.device)
        
        # RBF Kernel conversion: s = exp(-gamma * d)
        # Gamma parameter tuning is important.
        gamma = 1.0 
        
        # Convert distances to similarity/affinity
        # ot_distances is numpy array from previous step usually, let's ensure tensor
        if isinstance(ot_distances, np.ndarray):
            ot_distances = torch.from_numpy(ot_distances).float().to(support_labels.device)
            
        similarity = torch.exp(-gamma * ot_distances) # (B, K)
        
        # Aggregate evidence per class
        for i in range(B):
            for j in range(K):
                s_idx = topk_indices[i, j]
                label = support_labels[s_idx]
                score = similarity[i, j]
                evidence[i, label] += score
                
        # Scale evidence
        evidence = evidence * Config.EVIDENCE_SCALE
        
        return evidence
