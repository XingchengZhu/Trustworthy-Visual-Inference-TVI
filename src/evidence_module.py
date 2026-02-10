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
        # Using Exp to ensure positivity and match Softmax sharpness
        # This is critical for large K (e.g. 100) to overcome the S=K Dirichlet prior
        evidence = torch.exp(logits)
        return evidence

    def get_non_parametric_evidence(self, ot_distances, topk_indices, support_labels, gamma_scale=None, vo_distances=None):
        """
        Convert OT distances to evidence.
        ot_distances: (B, K)
        topk_indices: (B, K)
        support_labels: (S,) - tensor
        gamma_scale: float, optional adaptive gamma (usually 1/mean_dist)
        vo_distances: (B, K_vo), optional distance to virtual outliers
        """
        B, K = ot_distances.shape
        evidence = torch.zeros((B, self.num_classes)).to(support_labels.device)
        
        # RBF Kernel conversion: s = exp(-gamma * d)
        # If gamma_scale provided (Adaptive), use it. Else Config default.
        gamma = gamma_scale if gamma_scale is not None else Config.RBF_GAMMA
        
        # Convert distances to similarity/affinity
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
                
        # --- Virtual Outlier Logic (Refusal Reference Frame) ---
        if vo_distances is not None:
             # Calculate similarity to Virtual Outliers
             if isinstance(vo_distances, np.ndarray):
                 vo_distances = torch.from_numpy(vo_distances).float().to(support_labels.device)
                 
             # Use same Gamma for VOs? Yes, they are in same feature space.
             sim_vo = torch.exp(-gamma * vo_distances) # (B, K_vo)
             
             # If a sample is very close to a Virtual Outlier, trust in ID evidence should drop.
             # Measure: Max similarity to any VO.
             max_vo_sim, _ = torch.max(sim_vo, dim=1) # (B,)
             
             # Trust factor: 1 - max_vo_sim
             # If max_vo_sim is 1.0 (on top of outlier), trust is 0.
             trust = 1.0 - max_vo_sim
             trust = torch.clamp(trust, min=0.0, max=1.0)
             
             # Apply discounting to evidence
             evidence = evidence * trust.unsqueeze(1)
             
        # Scale evidence
        evidence = evidence * Config.EVIDENCE_SCALE
        
        return evidence
