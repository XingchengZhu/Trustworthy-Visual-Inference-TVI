"""
POT Module: Prototype-based Optimal Transport for OOD Detection.

Implements Batch-Level Sinkhorn OT with Contrastive Transport Cost,
inspired by "Prototype-based Optimal Transport for OOD Detection".

Key differences from per-sample OT (ot_module.py):
1. Operates on entire test batch vs class prototypes (not per-sample K-NN)
2. Uses Mass Conservation constraint (Sinkhorn)
3. Computes Contrastive Transport Cost: T_proto - T_vo
4. Generates dynamic Virtual Outliers at test time based on batch statistics
"""

import torch
import torch.nn.functional as F
import numpy as np


class POTScorer:
    """
    Prototype-based Optimal Transport scorer for OOD detection.
    
    Usage:
        scorer = POTScorer(prototypes_gap, device)
        scores = scorer.compute_contrastive_ot(batch_features)
        # Higher score -> more likely OOD
    """
    
    def __init__(self, prototypes_gap, device='cpu'):
        """
        Args:
            prototypes_gap: (C, D) class prototypes after GAP.
            device: torch device
        """
        self.device = device
        # Normalize prototypes to unit sphere for Cosine-like behavior
        self.prototypes = F.normalize(prototypes_gap.to(device).float(), p=2, dim=1)
        self.num_classes = prototypes_gap.shape[0]
        self.feat_dim = prototypes_gap.shape[1]
        
    def _sinkhorn(self, cost_matrix, reg=0.1, max_iter=50):
        """
        Log-domain Sinkhorn algorithm for numerical stability.
        
        Args:
            cost_matrix: (B, C) pairwise cost between batch samples and prototypes
            reg: entropic regularization coefficient (lambda)
            max_iter: number of Sinkhorn iterations
            
        Returns:
            transport_plan: (B, C) optimal transport plan gamma
        """
        B, C = cost_matrix.shape
        
        # Uniform marginals
        # mu: (B,) - each test sample has equal mass
        # nu: (C,) - each prototype has equal mass  
        log_mu = torch.full((B,), -np.log(B), device=self.device)
        log_nu = torch.full((C,), -np.log(C), device=self.device)
        
        # Log-domain kernel
        log_K = -cost_matrix / reg  # (B, C)
        
        # Initialize dual variables
        log_u = torch.zeros(B, device=self.device)
        log_v = torch.zeros(C, device=self.device)
        
        for _ in range(max_iter):
            # Update u: log_u = log_mu - logsumexp(log_K + log_v, dim=1)
            log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            # Update v: log_v = log_nu - logsumexp(log_K.T + log_u, dim=0)
            log_v = log_nu - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)
        
        # Recover transport plan
        # gamma_ij = exp(log_u_i + log_K_ij + log_v_j)
        log_gamma = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        gamma = torch.exp(log_gamma)
        
        return gamma
    
    def _compute_per_sample_cost(self, cost_matrix, transport_plan):
        """
        Extract per-sample transport cost from the transport plan.
        
        T_i = sum_j (gamma_ij * cost_ij) / sum_j(gamma_ij)
        
        Args:
            cost_matrix: (B, C)
            transport_plan: (B, C)
            
        Returns:
            per_sample_cost: (B,)
        """
        # Element-wise product and sum over prototypes
        weighted_cost = transport_plan * cost_matrix  # (B, C)
        per_sample_cost = weighted_cost.sum(dim=1)  # (B,)
        
        # Normalize by row mass (should be ~1/B for uniform marginals, but normalize for safety)
        row_mass = transport_plan.sum(dim=1).clamp(min=1e-10)
        per_sample_cost = per_sample_cost / row_mass
        
        return per_sample_cost

    def _generate_dynamic_vo(self, batch_features, omega=2.0):
        """
        Generate test-time dynamic Virtual Outliers.
        
        VO_i = prototype_i + omega * (batch_mean - prototype_i)
        
        When omega > 1, VOs are pushed beyond the batch mean,
        into the hypothetical OOD region.
        
        Args:
            batch_features: (B, D) GAP features of test batch
            omega: extrapolation coefficient (>1 to push beyond batch mean)
            
        Returns:
            virtual_outliers: (C, D) dynamic virtual outliers
        """
        batch_mean = batch_features.mean(dim=0)  # (D,)
        
        # VO_i = proto_i + omega * (M - proto_i) = (1-omega)*proto_i + omega*M
        virtual_outliers = self.prototypes + omega * (batch_mean.unsqueeze(0) - self.prototypes)
        
        return virtual_outliers
    
    def compute_contrastive_ot(self, batch_features_spatial, omega=2.0, sinkhorn_reg=0.1, max_iter=50):
        """
        Compute Contrastive Transport Cost for OOD detection.
        
        Score = T_proto - T_vo
        - ID samples: T_proto low (close to prototypes), T_vo high (far from VOs) -> Score negative/low
        - OOD samples: T_proto high (far from prototypes), T_vo low (close to VOs) -> Score positive/high
        
        Args:
            batch_features_spatial: (B, C_feat, H, W) spatial features from backbone
            omega: VO extrapolation coefficient
            sinkhorn_reg: Sinkhorn regularization
            max_iter: Sinkhorn iterations
            
        Returns:
            contrastive_scores: (B,) higher = more likely OOD
        """
        # Step 1: Global Average Pooling to get flat features
        if batch_features_spatial.dim() == 4:
            batch_flat = batch_features_spatial.mean(dim=(2, 3))  # (B, D)
        else:
            batch_flat = batch_features_spatial  # Already flat (B, D)
        
        B, D = batch_flat.shape
        
        # Handle edge case: batch too small for meaningful OT
        if B < 2:
            return torch.zeros(B, device=self.device)
        
        # Step 2: Normalize features for cosine-like distance
        batch_norm = F.normalize(batch_flat, p=2, dim=1)  # (B, D)
        # self.prototypes are already normalized in __init__
        
        # Step 3: Cost matrix to prototypes (Euclidean distance on normalized features)
        # Corresponds to sqrt(2 * (1 - cos_sim))
        cost_proto = torch.cdist(batch_norm, self.prototypes, p=2)  # (B, C)
        
        # Step 4: Sinkhorn OT -> transport plan -> per-sample cost to prototypes
        gamma_proto = self._sinkhorn(cost_proto, reg=sinkhorn_reg, max_iter=max_iter)
        T_proto = self._compute_per_sample_cost(cost_proto, gamma_proto)
        
        # Step 5: Generate dynamic Virtual Outliers
        vo = self._generate_dynamic_vo(batch_norm, omega=omega)  # (C, D)
        # Normalize VOs to unit sphere to maintain consistent distance metric
        vo = F.normalize(vo, p=2, dim=1)
        
        # Step 6: Cost matrix to VOs
        cost_vo = torch.cdist(batch_norm, vo, p=2)  # (B, C)
        
        # Step 7: Sinkhorn OT -> per-sample cost to VOs
        gamma_vo = self._sinkhorn(cost_vo, reg=sinkhorn_reg, max_iter=max_iter)
        T_vo = self._compute_per_sample_cost(cost_vo, gamma_vo)
        
        # Step 8: Contrastive Transport Cost
        # Higher T_proto - T_vo = more likely OOD
        contrastive_scores = T_proto - T_vo
        
        return contrastive_scores
    
    def compute_contrastive_ot_chunked(self, all_features_spatial, chunk_size=512, omega=2.0, sinkhorn_reg=0.1, max_iter=50):
        """
        Process features in chunks of chunk_size for batch-level OT.
        This simulates the POT paper's "test data arriving in batches" assumption.
        
        Args:
            all_features_spatial: (N, C_feat, H, W) all features
            chunk_size: POT batch size (paper default: 512)
            omega: VO extrapolation coefficient
            sinkhorn_reg: Sinkhorn regularization
            max_iter: Sinkhorn iterations
            
        Returns:
            all_scores: (N,) contrastive transport cost scores
        """
        N = all_features_spatial.shape[0]
        all_scores = []
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = all_features_spatial[start:end]
            
            scores = self.compute_contrastive_ot(
                chunk, omega=omega, 
                sinkhorn_reg=sinkhorn_reg, max_iter=max_iter
            )
            all_scores.append(scores)
        
        return torch.cat(all_scores, dim=0)
