import torch
import ot
import numpy as np
from src.config import Config

class OTMetric:
    def __init__(self, device='cpu'):
        self.device = device
        
    def compute_batch_ot(self, query_features, support_features, support_labels):
        """
        Computes OT distances in batch mode on GPU.
        
        query_features: (B, C, H, W)
        support_features: (S, C, H, W)
        support_labels: (S,)
        
        Returns: distances (B, K), topk_indices (B, K)
        """
        B, C, H, W = query_features.shape
        S = support_features.shape[0]
        N = H * W
        
        # 1. Prepare Features
        # Flatten spatial dims: (B, C, H, W) -> (B, C, N) -> (B, N, C) 
        # (Pixel as sample, Channel as feature)
        # Verify shape assumption: Cost is between pixels (N x N). Features are C-dim vectors.
        q_flat = query_features.view(B, C, N).permute(0, 2, 1) # (B, N, C)
        s_flat = support_features.view(S, C, N).permute(0, 2, 1) # (S, N, C)
        
        # Normalize for Cosine Distance
        q_norm = torch.nn.functional.normalize(q_flat, dim=2)
        s_norm = torch.nn.functional.normalize(s_flat, dim=2)
        
        # 2. Coarse Filtering (Euclidean/Cosine on Global Avg Pool)
        # GAP: (B, C, H, W) -> (B, C)
        q_avg = torch.mean(query_features, dim=[2, 3]) 
        s_avg = torch.mean(support_features, dim=[2, 3])
        
        q_avg = torch.nn.functional.normalize(q_avg, dim=1) 
        s_avg = torch.nn.functional.normalize(s_avg, dim=1)
        
        # Sim Matrix: (B, S)
        sim_coarse = torch.mm(q_avg, s_avg.T)
        
        # Top K
        k = min(Config.K_NEIGHBORS, S)
        _, topk_indices = torch.topk(sim_coarse, k, dim=1) # (B, K)
        
        # 3. Batch Construction for OT
        # We need to construct B*K pairs.
        
        # Gather Query: repeat each query K times -> (B, K, N, C)
        q_batch = q_norm.unsqueeze(1).expand(-1, k, -1, -1) 
        # Result: (B, K, N, C)
        
        # Gather Support: Select topk neighbors -> (B, K, N, C)
        # s_norm is (S, N, C). indices is (B, K).
        # We need to index dim 0 of s_norm with indices.
        # Flatten indices first option or use fancy indexing
        
        # Fancy indexing in pytorch:
        # s_norm[indices] will give (B, K, N, C)
        s_batch = s_norm[topk_indices] 
        
        # Flatten to (B*K, N, C) for batched OT
        curr_batch_size = B * k
        q_in = q_batch.reshape(curr_batch_size, N, C)
        s_in = s_batch.reshape(curr_batch_size, N, C)
        
        # 4. Compute Cost Matrix (Cosine Distance)
        # M = 1 - Q . S^T
        # Bmm: (B*K, N, C) x (B*K, C, N) -> (B*K, N, N)
        scores = torch.bmm(q_in, s_in.transpose(1, 2))
        M = 1 - scores
        
        # Ensure M is positive (numerical stability)
        M = torch.clamp(M, min=0.0)
        
        # 5. Solve Sinkhorn with custom PyTorch implementation for Batch support
        # ot.sinkhorn2 struggles with 3D inputs (Batched M)
        dist_flat = self.batch_sinkhorn_torch(M, Config.SINKHORN_EPS, Config.SINKHORN_MAX_ITER)
        
        # Output is (B*K,)
        distances = dist_flat.view(B, k)
        
        return distances, topk_indices

    def batch_sinkhorn_torch(self, M, reg, numItermax):
        """
        PyTorch implementation of Sinkhorn algorithm for batched cost matrices.
        M: (Batch, N, N)
        Returns: (Batch,) distances
        """
        B, N, _ = M.shape
        
        # Uniform marginals
        a = torch.ones((B, N), device=M.device) / N
        b = torch.ones((B, N), device=M.device) / N
        
        # K = exp(-M / reg)
        K = torch.exp(-M / reg)
        
        # Init u, v
        u = torch.ones((B, N), device=M.device) / N
        
        # Sinkhorn iterations
        for _ in range(numItermax):
            # v = b / (K^T @ u)
            # K is (B, N, N). transpose(1, 2) -> (B, N, N)
            # u is (B, N). unsqueeze(-1) -> (B, N, 1)
            # bmm result: (B, N, 1). squeeze -> (B, N)
            
            # K.transpose(1, 2) @ u.unsqueeze(2) -> (B, N, 1)
            Kv = torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2)
            v = b / (Kv + 1e-8)
            
            # u = a / (K @ v)
            Ku = torch.bmm(K, v.unsqueeze(2)).squeeze(2)
            u = a / (Ku + 1e-8)
            
        # Compute distance
        # dist = sum(u * (K * M) * v)
        # K * M -> elementwise
        # u.unsqueeze(2) * (K*M) * v.unsqueeze(1) -> (B, N, N)
        
        # Transport plan P = u.dimshuffle(0,1,x) * K * v.dimshuffle(0,x,1)
        # P = diag(u) K diag(v)
        P = u.unsqueeze(2) * K * v.unsqueeze(1)
        
        dist = torch.sum(P * M, dim=[1, 2])
        
        return dist
