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
        
        if Config.METRIC_TYPE == 'sinkhorn':
            # 5. Solve Sinkhorn with custom PyTorch implementation for Batch support
            # ot.sinkhorn2 struggles with 3D inputs (Batched M)
            dist_flat = self.batch_sinkhorn_torch(M, Config.SINKHORN_EPS, Config.SINKHORN_MAX_ITER)
            distances = dist_flat.view(B, k)
            
        elif Config.METRIC_TYPE == 'euclidean':
            # Variant A1: Euclidean Distance as simple k-NN
            # dist(Q, S) = ||Q - S||_2
            # Recompute exact Euclidean distance on full features (not GAP) using efficient matrix op
            # q_flat: (B, N, C), s_flat: (S, N, C)
            # This is too heavy for all-to-all. Use the Coarse Filtering results first!
            
            # We already have topk_indices for each query based on GAP cosine.
            # Let's refine the distance for these top K neighbors using full features.
            # Note: This is an approximation. True k-NN Euclidean requires exhaustive search.
            # But for "Ablation", using same candidates but different fine-grained metric is fair.
            # Challenge: Euclidean requires pixel-to-pixel alignment or Global Pooling.
            # Strategy: Global Average Pooling Euclidean Distance.
            # q_avg: (B, C), s_avg: (S, C) (Already computed in Coarse Step)
            
            # Let's recompute full euclidean on GAP for the top K neighbors to be consistent?
            # Or just use the global euclidean distance.
            
            B_idx = torch.arange(B).unsqueeze(1).expand(-1, k)
            s_avg_k = s_avg[topk_indices] # (B, K, C)
            q_avg_k = q_avg.unsqueeze(1).expand(-1, k, -1) # (B, K, C)
            
            # Euclidean distance between Global Features
            distances = torch.norm(q_avg_k - s_avg_k, dim=2, p=2) # (B, K)
            
        elif Config.METRIC_TYPE == 'cosine':
            # Variant A2: Cosine Distance
            # Distance = 1 - CosineSimilarity
            # We already computed sim_coarse (B, S).
            # Just gather top K values.
            
            sim_values = torch.gather(sim_coarse, 1, topk_indices) # (B, K)
            distances = 1.0 - sim_values
            # Clamp to >= 0
            distances = torch.clamp(distances, min=0.0)
            
        else:
             raise ValueError(f"Unknown Metric Type: {Config.METRIC_TYPE}")

        return distances, topk_indices

    def batch_sinkhorn_torch(self, M, reg, numItermax):
        """
        Log-domain PyTorch implementation of Sinkhorn algorithm for batched cost matrices.
        More stable for small reg or large costs.
        M: (Batch, N, N)
        Returns: (Batch,) distances
        """
        B, N, _ = M.shape
        
        # Log mu, Log nu (Uniform marginals)
        # log(1/N) = -log(N)
        log_mu = -torch.log(torch.tensor(float(N), device=M.device)).repeat(B, N)
        log_nu = -torch.log(torch.tensor(float(N), device=M.device)).repeat(B, N)
        
        # Init potentials f, g (dual variables)
        # f = zeros, g = zeros
        f = torch.zeros((B, N), device=M.device)
        g = torch.zeros((B, N), device=M.device)
        
        # Gibbs Kernel in log domain: K_log = -M / reg
        K_log = -M / reg
        
        for _ in range(numItermax):
            # Update f: f = reg * (log_mu - logsumexp((g + K_log)/reg)) ?
            # Standard Sinkhorn in log domain:
            # f_i = -reg * logsumexp( (g_j - M_ij/reg) ) + reg*log_mu_i  ... NO
            
            # Derived from: u = a / (K @ v).  f = reg * log(u). g = reg * log(v).
            # u = exp(f/reg), v = exp(g/reg)
            # exp(f/reg) = exp(log_mu) / ( exp(-M/reg) @ exp(g/reg) )
            # f/reg = log_mu - log( sum( exp(-M/reg + g/reg) ) )
            # f = reg * log_mu - reg * logsumexp( (-M + g_broad) / reg )
            
            # M is (B, N, N). g is (B, N).
            # g_broad (B, 1, N) for broadcasting over i
            # (-M + g) is (B, N, N)
            
            g_unsqueezed = g.unsqueeze(1) # (B, 1, N)
            
            # LogSumExp along dim=2 (j)
            lse_g = torch.logsumexp( (-M + g_unsqueezed) / reg, dim=2) # (B, N)
            
            f = reg * log_mu - reg * lse_g
            
            # Update g similarly
            # g = reg * log_nu - reg * logsumexp( (-M.T + f)/reg )
            
            f_unsqueezed = f.unsqueeze(2) # (B, N, 1) to broadcast over j?
            # Or transpose M.
            # (-M + f_broad) where f is for i.
            # We want sum over i for each j.
            # (-M_{ij} + f_i)
            
            # M is (B, N, N)
            # f_unsqueezed: (B, N, 1)
            # (-M + f_unsqueezed) -> (B, N, N)
            # lse along dim=1 (i) -> (B, N)
            
            lse_f = torch.logsumexp( (-M + f_unsqueezed) / reg, dim=1) # (B, N)
            g = reg * log_nu - reg * lse_f
            
        # Compute OT distance
        # dist = sum(P * M)
        # P = exp( (f + g - M) / reg ) roughly?
        # Actually P_{ij} = u_i * K_{ij} * v_j = exp(f_i/reg) * exp(-M_{ij}/reg) * exp(g_j/reg)
        # P_{ij} = exp( (f_i + g_j - M_{ij}) / reg )
        
        f_us = f.unsqueeze(2) # (B, N, 1)
        g_us = g.unsqueeze(1) # (B, 1, N)
        
        log_P = (f_us + g_us - M) / reg
        P = torch.exp(log_P)
        
        dist = torch.sum(P * M, dim=[1, 2])
        
        return dist
