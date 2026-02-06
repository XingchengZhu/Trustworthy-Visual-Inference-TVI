import torch
import ot
import numpy as np
from src.config import Config

class OTMetric:
    def __init__(self, device='cpu'):
        self.device = device
        
    def compute_batch_ot(self, query_features, support_features, support_labels, virtual_outliers=None, precision_matrix=None):
        """
        ...
        precision_matrix: (D, D) - Optional for Mahalanobis
        """
        B, C, H, W = query_features.shape
        S = support_features.shape[0]
        N = H * W
        
        # 1. Prepare Features (Flatten)
        q_flat = query_features.view(B, C, N).permute(0, 2, 1).contiguous() # (B, N, C)
        s_flat = support_features.view(S, C, N).permute(0, 2, 1).contiguous() # (S, N, C)
        
        # If Mahalanobis, project features first!
        # D_mah(x, y) = sqrt( (x-y) P (x-y).T ) = || L(x-y) || where P = L.T L
        # Let's perform Cholesky decomp on P? Or Eigen?
        # Actually P is derived from inv(Cov).
        # We can just matmul features with L where L @ L.T = P?
        # Or easier: matmul features with P_sqrt.
        # But we passed P (precision).
        
        if Config.METRIC_TYPE == 'mahalanobis' and precision_matrix is not None:
             # Decompose P = U @ S @ U.T
             # Transform T = U @ sqrt(S)
             # Then || T x ||^2 = x.T T.T T x = x.T (U sqrt(S) sqrt(S) U.T) x = x.T P x.
             # So we compute T.
             # Since P is PSD, use Eig or Cholesky.
             # Let's use svd for stability.
             # NOTE: Doing this every batch is slow? P is constant.
             # We should cache T. But precision_matrix passed is tensor.
             # Let's assume caller handles efficiency or we do it quickly (512x512 is fast).
             
             # Actually, just compute P_sqrt.
             # Or even easier: P is (D, D).
             # q_flat is (B, 1, D).
             # q_transformed = q_flat @ P_sqrt ? No.
             
             # Let's trust Euclidean on Transformed features.
             # T = torch.linalg.cholesky(precision_matrix) # Lower triangular L @ L.T = P
             # transformed_x = x @ L
             
             try:
                 L = torch.linalg.cholesky(precision_matrix)
             except:
                 # Precision matrix might not be perfectly PD?
                 # Fallback to SVD
                 u, s, v = torch.svd(precision_matrix)
                 L = torch.matmul(u, torch.diag(torch.sqrt(s)))
                 
             # Transform features
             # q_flat: (B, N, C). C=512.
             # L: (C, C).
             # q_new = q @ L
             q_flat = torch.matmul(q_flat, L)
             s_flat = torch.matmul(s_flat, L)
             
             # Proceed with Euclidean!
             # We SKIP normalization because Mahalanobis relies on magnitude variance.
             q_norm = q_flat
             s_norm = s_flat
             
        else:
            # Normalize for Cosine/Metric
            q_norm = torch.nn.functional.normalize(q_flat, dim=2)
            s_norm = torch.nn.functional.normalize(s_flat, dim=2)
        
        # Define SPP function (Used for Coarse Filtering in all modes)
        def spp_func(x):
            # x: (B, C, H, W)
            levels = [1, 2, 4]
            pooled = []
            for k in levels:
                p = torch.nn.functional.adaptive_avg_pool2d(x, (k, k))
                # Flatten: (B, C, k, k) -> (B, C, k*k)
                p = p.view(x.size(0), C, -1)
                pooled.append(p)
            return torch.cat(pooled, dim=2).view(x.size(0), -1)

        # 2. SPP (Skip for Mahalanobis/Flat features to keep it simple, or apply SPP on transformed?)
        if Config.METRIC_TYPE == 'mahalanobis':
             # If input was 1x1, SPP makes no sense or just repeats. 
             # We just use q_norm directly.
             q_spp = q_norm.view(B, -1)
             s_spp = s_norm.view(S, -1)
             k = min(Config.K_NEIGHBORS, S)
             
        else:
             q_spp = spp_func(query_features)
             s_spp = spp_func(support_features)
             q_spp = torch.nn.functional.normalize(q_spp, dim=1)
             s_spp = torch.nn.functional.normalize(s_spp, dim=1)
             k = min(Config.K_NEIGHBORS, S)
        
        # Sim Matrix
        sim_coarse = torch.mm(q_spp, s_spp.T)
        _, topk_indices = torch.topk(sim_coarse, k, dim=1)

        # ... (Rest of Euclidean logic matches)
        # Use Euclidean if Metric is Mahalanobis (since we transformed features)
        
        # Batch Construction
        q_batch = q_norm.unsqueeze(1).expand(-1, k, -1, -1) 
        s_batch = s_norm[topk_indices] 
        curr_batch_size = B * k
        q_in = q_batch.reshape(curr_batch_size, N, C)
        s_in = s_batch.reshape(curr_batch_size, N, C)
        
        scores = torch.bmm(q_in, s_in.transpose(1, 2))
        M = 1 - scores # This is Cosine distance logic, INVALID for Mahalanobis?
        
        if Config.METRIC_TYPE == 'mahalanobis':
             # Use Euclidean Distance on Transformed Features
             # q_in, s_in are (BS, 1, D)
             dists = torch.norm(q_in - s_in, dim=2, p=2) # (BS, 1) -> (BS,)
             distances = dists.view(B, k)
             
        elif Config.METRIC_TYPE == 'sinkhorn':
             # ...
             M = torch.clamp(M, min=0.0)
             dist_flat = self.batch_sinkhorn_torch(M, Config.SINKHORN_EPS, Config.SINKHORN_MAX_ITER)
             distances = dist_flat.view(B, k)
        elif Config.METRIC_TYPE == 'euclidean':
             # ... SPP based euclidean ...
             # But here we want Flat Feature Euclidean?
             # q_in, s_in are flat features (if using flat input).
             dists = torch.norm(q_in - s_in, dim=2, p=2) 
             distances = dists.view(B, k)
        elif Config.METRIC_TYPE == 'cosine':
             sim_values = torch.gather(sim_coarse, 1, topk_indices)
             distances = 1.0 - sim_values
             distances = torch.clamp(distances, min=0.0)
        else:
             raise ValueError(f"Unknown Metric Type: {Config.METRIC_TYPE}")
             
        # 5. Virtual Outliers (Dual Stream)
        vo_distances = None
        if virtual_outliers is not None:
            # virtual_outliers: (Num_VO, C, H, W)
            # Usually Num_VO is small (e.g. 100 classes).
            # We compute distance from EACH query to ALL virtual outliers (or closest ones).
            # But Sinkhorn is expensive (B * Num_VO * N^2). 
            # If Num_VO is 100, B=128 => 12800 Sinkhorns. Might be slow.
            # Proposal says: "Calculate d_VO to Virtual Outlier set". Usually implies min distance or avg distance.
            # Strategy: Filter top-K VOs using SPP as well?
            
            V = virtual_outliers.shape[0]
            v_flat = virtual_outliers.view(V, C, N).permute(0, 2, 1).contiguous()
            v_norm = torch.nn.functional.normalize(v_flat, dim=2)
            
            # Coarse filter for VOs
            if Config.METRIC_TYPE == 'mahalanobis' and precision_matrix is not None:
                 # Calculate L again? Or assume L is cached/available?
                 # We need to re-compute L or pass it. 
                 # Optimization: Pass L instead of P? 
                 # For now, re-compute L (fast)
                 try:
                     L = torch.linalg.cholesky(precision_matrix)
                 except:
                     u, s, v = torch.svd(precision_matrix)
                     L = torch.matmul(u, torch.diag(torch.sqrt(s)))
                 
                 # Project VOS
                 # v_flat: (V, N, C) -> (V, N, C)
                 v_flat = torch.matmul(v_flat, L)
                 v_norm = v_flat # already projected
                 
                 # SPP for VO (Projected)
                 v_spp = v_norm.view(V, -1)
                 
                 # Recalculate q_spp using updated q_norm (projected)?
                 # q_spp is already set above to q_norm.view(-1).
                 # So we are consistent.
                 
            else:
                 v_spp = spp_func(virtual_outliers)
                 v_spp = torch.nn.functional.normalize(v_spp, dim=1)
            
            sim_vo = torch.mm(q_spp, v_spp.T) # (B, V)
            
            # Select top-K close VOs? Or top-1?
            # Usually we want the closest "boundary" point.
            # Let's pick Top-K (Config.K_NEIGHBORS) same as ID
            k_vo = min(Config.K_NEIGHBORS, V)
            _, topk_vo_indices = torch.topk(sim_vo, k_vo, dim=1) # (B, K_vo)
            
            # Gather VOs
            v_batch = v_norm[topk_vo_indices] # (B, K, N, C)
            q_batch_vo = q_norm.unsqueeze(1).expand(-1, k_vo, -1, -1)
            
            # Flatten
            bs_vo = B * k_vo
            q_in_vo = q_batch_vo.reshape(bs_vo, N, C)
            v_in_vo = v_batch.reshape(bs_vo, N, C)
            
            # OT Cost
            scores_vo = torch.bmm(q_in_vo, v_in_vo.transpose(1, 2))
            M_vo = 1 - scores_vo
            M_vo = torch.clamp(M_vo, min=0.0)
            
            if Config.METRIC_TYPE == 'sinkhorn':
                 d_vo_flat = self.batch_sinkhorn_torch(M_vo, Config.SINKHORN_EPS, Config.SINKHORN_MAX_ITER)
                 vo_distances = d_vo_flat.view(B, k_vo)
            else:
                 # Fallback
                 sim_vo_vals = torch.gather(sim_vo, 1, topk_vo_indices)
                 vo_distances = 1.0 - sim_vo_vals
                 vo_distances = torch.clamp(vo_distances, min=0.0)

        return distances, topk_indices, vo_distances

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
