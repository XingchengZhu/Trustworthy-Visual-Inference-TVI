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
            # M is actually squared euclidean / something if we used cosine logic? No.
            # M above was calculated as 1 - cosine. Cost Matrix.
            # If we want Euclidean, we should ignore M construction above?
            # Or just use the M (Cosine Distance) as the distance directly?
            # User specifically asked for "Variant A: Euclidean OR Cosine".
            # My current M is Cosine Distance (1 - sim).
            # The structure of this method is heavily optimized for Sinkhorn (batch M).
            
            # If Metric is Cosine Distance (without OT, just sum or min? No, usually Avg dist or Min dist).
            # But "OT-based Evidence" usually implies a distribution distance.
            # If "Euclidean", maybe we just want the distance to the K neighbors?
            
            # Let's interpret "Euclidean/Cosine as k-NN distance".
            # OT computes Earth Mover between Query Distribution and Support Distribution.
            # If we disable OT, we treat it as simple k-NN.
            # Usually Evidence = RBF(dist).
            # If standard k-NN, distance is usually Mean of Distances to K neighbors, or Min.
            
            # Let's use Mean Distance to K neighbors as the "No-OT" baseline of "Set Distance".
            # Reusing M (which is Cost/Distance Matrix).
            # M: (B*K, N, N). We have pixel-to-pixel costs.
            # If we don't do OT, how do we aggregate pixel costs?
            # Default: Mean over pixels.
            
            # Dist = Mean(M) over (N, N) ? No, that's just global diff.
            # Diagonal? No, not aligned.
            
            # Wait, if we use Euclidean/Cosine as "Metric", it likely means 
            # we don't do Sinkhorn on the cost matrix M, but maybe we just sum it?
            # OR, we define the distance between Image A and Image B differently.
            # Currently my code decomposes images into N pixels.
            # OT aligns pixels.
            
            # If I use Euclidean, I should probably treat images as vectors.
            # (B, C, H, W) -> (B, D).
            # distance(a, b) = ||a - b||.
            
            # Let's stick to the prompt's likely intent:
            # "Why OT? Euclidean doesn't work for deformation."
            # So I should compute Euclidean distance between the full images.
            
            # My current `compute_batch_ot` first finds top-K neighbors using Cosine on GAP.
            # Then it does fine-grained matching.
            
            # If Config.METRIC_TYPE == 'euclidean':
            # We should probably compute exact Euclidean between Q and S_neighbors.
            # q_in: (B*K, N, C). s_in: (B*K, N, C).
            # Euclidean: sum((q - s)^2) per pixel?
            # No, that requires valid spatial alignment (pixel 1 to pixel 1).
            # That is exactly what Euclidean assumes (no shuffle).
            
            diff = q_in - s_in # (B*K, N, C)
            dist_sq = torch.sum(diff ** 2, dim=[1, 2]) # Sum over Spatial and Channels
            distances = torch.sqrt(dist_sq).view(B, k)
            
        elif Config.METRIC_TYPE == 'cosine':
            # Cosine Distance between full feature vectors.
            # q_in, s_in are normalized per pixel in the preparation steps? 
            # q_norm in step 1 was normalized on dim=2 (Channels).
            # q_in is (B*K, N, C).
            
            # We want Global Cosine Distance? 
            # Or just Mean of Pixel Cosine Distances?
            # M is (B*K, N, N) containing 1 - cos(p_i, p_j).
            # If we assume no alignment, we compare p_i with p_i.
            # That corresponds to the diagonal of M.
            
            # M_diag = M[:, i, i]
            # Average diagonal is the mean cosine distance of corresponding pixels.
            
            # But M was constructed as bmm(q, s.T). 
            # q: (M, N, C). s: (M, N, C).
            # scores = q @ s.T -> (M, N, N). scores[b, i, j] = q[b, i] . s[b, j]
            # Diagonal: scores[b, i, i] = q[b, i] . s[b, i]
            
            scores_diag = torch.diagonal(scores, dim1=1, dim2=2) # (B*K, N)
            # Sum over N? Mean over N?
            # Cosine Sim of images = (A . B) / |A||B|.
            # If pixels are normalized, Sum(A_i . B_i) is roughly A.B?
            # Let's use Mean Diagonal Cost (1 - similarity).
            
            dist_per_px = 1 - scores_diag # (B*K, N)
            distances = torch.mean(dist_per_px, dim=1).view(B, k)
            
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
