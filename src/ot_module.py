import torch
import ot
import numpy as np
from src.config import Config

class OTMetric:
    def __init__(self, device='cpu'):
        self.device = device
        
    def compute_sinkhorn_distance(self, feature_map_a, feature_map_b):
        """
        Computes Sinkhorn distance between two feature maps.
        feature_map_a: (C, H, W) -> (C, N) where N=H*W
        feature_map_b: (C, H, W) -> (C, N)
        """
        # Flatten spatial dimensions: (C, H, W) -> (N, C) so each row is a feature vector
        # Note: Input shape handling. The model returns (B, 512, 4, 4).
        # We assume this function handles a single pair or batched pair. 
        # For simplicity in this demo, let's assume single pair inputs (C, H, W).
        
        # Reshape to (N, C)
        fa = feature_map_a.reshape(feature_map_a.shape[0], -1).T # (16, 512)
        fb = feature_map_b.reshape(feature_map_b.shape[0], -1).T # (16, 512)
        
        # Normalize features?
        # fa = torch.nn.functional.normalize(fa, dim=1)
        # fb = torch.nn.functional.normalize(fb, dim=1)
        
        # Cost Matrix: Cosine distance usually preferred, or Euclidean.
        # Plan says "Cosine distance"
        # M = 1 - CosineSimilarity(fa, fb)
        
        # Using Euclidean for generic OT usually works well too, but let's stick to plan if possible.
        # pot.dist calculates squared euclidean distance by default.
        # For cosine: 1 - A.B / (|A||B|)
        
        # Let's convert to numpy for POT
        fa_np = fa.detach().cpu().numpy().astype(np.float64)
        fb_np = fb.detach().cpu().numpy().astype(np.float64)
        
        # Normalize to ensure stability
        fa_np /= (np.linalg.norm(fa_np, axis=1, keepdims=True) + 1e-8)
        fb_np /= (np.linalg.norm(fb_np, axis=1, keepdims=True) + 1e-8)
        
        # Cost matrix: 1 - cosine similarity
        M = 1 - np.dot(fa_np, fb_np.T)
        
        # Distribution weights: Uniform
        n_a = fa_np.shape[0]
        n_b = fb_np.shape[0]
        a = np.ones((n_a,)) / n_a
        b = np.ones((n_b,)) / n_b
        
        # Sinkhorn
        # reg is lambda/entropy parameter. 
        distance = ot.sinkhorn2(a, b, M, reg=Config.SINKHORN_EPS, numItermax=Config.SINKHORN_MAX_ITER)
        
        # ot.sinkhorn2 returns a scalar (float) or array depending on input.
        # Here we have single inputs so it returns a scalar float/tensor
        return distance

    def compute_batch_ot(self, query_features, support_features, support_labels):
        """
        query_features: (B, C, H, W)
        support_features: (S, C, H, W)
        support_labels: (S,)
        
        Returns: Evidence vector (B, NumClasses) based on OT distances
        """
        # This is computationally expensive.
        # Strategy:
        # 1. Avg Pooling for Coarse Filtering (Euclidean/Cosine) -> Top K candidates
        # 2. Fine-grained OT on Top K
        
        B = query_features.size(0)
        S = support_features.size(0)
        
        # Coarse step
        q_avg = torch.mean(query_features, dim=[2, 3]) # (B, C)
        s_avg = torch.mean(support_features, dim=[2, 3]) # (S, C)
        
        q_avg = torch.nn.functional.normalize(q_avg, dim=1)
        s_avg = torch.nn.functional.normalize(s_avg, dim=1)
        
        # Similarity matrix (B, S)
        sim = torch.mm(q_avg, s_avg.T) 
        
        # Select Top K
        k = min(Config.K_NEIGHBORS, S)
        topk_vals, topk_indices = torch.topk(sim, k, dim=1) # (B, K)
        
        distances = np.zeros((B, k))
        
        # Fine step (Looping for now, can be parallelized but OT is tricky to batch fully efficiently without custom kernel)
        for i in range(B):
            q_map = query_features[i]
            for j in range(k):
                s_idx = topk_indices[i, j]
                s_map = support_features[s_idx]
                
                # Compute OT
                dist = self.compute_sinkhorn_distance(q_map, s_map)
                distances[i, j] = dist
                
        return distances, topk_indices
