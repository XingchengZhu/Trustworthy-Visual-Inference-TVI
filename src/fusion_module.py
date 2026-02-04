import torch

class DempsterShaferFusion:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        
    def ds_combination(self, alpha1, alpha2):
        """
        Combine two sets of Dirichlet alphas (evidence + 1).
        
        Let e1, e2 be evidence.
        b (belief) = e / S, where S = sum(e) + K
        u (uncertainty) = K / S
        
        Dempster's Rule for two mass functions m1, m2:
        m(A) = (m1(A)m2(Θ) + m1(Θ)m2(A) + m1(A)m2(A)) / (1 - C)  <-- This is approximate / specific interpretation
        
        Actually, for DST on Dirichlet/Evidence Deep Learning:
        There are different versions. 
        One common version (Han et al., NeurIPS 2018 / Sensoy et al.):
        Belief masses assignment:
        b_k = e_k / S, u = num_classes / S
        
        The plan mentions:
        Message 1: {b1_k, u1}
        Message 2: {b2_k, u2}
        
        Fused belief:
        b_fuse_k = (b1_k * b2_k + b1_k * u2 + b2_k * u1) / (1 - C)
        u_fuse = (u1 * u2) / (1 - C)
        C = sum_{i!=j} b1_i * b2_j  (Conflict)
        """
        
        # Calculate belief mass and uncertainty for source 1
        S1 = torch.sum(alpha1, dim=1, keepdim=True)
        b1 = (alpha1 - 1) / S1
        u1 = self.num_classes / S1
        
        # Calculate belief mass and uncertainty for source 2
        S2 = torch.sum(alpha2, dim=1, keepdim=True)
        b2 = (alpha2 - 1) / S2
        u2 = self.num_classes / S2
        
        # Compute Conflict C
        # C = sum(b1_i * b2_j) for i != j
        # => sum(b1_i * (sum(b2) - b2_i))
        # => sum(b1_i * sum(b2)) - sum(b1_i * b2_i)
        # Note: sum(b) = 1 - u
        
        sum_b1 = 1 - u1
        sum_b2 = 1 - u2
        
        sum_b1_b2_dot = torch.sum(b1 * b2, dim=1, keepdim=True)
        
        C = sum_b1 * sum_b2 - sum_b1_b2_dot
        
        # Normalize factor
        norm = 1 - C + 1e-8 # add epsilon to avoid div by zero
        
        # Fused Belief
        # b_fuse = (b1 * b2 + b1 * u2 + b2 * u1) / norm
        b_fuse = (b1 * b2 + b1 * u2 + b2 * u1) / norm
        
        # Fused Uncertainty
        u_fuse = (u1 * u2) / norm
        
        # Convert back to alpha (optional, or just return probabilities)
        # S_fuse = K / u_fuse
        # e_fuse = b_fuse * S_fuse
        # alpha_fuse = e_fuse + 1
        
        S_fuse = self.num_classes / (u_fuse + 1e-8)
        alpha_fuse = b_fuse * S_fuse + 1
        
        return alpha_fuse, u_fuse, C
