import torch
from src.config import Config

class DempsterShaferFusion:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        
    def compute_entropy(self, alpha):
        """
        Compute Normalized Shannon Entropy of the expected probability distribution.
        p = alpha / sum(alpha)
        H = -sum(p * log(p)) / log(K)
        """
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        
        # Clip probs to avoid log(0)
        probs = torch.clamp(probs, min=1e-8)
        
        entropy = -torch.sum(probs * torch.log(probs), dim=1)
        
        # Normalize by log(K)
        max_entropy = torch.log(torch.tensor(float(self.num_classes), device=alpha.device))
        norm_entropy = entropy / max_entropy
        
        # Clamp to [0,1] just in case
        norm_entropy = torch.clamp(norm_entropy, 0.0, 1.0)
        
        return norm_entropy

    def ds_combination(self, alpha1, alpha2, discount_factor=None):
        """
        Combine two sets of Dirichlet alphas (evidence + 1).
        
        Parameters:
        alpha1: Parametric Alpha (usually)
        alpha2: Non-Parametric Alpha
        discount_factor: (B,) - If provided (e.g., entropy of alpha1), apply mass discounting.
        """
        
        if Config.FUSION_TYPE == 'average':
            # Variant A: Simple Average
            alpha_fuse = (alpha1 + alpha2) / 2
            
            # Recalculate Uncertainty for the fused alpha
            S_fuse = torch.sum(alpha_fuse, dim=1, keepdim=True)
            u_fuse = self.num_classes / S_fuse
            
            C = torch.zeros_like(u_fuse)
            
            return alpha_fuse, u_fuse, C

        # Default: Dempster-Shafer
        
        # Calculate belief mass and uncertainty for source 1
        S1 = torch.sum(alpha1, dim=1, keepdim=True)
        b1 = (alpha1 - 1) / S1
        u1 = self.num_classes / S1
        
        # Apply Discounting to Source 1 if provided
        # Formula: m_new = m_old * (1 - entropy)
        # Assigned discounted mass to Uncertainty? 
        # Usually: u_new = u_old + sum(m_old * entropy) = 1 - sum(m_new)
        if discount_factor is not None:
            # discount_factor shape (B,) - High Entropy => High Discount => Low Trust
            # trust = 1 - discount
            trust = (1.0 - discount_factor).unsqueeze(1) 
            trust = torch.clamp(trust, 0.0, 1.0)
            
            b1 = b1 * trust
            # Recalculate u1 (remaining mass goes to uncertainty)
            u1 = 1.0 - torch.sum(b1, dim=1, keepdim=True)
        
        # Calculate belief mass and uncertainty for source 2
        S2 = torch.sum(alpha2, dim=1, keepdim=True)
        b2 = (alpha2 - 1) / S2
        u2 = self.num_classes / S2
        
        # Compute Conflict C
        # C = sum_{i!=j} b1_i * b2_j
        
        sum_b1 = 1 - u1
        sum_b2 = 1 - u2
        
        sum_b1_b2_dot = torch.sum(b1 * b2, dim=1, keepdim=True)
        
        C = sum_b1 * sum_b2 - sum_b1_b2_dot
        
        # Normalize factor
        # Dempster: Norm = 1 - C. (Redistribute C to all propositions proportional to mass)
        # Yager: Norm = 1. (Assign C to Omega/Uncertainty)
        
        if getattr(Config, 'FUSION_STRATEGY', 'dempster') == 'yager':
             # Yager's Rule: C goes to Uncertainty
             # m(A) = sum(b1*b2)
             # m(Omega) = u1*u2 + C
             
             # Calculate raw intersection masses
             # b_fuse_unnorm = b1*u2 + b2*u1 + b1*b2 (Intersection)
             # But strictly Yager:
             # m(A) = sum_{B \cap D = A} m1(B)m2(D)
             # For Dirichlet:
             # m(A): b1(A)b2(A) + b1(A)u2 + u1b2(A)
             # u_new = u1*u2 + C
             
             b_fuse = b1 * u2 + b2 * u1 + b1 * b2
             u_fuse = u1 * u2 + C
             
             # No normalization by (1-C)
             # Check sum
             # sum(b_fuse) + u_fuse = ...
             # sum(b1u2 + b2u1 + b1b2) + u1u2 + C
             # = u2(1-u1) + u1(1-u2) + sum(b1b2) + u1u2 + C
             # = u2 - u1u2 + u1 - u1u2 + sum(b1b2) + u1u2 + C
             # = u1 + u2 - u1u2 + sum(b1b2) + C
             # C = (1-u1)(1-u2) - sum(b1b2)
             # C = 1 - u1 - u2 + u1u2 - sum(b1b2)
             # sum = u1 + u2 - u1u2 + sum(b1b2) + 1 - u1 - u2 + u1u2 - sum(b1b2)
             # sum = 1.
             # Correct.
             
        else:
            # Standard Dempster (Drop C)
            norm = 1 - C + 1e-8 
            b_fuse = (b1 * u2 + b2 * u1 + b1 * b2) / norm
            u_fuse = (u1 * u2) / norm
        
        # Convert back to alpha
        # S = K / u
        S_fuse = self.num_classes / (u_fuse + 1e-8)
        alpha_fuse = b_fuse * S_fuse + 1
        
        # Clip to be safe >= 1
        alpha_fuse = torch.clamp(alpha_fuse, min=1.0)
        
        return alpha_fuse, u_fuse, C

    def adaptive_ds_combination(self, alpha_param, alpha_nonparam):
        """
        Adaptive Fusion: Trust Parametric ONLY if Non-Parametric (OT) is certain.
        If OT is uncertain (high distance), we discount the Parametric belief.
        This fixes the "Confident but Wrong" problem of Softmax on OOD.
        
        Mechanism:
        trust_param = 1.0 - u_ot
        alpha_param_new = (alpha_param - 1) * trust_param + 1
        """
        # 1. Calculate OT Uncertainty
        S_ot = torch.sum(alpha_nonparam, dim=1, keepdim=True)
        # u = K / S
        u_ot = self.num_classes / S_ot
        
        # 2. Calculate Trust Factor
        # If u_ot is 1.0 (Total OOD), trust is 0.0 -> Param becomes uniform/uncertain.
        # If u_ot is 0.0 (Perfect ID), trust is 1.0 -> Param kept as is.
        trust_param = 1.0 - u_ot
        trust_param = torch.clamp(trust_param, 0.0, 1.0)
        
        # 3. Discount Parametric Alpha
        # alpha = evidence + 1
        # new_evidence = old_evidence * trust
        # new_alpha = new_evidence + 1 = (alpha - 1) * trust + 1
        alpha_param_new = (alpha_param - 1) * trust_param + 1
        
        # 4. Standard Fusion with Discounted Param
        return self.ds_combination(alpha_param_new, alpha_nonparam)
