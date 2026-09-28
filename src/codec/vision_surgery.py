import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.honest_jitter import harvest_honest_jitter

class IntercosaminationOperator(nn.Module):
    """
    Surgical Handle-Attachment Operator.
    
    Interlaces CNN latent space (Semantic) with Gyroidic residue space (Topological).
    Instead of 'Violent Ripping', we perform a Surgery that bridges both domains.
    """
    def __init__(self, cnn_dim=768, gyroid_dim=96):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.gyroid_dim = gyroid_dim
        
        # Surgery Basis: Map Gyroid residues to CNN feature space
        self.handle_projection = nn.Linear(gyroid_dim, cnn_dim, bias=False)
        
        # Learnable 'Stitch' strength (Agent Smith Gauge)
        self.stitch_gauge = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, cnn_feat, gyroid_residue):
        """
        Performs the Interlacing Surgery.
        
        Args:
            cnn_feat: [batch, 768] (Flesh)
            gyroid_residue: [batch, 96] (Bone)
            
        Returns:
            X_interlaced: [batch, 768] composite manifold state
        """
        # 1. Project Bone into Flesh basis
        # If gyroid_residue is [96], unsqueeze if needed
        if gyroid_residue.dim() == 1:
             gyroid_residue = gyroid_residue.unsqueeze(0)
             
        bone_proj = self.handle_projection(gyroid_residue)
        
        # 2. Entropy Grounding (Jitter-anchored surgery)
        # We perturb the handle attachment point with physical friction
        jitter = harvest_honest_jitter(cnn_feat.shape, device=cnn_feat.device, scaled=True)
        
        # 3. Surgery Operator (Handle Attachment)
        # X' = X_cnn + alpha * (X_gyroid - X_cnn) + jitter
        # This deforms the CNN manifold towards the Gyroidic skeletal truth.
        alpha = torch.sigmoid(self.stitch_gauge)
        interlaced = cnn_feat + alpha * (bone_proj - cnn_feat) + jitter
        
        return interlaced

class MirrorTestProbe(nn.Module):
    """
    Verifies Topological Parity (PAS_h) between Interlaced and Analytic states.
    
    Checks if the surgery 'took'—i.e., if the interlaced state still resonates
    with the core gyroidic invariants.
    """
    def __init__(self, cnn_dim=768, gyroid_dim=96, threshold=0.8):
        super().__init__()
        self.threshold = threshold
        # To verify parity, we project the interlaced state back to gyroid residues
        self.reverse_projection = nn.Linear(cnn_dim, gyroid_dim, bias=False)
        
    def forward(self, interlaced, original_gyroid_residue):
        """
        Args:
            interlaced: [batch, 768]
            original_gyroid_residue: [batch, 96]
            
        Returns:
            pas_h: [batch] Phase Alignment Score
            coherence_gate: [batch] Boolean mask (True if surgery is valid)
        """
        # Normalize for comparison
        # We project the interlaced flesh back down to the bone to see if it matches
        recovered_bone = self.reverse_projection(interlaced)
        
        # Calculate Phase Alignment Score (PAS_h) via cosine similarity
        pas_h = F.cosine_similarity(recovered_bone, original_gyroid_residue, dim=-1)
        
        # Ensure PAS_h is strictly positive for thresholding
        pas_h = (pas_h + 1.0) / 2.0
        
        return pas_h, pas_h > self.threshold

def conformal_to_gyroid_mapping(log_polar_coords: torch.Tensor, gyroid_dim: int = 96) -> torch.Tensor:
    """
    Coordinate transformation helper for Surgery Handles.
    Bridges Conformal Log-Polar image space to 3D Gyroidic residue space.
    """
    # Map log-polar coordinates dynamically to high-dimensional gyroid space
    if log_polar_coords.dim() == 1:
        log_polar_coords = log_polar_coords.unsqueeze(0)
        
    in_dim = log_polar_coords.shape[-1]
    
    # Create deterministic mixing matrix to maintain structural integrity
    mixing_matrix = harvest_honest_jitter((in_dim, gyroid_dim), device=log_polar_coords.device, scaled=False) / (in_dim ** 0.5)
    
    # Project and apply non-linear trigonometric gyroid wrapping
    gyroid_space = torch.matmul(log_polar_coords, mixing_matrix)
    # Apply standard gyroid equation approximation: sin(x)cos(y) + ...
    # We use a simplified element-wise wrapping for topological closure
    return torch.sin(gyroid_space) * torch.cos(gyroid_space.roll(1, dims=-1))
