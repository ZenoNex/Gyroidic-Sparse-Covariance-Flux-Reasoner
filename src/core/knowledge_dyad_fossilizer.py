import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import datetime

@dataclass
class KnowledgeDyad:
    """
    A single unit of multi-modal knowledge: (Image Fingerprint, Linguistic Description).
    Acts as a 'Topological Obstruction' in the manifold.
    """
    image_fingerprint: torch.Tensor # [137] vector
    linguistic_description: str
    gyroid_residue: Optional[torch.Tensor] = None # [n, n] irreducible entanglement
    relevance_score: float = 1.0
    timestamp: str = ""
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()

class ResidueFusion(nn.Module):
    """
    Computes the 'Cross-Modality Torsion' between image and text features.
    Unlike standard fusion (concatenation), this computes the *mismatch* (residue).
    """
    def __init__(self, feature_dim: int = 512, fingerprint_dim: int = 137):
        super().__init__()
        self.image_proj = nn.Linear(fingerprint_dim, feature_dim)
        self.text_proj = nn.Linear(feature_dim, feature_dim) # Assuming text is already embedded
        
        # Torsion operator: computes the 'twist' between the two vectors
        self.torsion_matrix = nn.Parameter(torch.randn(feature_dim, feature_dim))
        
    def forward(self, 
                image_fingerprint: torch.Tensor, 
                text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute Residue R = Torsion(I, L).
        R represents the 'unexplained' part of the dyad.
        """
        img_proj = self.image_proj(image_fingerprint)
        txt_proj = self.text_proj(text_embedding)
        
        # Calculate torsion: (I - L) varies with the metric twist
        diff = img_proj - txt_proj
        torsion = torch.matmul(diff, self.torsion_matrix)
        
        # The residue is the magnitude of this torsion
        residue = torch.tanh(torsion) 
        
        return residue

class DyadFossilizer:
    """
    Handles the persistent storage ('Fossilization') of Knowledge Dyads.
    Ensures 'No Erasing of Implication' by saving precise states to disk.
    """
    
    def __init__(self, 
                 storage_dir: str = "data/encodings",
                 fusion_layer: Optional[ResidueFusion] = None):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.fusion_layer = fusion_layer or ResidueFusion()
        
    def compute_poincaré_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a Euclidean vector x to the Poincaré disk B^n (System 2 Speculative Recovery).
        Formula: z = 2 * tanh(dist/2) * unit(x).
        This unfolding prevents NaN/INF collapse by providing non-Euclidean volume.
        """
        norm = torch.norm(x, dim=-1, keepdim=True)
        eps = 1e-8
        safe_norm = torch.clamp(norm, min=eps)
        # User dynamic: z -> 2 tanh(dist/2)
        scale = 2.0 * torch.tanh(safe_norm / 2.0)
        return scale * (x / safe_norm)

    def fossilize(self, 
                  dyad: KnowledgeDyad, 
                  text_embedding: torch.Tensor) -> str:
        """
        Save the dyad and its computed residue to disk.
        Returns the filename of the fossil.
        """
        # 1. Compute Residue (The 'Meaning' of the association)
        # Ensure inputs are tensors
        if not isinstance(dyad.image_fingerprint, torch.Tensor):
             img_tensor = torch.tensor(dyad.image_fingerprint)
        else:
             img_tensor = dyad.image_fingerprint
             
        residue = self.fusion_layer(img_tensor, text_embedding)
        
        # 2. System 2 Hyperbolic Unfolding (Speculative Recovery)
        # We only perform this 'expensive' magic during fossilization to save heuristic speed.
        hyperbolic_residue = self.compute_poincaré_embedding(residue)
        
        # 3. Prepare Payload
        payload = {
            'type': 'knowledge_dyad',
            'description': dyad.linguistic_description,
            'image_fingerprint': dyad.image_fingerprint,
            'residue_vector': residue.detach().cpu(),
            'gyroid_residue': dyad.gyroid_residue.detach().cpu() if dyad.gyroid_residue is not None else None,
            'hyperbolic_residue': hyperbolic_residue.detach().cpu(),
            'timestamp': dyad.timestamp,
            'metrics': {
                'relevance': dyad.relevance_score,
                'hyperbolic_eccentricity': torch.norm(hyperbolic_residue).item()
            }
        }
        
        # 4. Save to Disk (Safe, atomic-like write)
        safe_desc = "".join(c for c in dyad.linguistic_description[:20] if c.isalnum())
        filename = f"encoding_{safe_desc}_{int(datetime.datetime.now().timestamp())}.pt"
        filepath = os.path.join(self.storage_dir, filename)
        
        torch.save(payload, filepath)
        
        return filepath
        
    def recover_fossils(self) -> List[Dict]:
        """Load all fossilized dyads for 'Speculative Coprime Gating'."""
        fossils = []
        if not os.path.exists(self.storage_dir):
            return fossils
            
        for f in os.listdir(self.storage_dir):
            if f.endswith(".pt"):
                filepath = os.path.join(self.storage_dir, f)
                try:
                    data = torch.load(filepath)
                    if isinstance(data, dict) and 'residue_vector' in data:
                        fossils.append(data)
                    else:
                        print(f"[RECOVERY] Deleting invalid fossil (missing residue_vector): {f}")
                        os.remove(filepath)
                except Exception as e:
                    print(f"[RECOVERY] Deleting corrupted fossil {f}: {e}")
                    try:
                        os.remove(filepath)
                    except:
                        pass
        return fossils
