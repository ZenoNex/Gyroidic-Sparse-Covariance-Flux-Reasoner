import torch
import torch.nn as nn
import os
import json
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
    Handles dynamic fingerprint dimensions (137 legacy, 96 Chebyshev un-lobotomized).
    """
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        # Dynamic projectors to handle different input standards
        self.image_proj_137 = nn.Linear(137, feature_dim)
        self.image_proj_96 = nn.Linear(96, feature_dim)
        self.text_proj = nn.Linear(feature_dim, feature_dim)
        
        # Torsion operator: computes the 'twist' between the two vectors
        self.torsion_matrix = nn.Parameter(torch.randn(feature_dim, feature_dim))
        
    def forward(self, 
                image_fingerprint: torch.Tensor, 
                text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute Residue R = Torsion(I, L).
        Automatically aligns input dimensions to feature_dim.
        """
        # Handle input dimension drift (Anti-Lobotomy alignment)
        in_dim = image_fingerprint.size(-1)
        if in_dim == 137:
            img_proj = self.image_proj_137(image_fingerprint)
        elif in_dim == 96:
            img_proj = self.image_proj_96(image_fingerprint)
        else:
            # Fallback zero-pad or trim to 137
            padded = torch.zeros(*image_fingerprint.shape[:-1], 137, device=image_fingerprint.device)
            min_dim = min(in_dim, 137)
            padded[..., :min_dim] = image_fingerprint[..., :min_dim]
            img_proj = self.image_proj_137(padded)

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

    def export_agent_smith(self, 
                           dyad: KnowledgeDyad, 
                           prime_frequencies: torch.Tensor, 
                           betti_numbers: Dict[int, float], 
                           filename: str = "soliton_smith") -> str:
        """
        Exports the mathematical identity of the agent.
        The Agent State is exported purely as symbolic residue tuples, prime-ladder frequencies,
        and topological invariant shapes (Betti numbers), achieving extraction free of 
        local hardware/latent representations.
        """
        # Ensure we have a valid json filename
        if not filename.endswith(".json"):
             filename += ".json"
             
        payload = {
            "type": "soliton_smith",
            "description": dyad.linguistic_description,
            "gyroid_residue": dyad.gyroid_residue.tolist() if dyad.gyroid_residue is not None else None,
            "prime_frequencies": prime_frequencies.tolist() if isinstance(prime_frequencies, torch.Tensor) else prime_frequencies,
            "betti_numbers": betti_numbers,
            "timestamp": dyad.timestamp
        }
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)
        return filepath

    def inject_agent_smith(self, filepath: str) -> Dict:
        """
        Loads the mathematical identity of an agent (Agent Smith) back into the system,
        allowing the local hardware to 'breathe' its own unique life into the configuration.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Agent Smith file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            payload = json.load(f)
             
        # Minimal validation
        if payload.get("type") != "soliton_smith":
             raise ValueError("File is not a valid Agent Smith (soliton_smith) JSON payload.")
             
        return payload

