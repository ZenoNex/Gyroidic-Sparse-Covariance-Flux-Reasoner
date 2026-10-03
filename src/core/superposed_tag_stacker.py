import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from src.data.textbook_filter import TextbookFilter, QualityReport

class SuperposedTagStacker(nn.Module):
    """
    Ganbreeder-style Vector Stacker for Non-Simplifying Coordinate Superposition.
    
    Replaces conformal snapping (argmax cosine similarity) with a multi-scalar
    linear combination of dynamic, textbook-filtered coordinates.
    
    To 'intercosaminate its learning', new coordinate additions are gated by the
    Phi-1 TextbookFilter. Only textbook-quality contexts are allowed to assign
    semantic tags to topological vectors.
    """
    def __init__(self, state_dim: int, device: str = None):
        super().__init__()
        self.state_dim = state_dim
        self.device = device if device is not None else 'cpu'
        
        # Textbook Filter to gate new tag associations
        self.textbook_filter = TextbookFilter()
        
        # Dynamic Coordinate Catalog: {tag_name: (vector, quality_report)}
        # We store them in a ParameterDict or simply as parameters to support checkpointing.
        self.catalog_vectors = nn.ParameterDict()
        
        # In-memory storage for metadata
        self.catalog_metadata = {}

    @staticmethod
    def sanitize_tag_name(tag_name: str) -> str:
        """Sanitizes tag names for PyTorch ParameterDict compatibility (no dots or special characters)."""
        import re
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', tag_name.strip()).lower()
        clean = re.sub(r'_+', '_', clean).strip('_')
        return clean or "unnamed_tag"

    def __contains__(self, tag_name: str) -> bool:
        return self.sanitize_tag_name(tag_name) in self.catalog_vectors

    def add_tag(self, tag_name: str, vector: torch.Tensor, context_text: str) -> Tuple[bool, QualityReport]:
        """
        Harvest a new coordinate and bind it to a semantic tag.
        
        INTERCOSAMINATED LEARNING:
        The context_text is evaluated against the 5 non-scalar Phi-1 dimensions.
        If it fails, the system refuses to learn the coordinate, maintaining
        structural honesty.
        """
        # Assess semantic quality
        report = self.textbook_filter.assess(context_text, source="stacker_learning")
        
        if report.is_admissible:
            # Ensure proper shape [state_dim]
            if vector.dim() > 1:
                vector = vector.flatten()[:self.state_dim]
            elif vector.shape[0] < self.state_dim:
                vector = torch.nn.functional.pad(vector, (0, self.state_dim - vector.shape[0]))
            else:
                vector = vector[:self.state_dim]
                
            # Normalize vector to ensure stable scalar stacking
            vector = torch.nn.functional.normalize(vector.float(), dim=-1)
            
            # Save to catalog with sanitized name (no dots allowed in ParameterDict keys)
            safe_name = self.sanitize_tag_name(tag_name)
            self.catalog_vectors[safe_name] = nn.Parameter(vector.to(self.device))
            self.catalog_metadata[safe_name] = report
            
            return True, report
        
        # Refusal: Do not learn the tag if the textbook filter fails
        return False, report

    def compute_composite_target(
        self, 
        tag_weights: Optional[Dict[str, float]] = None, 
        current_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the multi-scalar superposition of requested tags.
        
        If tag_weights is None/empty and current_state is provided, weights
        are automatically derived by projecting current_state onto catalog vectors
        using cosine similarity.
        
        Weights are unbound (can be >1 or <0), enabling hyperbolic exploration
        and feature subtraction.
        """
        target = torch.zeros(self.state_dim, device=self.device, dtype=torch.float64)
        
        if (tag_weights is None or len(tag_weights) == 0) and current_state is not None:
            if len(self.catalog_vectors) > 0:
                derived_weights = {}
                state_val = current_state.detach().to(self.device).float()
                
                # Standardize state vector to 1D [state_dim]
                if state_val.dim() > 1:
                    state_vec = state_val.mean(dim=list(range(state_val.dim() - 1)))
                else:
                    state_vec = state_val
                
                if state_vec.shape[0] > self.state_dim:
                    state_vec = state_vec[:self.state_dim]
                elif state_vec.shape[0] < self.state_dim:
                    state_vec = torch.nn.functional.pad(state_vec, (0, self.state_dim - state_vec.shape[0]))
                
                norm_state = torch.nn.functional.normalize(state_vec, dim=-1)
                
                for name, param in self.catalog_vectors.items():
                    norm_param = torch.nn.functional.normalize(param.float(), dim=-1)
                    cos_sim = torch.dot(norm_state, norm_param).item()
                    derived_weights[name] = cos_sim
                
                tag_weights = derived_weights
            else:
                return target.to(torch.float32)

        if not tag_weights or len(self.catalog_vectors) == 0:
            return target.to(torch.float32)
            
        import math
        for tag, val in tag_weights.items():
            safe_name = self.sanitize_tag_name(tag)
            if safe_name in self.catalog_vectors:
                param = self.catalog_vectors[safe_name].to(self.device).to(torch.float64)
                
                # Check if val is dictionary with user-selectable quotient gamma
                if isinstance(val, dict):
                    u = float(val.get('u', val.get('weight', 0.0)))
                    gamma = float(val.get('gamma', 1.0))
                    if abs(gamma) < 1e-4:
                        alpha = u
                    else:
                        sign_u = 1.0 if u >= 0 else -1.0
                        abs_u = abs(u)
                        try:
                            alpha = sign_u * (math.sinh(gamma * abs_u) / (math.sinh(gamma) + 1e-12))
                        except OverflowError:
                            alpha = sign_u * 10.0
                elif isinstance(val, (int, float)):
                    alpha = float(val)
                else:
                    try:
                        alpha = float(val)
                    except Exception:
                        alpha = 0.0
                        
                # Fractional vector superposition retaining rational precision
                target += param * alpha
                
        return target.to(torch.float32)

    def get_catalog_summary(self) -> Dict[str, Dict]:
        """Returns a summary of the currently learned coordinates."""
        return {
            tag: {
                "admissibility": self.catalog_metadata[tag].is_admissible if tag in self.catalog_metadata else True,
                "norm": self.catalog_vectors[tag].norm().item()
            }
            for tag in self.catalog_vectors.keys()
        }


class FractionalQuotientTagStacker(nn.Module):
    """
    Superposed Tag Stacker supporting user-selectable hyperbolic quotients (gamma)
    and fractional quotient algebra without premature integer truncation.
    """
    def __init__(self, dim: int = 256, num_functionals: int = 8):
        super().__init__()
        self.dim = dim
        self.num_functionals = num_functionals

    def forward(self, base_vector: torch.Tensor, tags: list) -> torch.Tensor:
        """
        base_vector: [Batch, Dim] float/rational tensor
        tags: list of dicts {'vector': Tensor, 'u': float, 'gamma': float}
        """
        import math
        target_vector = base_vector.clone().to(torch.float64)
        
        for tag in tags:
            u = float(tag.get('u', 0.0))
            gamma = float(tag.get('gamma', 1.0))
            tag_vec = tag['vector'].to(torch.float64)
            
            # 1. Apply user-selectable quotient of hyperbolicity
            if abs(gamma) < 1e-4:
                alpha = u
            else:
                sign_u = 1.0 if u >= 0 else -1.0
                abs_u = abs(u)
                try:
                    alpha = sign_u * (math.sinh(gamma * abs_u) / (math.sinh(gamma) + 1e-12))
                except OverflowError:
                    alpha = sign_u * 10.0
            
            # 2. Superpose in continuous fractional vector space
            target_vector = target_vector + alpha * tag_vec

        # 3. Fractional Modular Quotient Projection
        return target_vector.to(torch.float32)

