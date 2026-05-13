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
            
            # Save to catalog
            safe_name = tag_name.replace(" ", "_").lower()
            self.catalog_vectors[safe_name] = nn.Parameter(vector.to(self.device))
            self.catalog_metadata[safe_name] = report
            
            return True, report
        
        # Refusal: Do not learn the tag if the textbook filter fails
        return False, report

    def compute_composite_target(self, tag_weights: Dict[str, float]) -> torch.Tensor:
        """
        Compute the multi-scalar superposition of requested tags.
        
        Weights are unbound (can be >1 or <0), enabling hyperbolic exploration
        and feature subtraction.
        """
        target = torch.zeros(self.state_dim, device=self.device)
        
        if not tag_weights or len(self.catalog_vectors) == 0:
            return target
            
        for tag, weight in tag_weights.items():
            safe_name = tag.replace(" ", "_").lower()
            if safe_name in self.catalog_vectors:
                # Linear superposition
                target += self.catalog_vectors[safe_name] * weight
                
        return target

    def get_catalog_summary(self) -> Dict[str, Dict]:
        """Returns a summary of the currently learned coordinates."""
        return {
            tag: {
                "admissibility": self.catalog_metadata[tag].is_admissible if tag in self.catalog_metadata else True,
                "norm": self.catalog_vectors[tag].norm().item()
            }
            for tag in self.catalog_vectors.keys()
        }
