import torch
import torch.nn as nn
import itertools
from typing import Dict, List, Optional
from src.core.honest_jitter import harvest_honest_jitter

class SparseHigherOrderTensorDynamics(nn.Module):
    """
    Implements sparse higher-order tensor interactions using Matrioshka shells.
    Only computes N-th order dynamics along active polytope facets.
    """
    def __init__(self, max_order: int = 3, num_shells: int = 3, base_dim: int = 64):
        super().__init__()
        self.max_order = max_order
        self.num_shells = num_shells
        self.base_dim = base_dim
        
        # Simulate tensor weights for different orders
        # SILICON SOVEREIGNTY: Initialized with Honest Jitter
        self.tensor_weights = nn.ParameterDict({
            str(order): nn.Parameter(harvest_honest_jitter((base_dim, base_dim), scaled=True) * 0.2)
            for order in range(1, max_order + 1)
        })
        
    def forward(self, x: torch.Tensor, active_facets: Optional[List[int]] = None, mode: str = 'PLAY') -> Dict[int, torch.Tensor]:
        """
        Compute tensor dynamics.
        If active_facets is provided, compute sparsely.
        """
        results = {}
        batch_size, dim = x.shape
        
        is_play = mode.upper() in ('PLAY', 'GOO')
        is_serious = mode.upper() in ('SERIOUSNESS', 'PRICKLES')
        
        def _sovereign_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            if not (lhs.requires_grad or rhs.requires_grad) and dim >= 8:
                try:
                    from src.core.hardware_monitor import should_offload_gpu_sovereign
                    gpu_ok, _ = should_offload_gpu_sovereign(dim=dim, min_dim=8)
                    if gpu_ok:
                        from src.core.pyopencl_sovereignty import get_sovereignty_engine
                        engine = get_sovereignty_engine()
                        if engine is not None and engine.is_initialized:
                            return engine.hardware_sovereign_matmul(lhs, rhs).to(device=lhs.device, dtype=lhs.dtype)
                except Exception:
                    pass
            return torch.matmul(lhs, rhs)

        # 1st Order (Linear) - Always computed fully
        w1 = self.tensor_weights['1']
        results[1] = _sovereign_mm(x, w1)
        
        if self.max_order < 2:
            return results
            
        # Determine facets to process
        # Detect clustered nodes of tensor activity using local correlation
        from src.core.martinova_correlation import compute_bounded_correlation
        node_patterns = x.T.unsqueeze(0) # [1, dim, batch]
        node_corrs = compute_bounded_correlation(node_patterns).squeeze(0) # [dim]
        combined_scores = x.abs().mean(dim=0) * (1.0 + node_corrs)
        
        if is_serious:
            k_limit = max(1, int(dim * 0.25))
            top_indices = torch.topk(combined_scores, k=k_limit).indices.tolist()
            if active_facets is None:
                active_facets = top_indices
            else:
                active_facets = [f for f in active_facets if f in top_indices]
                if not active_facets:
                    active_facets = top_indices
        else:
            if active_facets is None:
                active_facets = torch.topk(combined_scores, k=max(1, int(dim * 0.1))).indices.tolist()
            
        # Create a mask for sparse computation
        mask = torch.zeros(dim, device=x.device)
        mask[active_facets] = 1.0
        
        # 2nd Order (Quadratic)
        if self.max_order >= 2:
            w2 = self.tensor_weights['2']
            # Sparse: (x * mask) W (x * mask)^T
            # approximation: element-wise interaction with weight matrix
            # For efficiency in this demo, we mask the input
            x_sparse = x * mask
            results[2] = _sovereign_mm(x_sparse, w2) * x_sparse
            
        # 3rd Order (Cubic)
        if self.max_order >= 3:
            w3 = self.tensor_weights['3']
            if is_play:
                soft_mask = torch.sigmoid(combined_scores)
                x_soft = x * soft_mask
                results[3] = (_sovereign_mm(x_soft, w3) ** 2) * x_soft
            else:
                x_sparse = x * mask
                # Mock cubic interaction
                results[3] = (_sovereign_mm(x_sparse, w3) ** 2) * x_sparse
            
        return results

    def compute_computational_savings(self, x: torch.Tensor, active_facets: List[int]) -> Dict[str, float]:
        """Estimate savings from sparse computation."""
        dim = x.shape[1]
        active = len(active_facets)
        
        full_ops = dim ** 2 # quadratic base
        sparse_ops = active ** 2
        
        sparsity = 1.0 - (active / dim)
        speedup = full_ops / (sparse_ops + 1e-9)
        
        return {
            "sparsity_ratio": sparsity,
            "theoretical_speedup": speedup,
            "active_params": active
        }