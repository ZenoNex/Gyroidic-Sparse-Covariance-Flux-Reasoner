"""
Polynomial functional embeddings for multi-modal inputs.

Projects text, graph, and numerical inputs into polynomial coefficient distributions
for co-prime polynomial functionals.

Author: William Matthew Bryant
Created: January 2026
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union

from ..core.polynomial_coprime import PolynomialCoprimeConfig
from ..core.primitive_ops import FixedPointField, LearnedPrimitivePerturbation
from ..core.gdpo_normalization import GDPONormalization
from ..core.invariants import PHI, get_ihc_ladder_scale



class PolynomialFunctionalEmbedder(nn.Module):
    """
    Multi-modal encoder that projects inputs into polynomial coefficient distributions.
    
    For each polynomial functional φ_k, outputs a distribution over basis coefficients.
    Supports evolutionary saturation for symbolic-first reasoning.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        graph_dim: int = 256,
        num_dim: int = 64,
        hidden_dim: int = 512,
        poly_config: Optional[PolynomialCoprimeConfig] = None,
        use_text: bool = True,
        use_graph: bool = True,
        use_num: bool = True,
        use_saturation: bool = False,
        use_kan_heads: bool = False
    ):
        """
        Args:
            text_dim: Dimension of text embeddings
            graph_dim: Dimension of graph embeddings
            num_dim: Dimension of numerical features
            hidden_dim: Hidden dimension for fusion
            poly_config: Polynomial co-prime configuration
            use_text: Whether to use text modality
            use_graph: Whether to use graph modality
            use_num: Whether to use numerical modality
            use_saturation: Whether to use saturation gate
            use_kan_heads: Whether to use True B-spline KANLayer heads instead of Linear
        """
        super().__init__()
        
        if poly_config is None:
            poly_config = PolynomialCoprimeConfig(k=5, degree=4)
        
        self.config = poly_config
        self.K = poly_config.k
        self.D = poly_config.degree + 1
        self.use_text = use_text
        self.use_graph = use_graph
        self.use_num = use_num
        self.use_saturation = use_saturation
        
        # Input projections
        input_dim = 0
        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            input_dim += hidden_dim
        if use_graph:
            self.graph_proj = nn.Linear(graph_dim, hidden_dim)
            input_dim += hidden_dim
        if use_num:
            self.num_proj = nn.Linear(num_dim, hidden_dim)
            input_dim += hidden_dim
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            GDPONormalization(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Per-functional structure predictors (JEPA abstract representations)
        # Projects to topological dimensions directly using True B-Splines or Linear projections
        use_bspline = (getattr(self.config, 'basis_type', '') == 'bspline') or use_kan_heads
        if use_bspline:
            from ..surrogates.kagh_networks import KANLayer
            self.coeff_heads = nn.ModuleList([
                KANLayer(hidden_dim, self.D, grid_size=5, spline_order=3) for _ in range(self.K)
            ])
        else:
            self.coeff_heads = nn.ModuleList([
                nn.Linear(hidden_dim, self.D) for _ in range(self.K)
            ])
        
        # Bimodal Routing (evolutionary genome selection)
        # SILICON SOVEREIGNTY: Replaced torch.randint with Honest Jitter derivation
        from ..core.honest_jitter import harvest_honest_jitter
        initial_genome = (harvest_honest_jitter((self.K,), scaled=False) > 0.0).long()
        self.register_buffer('bimodal_genome', initial_genome)
        
        # Learned Primitive Perturbation (Phase 6 optimization)
        # Allows adaptive quantization grid deformation
        self.primitive_perturbation = LearnedPrimitivePerturbation(dim=self.D)
    
    def embed(
        self,
        text: Optional[Union[str, torch.Tensor]] = None,
        graph: Optional[torch.Tensor] = None,
        num: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Convenience method returning the fused hidden state [batch, hidden_dim]."""
        out = self.forward(text_emb=text, graph_emb=graph, num_features=num)
        return out['fused_hidden']

    def forward(
        self,
        text_emb: Optional[Union[torch.Tensor, str]] = None,
        graph_emb: Optional[torch.Tensor] = None,
        num_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multi-modal inputs into polynomial coefficient distributions.
        
        Args:
            text_emb: [batch, text_dim] text embeddings or raw str
            graph_emb: [batch, graph_dim] graph embeddings
            num_features: [batch, num_dim] numerical features
            
        Returns:
            Dictionary containing:
                - 'residue_distributions': [batch, K, D] coefficient distributions
                - 'primitive_field': [batch, K, D] fixed-point field
                - 'coeff_logits': List of [batch, D] logits per functional
                - 'fused_hidden': [batch, hidden_dim] fused representation
        """
        if isinstance(text_emb, str):
            from ..data.canonical_projection import CanonicalProjector
            dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
            in_dim = getattr(self.text_proj, 'in_features', 768) if hasattr(self, 'text_proj') else 768
            proj = CanonicalProjector(dim=in_dim, device=str(dev))
            text_emb = proj._text_to_tensor(text_emb)

        batch_size = (
            text_emb.shape[0] if text_emb is not None else
            graph_emb.shape[0] if graph_emb is not None else
            num_features.shape[0]
        )
        
        # Project each modality
        modality_features = []
        
        ref_tensor = None
        for t in [text_emb, graph_emb, num_features]:
            if t is not None:
                ref_tensor = t
                break
                
        device = ref_tensor.device if ref_tensor is not None else torch.device("cpu")
        dtype = ref_tensor.dtype if ref_tensor is not None else torch.float32
        prefix_shape = ref_tensor.shape[:-1] if ref_tensor is not None else (batch_size,)

        if self.use_text:
            if text_emb is not None:
                if text_emb.shape[-1] != self.text_proj.in_features:
                    in_feat = self.text_proj.in_features
                    if text_emb.dim() >= 2:
                        orig_shape = text_emb.shape
                        flat_t = text_emb.reshape(-1, orig_shape[-1])
                        adapted_t = torch.nn.functional.adaptive_avg_pool1d(
                            flat_t.unsqueeze(1), in_feat
                        ).squeeze(1).reshape(*orig_shape[:-1], in_feat)
                        modality_features.append(self.text_proj(adapted_t))
                    else:
                        adapted_t = torch.nn.functional.adaptive_avg_pool1d(
                            text_emb.unsqueeze(0).unsqueeze(1), in_feat
                        ).squeeze()
                        modality_features.append(self.text_proj(adapted_t))
                else:
                    modality_features.append(self.text_proj(text_emb))
            else:
                modality_features.append(torch.zeros(*prefix_shape, self.text_proj.out_features, device=device, dtype=dtype))
        
        if self.use_graph:
            if graph_emb is not None:
                if graph_emb.shape[-1] != self.graph_proj.in_features:
                    in_feat = self.graph_proj.in_features
                    orig_shape = graph_emb.shape
                    flat_g = graph_emb.reshape(-1, orig_shape[-1])
                    adapted_g = torch.nn.functional.adaptive_avg_pool1d(
                        flat_g.unsqueeze(1), in_feat
                    ).squeeze(1).reshape(*orig_shape[:-1], in_feat)
                    modality_features.append(self.graph_proj(adapted_g))
                else:
                    modality_features.append(self.graph_proj(graph_emb))
            else:
                modality_features.append(torch.zeros(*prefix_shape, self.graph_proj.out_features, device=device, dtype=dtype))
        
        if self.use_num:
            if num_features is not None:
                if num_features.shape[-1] != self.num_proj.in_features:
                    in_feat = self.num_proj.in_features
                    orig_shape = num_features.shape
                    flat_n = num_features.reshape(-1, orig_shape[-1])
                    adapted_n = torch.nn.functional.adaptive_avg_pool1d(
                        flat_n.unsqueeze(1), in_feat
                    ).squeeze(1).reshape(*orig_shape[:-1], in_feat)
                    modality_features.append(self.num_proj(adapted_n))
                else:
                    modality_features.append(self.num_proj(num_features))
            else:
                modality_features.append(torch.zeros(*prefix_shape, self.num_proj.out_features, device=device, dtype=dtype))
        
        # Fuse modalities
        if len(modality_features) == 0:
            raise ValueError("At least one modality must be provided")
        
        fused = torch.cat(modality_features, dim=-1)
        h = self.fusion(fused)  # [batch, hidden_dim]
        
        # Compute per-functional structural embeddings (JEPA)
        coeff_logits = []
        residues = []
        
        for k, head in enumerate(self.coeff_heads):
            logits_k = head(h)  # [batch, D]
            
            # Bimodal routing (evolutionary genome selection)
            if hasattr(self, 'bimodal_genome') and self.bimodal_genome[k] == 1 and self.use_saturation and hasattr(self.config, 'saturation_gate'):
                # Hard mode (saturation)
                res_k = self.config.saturation_gate(logits_k)
            else:
                # Soft mode
                res_k = torch.tanh(logits_k)
            
            # Inverted Hypersphere Cosmology (IHC): 33-shell golden ratio ladder scaling
            # Stabilizes UV-IR seesaw across co-prime functional hierarchy without hard clipping
            # Bound directly to canonical invariants IHC 33-shell hierarchy
            ihc_ladder_scale = get_ihc_ladder_scale(k, num_shells=33)
            res_k = res_k * ihc_ladder_scale

            
            coeff_logits.append(logits_k)
            residues.append(res_k)
        
        # Stack into tensor: [batch, K, D] (where D is Dim // K)
        residue_distributions = torch.stack(residues, dim=1)
        
        # Quantize to Fixed Point Operational Primitive
        # "Floating-point arithmetic introduces nondeterminism"
        fixed_field = FixedPointField(residue_distributions)
        
        # Apply learned perturbation (on the primitive integers)
        perturbed_field = self.primitive_perturbation(fixed_field)
        
        return {
            'residue_distributions': residue_distributions, # Continuous (Legacy compatibility) or Saturated
            'primitive_field': perturbed_field,             # Operational Primitive (Invariant Optimization)
            'coeff_logits': coeff_logits,
            'fused_hidden': h
        }
    
    def compute_expected_residues(
        self,
        residue_distributions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute expected polynomial coefficients.
        
        Args:
            residue_distributions: [batch, K, D]
            
        Returns:
            expected_residues: [batch, K, D] expected coefficients
        """
        # For polynomial functionals, distributions already represent  
        # expected coefficients (softmax-weighted)
        return residue_distributions

    def adapt_online(
        self,
        support_text: Optional[torch.Tensor] = None,
        support_graph: Optional[torch.Tensor] = None,
        support_num: Optional[torch.Tensor] = None,
        steps: int = 1,
        lr: float = 0.01,
        entropy: Optional[Any] = None
    ) -> 'PolynomialFunctionalEmbedder':
        """
        Perform online inner-loop adaptation (MAML step) on multimodal dyad support data.
        Returns an adapted instance of PolynomialFunctionalEmbedder.
        """
        def clone_module(module):
            import copy
            clone = copy.copy(module)
            clone._parameters = {}
            for k, v in module._parameters.items():
                if v is not None:
                    clone._parameters[k] = nn.Parameter(v.clone(), requires_grad=v.requires_grad)
                else:
                    clone._parameters[k] = None
            clone._buffers = {k: v.clone() if v is not None else None for k, v in module._buffers.items()}
            clone._modules = {}
            for k, v in module._modules.items():
                if v is not None:
                    clone._modules[k] = clone_module(v)
                else:
                    clone._modules[k] = None
            return clone

        adapted_embedder = clone_module(self)
        
        effective_lr = lr
        if entropy is not None:
            entropy_val = entropy.mean().item() if isinstance(entropy, torch.Tensor) else float(entropy)
            effective_lr = lr * (1.0 + abs(entropy_val))
            
        optimizer = torch.optim.SGD(adapted_embedder.parameters(), lr=effective_lr)
        
        for p in adapted_embedder.parameters():
            p.requires_grad_(True)
            
        adapted_embedder.train()
        for step in range(steps):
            optimizer.zero_grad()
            out = adapted_embedder(support_text, support_graph, support_num)
            residues = out['residue_distributions']
            # Real Differentiable PAS_h loss: maximize magnitude of expected phase alignment across functional heads
            harmonics = torch.tensor([abs(math.cos((2*n - 1) * math.pi / 12) * 10) + 1 for n in range(1, 7)], device=residues.device)
            phase = residues.mean(dim=-1)
            score = torch.cos(phase.unsqueeze(-1) * harmonics).mean(dim=-1)
            pas_loss = 1.0 - score.mean()
            
            pas_loss.backward()
            optimizer.step()
            
        return adapted_embedder

    def compute_jepa_monge_ampere_energy(
        self,
        context_state: torch.Tensor,
        target_state: torch.Tensor,
        moment_transport: Optional[Any] = None,
        pas_anisotropy: Optional[torch.Tensor] = None,
        shell_depth: int = 0,
        is_honeybee_mode: bool = False
    ) -> torch.Tensor:
        """
        Computes the JEPA latent prediction energy using the dual Kantorovich-Wasserstein 
        Monge-Ampere formulation rather than unconstrained Euclidean L2 or cosine distance.
        
        Evaluates the potential through the Input Convex Neural Network (ICNN):
        E_JEPA = E_x[psi(x)] + E_y[psi^*(y)] scaled by shell depth and PAS anisotropy.
        Binds the context latent to the Cayley cubic surface (projected via PyOpenCL when available).
        Dynamically triggers Honeybee collective mode (curvature collapse) when ALGORITHMIC affordance drops.
        """
        # Affordance-gated Honeybee mode activation
        try:
            from ..core.hardware_monitor import HardwareMonitor, ComputeAffordanceType
            monitor = HardwareMonitor.get_instance()
            if not is_honeybee_mode:
                algo_aff = monitor.get_typed_affordance(ComputeAffordanceType.ALGORITHMIC)
                if not algo_aff.is_admissible or algo_aff.gauge < 0.25:
                    is_honeybee_mode = True
        except Exception:
            pass

        # Encode context and target through the embedder fusion network if not already in fused space
        hidden_dim = self.fusion[-1].out_features if hasattr(self.fusion[-1], 'out_features') else 512
        if context_state.shape[-1] == hidden_dim:
            context_h = context_state
        else:
            context_out = self.forward(text_emb=context_state)
            context_h = context_out['fused_hidden']

        if target_state.shape[-1] == hidden_dim:
            target_h = target_state
        else:
            target_out = self.forward(text_emb=target_state)
            target_h = target_out['fused_hidden']
        
        if moment_transport is None:
            from ..core.conjugate_moment_transport import ConjugateMomentTransport
            moment_transport = ConjugateMomentTransport(dim=context_h.shape[-1]).to(context_h.device)
            
        # Hard-project context state to Cayley Cubic surface V(C) = 0 if outside honeybee mode
        if not is_honeybee_mode and hasattr(moment_transport, 'project_to_cayley_surface'):
            context_h = moment_transport.project_to_cayley_surface(context_h, is_honeybee_mode=is_honeybee_mode)
            
        if pas_anisotropy is None:
            # Default to isotropic PAS unity
            pas_anisotropy = torch.ones(context_h.shape[0], 1, device=context_h.device)
            
        return moment_transport.compute_context_adaptive_monge_ampere_loss(
            source=context_h,
            target=target_h,
            pas_anisotropy=pas_anisotropy,
            shell_depth=shell_depth
        )


