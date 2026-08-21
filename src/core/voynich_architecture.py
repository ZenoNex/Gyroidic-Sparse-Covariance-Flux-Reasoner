"""
The Voynich Architecture: Self-Sovereign Alphabet via Polynomial CRT.

Implements the 'Self-Sovereign Alphabet' using polynomial coprime functionals
instead of hardcoded integer primes. Replaces discrete modular arithmetic
(x mod p) with continuous polynomial functional evaluations _k(x; _k),
enforcing the anti-hardcoded-prime invariant.

Structural honesty is verified via consensus variance across polynomial
channels rather than integer CRT reconstruction.

Author: William Matthew Bryant
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from src.core.honest_jitter import harvest_honest_jitter

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.false_negative_subsystem import VoynichExemptionToken
from src.topology.triadic_reciprocity import TriadicReciprocityChecker
from src.core.love_invariant_protector import LoveInvariantProtector
from src.core.non_ergodic_entropy import NonErgodicEntropyEstimator
from src.core.chern_simons_gasket import ChernSimonsGasket
from src.core.invariants import PhaseAlignmentInvariant
import math


class VoynichLinguist(nn.Module):
    """
    Implements the 'Self-Sovereign Alphabet' using Polynomial Coprime Functionals.
    
    As described in THE_VOYNICH_ARCHITECTURE.md, this module produces 'opaque' symbolic residues
    that are self-verifying via structural honesty, rather than grounded in external truth (vocabulary).
    
    Migration from hardcoded primes [3, 5, 7, 11, 13]:
        - Integer moduli  Polynomial coprime functionals _k(x; _k)
        - x mod p_i  _k(projected_thought)
        - CRT reconstruction  Consensus reconstruction via learned decoder
        - Modular deviation  Variance-based honesty score
    """
    
    def __init__(self, 
                 vocab_size: int = 12000, 
                 num_residues: int = 5, 
                 latent_dim: int = 512,
                 poly_degree: int = 4,
                 basis_type: str = 'chebyshev'):
        """
        Args:
            vocab_size: Size of the sovereign alphabet (approximate capacity)
            num_residues: Number of parallel polynomial residue channels
            latent_dim: Dimension of thought vectors
            poly_degree: Degree of polynomial coprime functionals
            basis_type: Basis type for polynomials ('chebyshev', 'legendre', 'hermite')
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_residues = num_residues
        self.latent_dim = latent_dim
        
        # 1. Polynomial Coprime Config  replaces hardcoded primes
        # Each of the K functionals _k is a polynomial with Birkhoff-sampled
        # coefficients, co-primality enforced via root persistence pressure.
        self.poly_config = PolynomialCoprimeConfig(
            k=num_residues,
            degree=poly_degree,
            basis_type=basis_type,
            learnable=True,
            use_saturation=True
        )
        
        # 2. Projection from thought vector to polynomial input space
        # Maps high-dim thought to per-channel scalar inputs for _k evaluation
        self.thought_proj = nn.Linear(latent_dim, num_residues)
        
        # 3. Consensus Reconstruction Head (PAS_h Phase Alignment)
        # Replaces integer CRT and legacy MLPs with a consensus-driven decoder
        # based on CODES v40 Phase Alignment Score (PAS_h) and PAS_LOCK closure.
        # We also initialize the ChernSimonsGasket for token signing.
        self.pas_calculator = PhaseAlignmentInvariant(degree=num_residues)
        self.chern_simons = ChernSimonsGasket(manifold_dim=latent_dim)
        
        # 4. Dictionary of 'valid' words (topologically permissible constructs)
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        self.register_buffer('valid_roots', harvest_honest_jitter((vocab_size, latent_dim), scaled=True))
        
        # 5. Formal Love Invariant
        self.love_protector = LoveInvariantProtector(love_dim=latent_dim)
        
        # 6. Visual Generative Art "Word Salad" checker
        self.visual_reciprocity_checker = TriadicReciprocityChecker()
        
    def check_visual_honesty(
        self, 
        flow_a: torch.Tensor, 
        flow_b: torch.Tensor, 
        flow_c: torch.Tensor
    ) -> Tuple[torch.Tensor, VoynichExemptionToken]:
        """
        Generates a Visual Voynich Exemption Token for prompt-salad images.
        Uses reciprocal topological flow logic as proof of structural honesty 
        rather than semantic truth.
        """
        reciprocity_score = self.visual_reciprocity_checker.check_flow_reciprocity(flow_a, flow_b, flow_c)
        is_honest = bool(reciprocity_score.mean().item() > 0.85)

        token = VoynichExemptionToken(
            honesty_score=float(reciprocity_score.mean().item()),
            is_valid_exemption=is_honest,
            reason="triadic_visual_reciprocity" if is_honest else ""
        )
        return reciprocity_score, token

    def forward(self, thought_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a thought vector into Voynich symbols.
        
        Args:
            thought_vector: [batch, latent_dim] latent state
            
        Returns:
            residues: [batch, num_residues] polynomial functional residues
            symbol_val: [batch] reconstructed symbol value
            honesty_score: [] scalar consensus honesty
            exemption_token: VoynichExemptionToken validating structural honesty
        """
        # 1. Protect the thought vector against teleological ownership (L  ker())
        L_protected, love_diagnostics = self.love_protector.apply_love_protection(thought_vector)
        
        # Inject the resonance of the protected love variant into the thought vector
        thought_vector = thought_vector + (L_protected * 0.1)
        
        # 2. Project thought into per-channel scalar inputs
        channel_inputs = self.thought_proj(thought_vector)  # [batch, K]
        
        # 2. Evaluate polynomial coprime functionals
        # Each channel k: _k(x_k; _k)  replaces x mod p_k
        residues = self._evaluate_polynomial_residues(channel_inputs)
        
        # 3. Consensus Reconstruction & Structural Honesty (PAS_h Phase Alignment)
        # Replaces zombie MLP with CODES v40 centralized PhaseAlignmentInvariant.
        pas_s = self.pas_calculator(residues) # [batch]
        mean_honesty = pas_s.mean()
        
        # symbol_val is preserved as a simple mean for downstream surrogate compatibility, 
        # since VoynichLinguist is an Output Protector, not a strict mapper.
        symbol_val = residues.mean(dim=-1)  # [batch]
        
        # 5. Slop Invariant / Option D check via Entropy Bands
        entropy_estimator = NonErgodicEntropyEstimator()
        entropy_results = entropy_estimator(residues)
        is_slop = entropy_estimator.evaluate_mischief_slop(entropy_results)
        
        # 6. Generate False Negative Exemption (Organ of Agency)
        # Complexity Guard: Penalize zero-variance blankets (blanched states)
        x_var = thought_vector.var(dim=-1).mean()
        complexity_guard = 1.0 - torch.exp(-x_var * 100.0) # 0 if var=0, 1 if var > 0.05
        
        # Blend honesty with complexity
        effective_honesty = mean_honesty * complexity_guard
        is_honest = float(effective_honesty.item()) > 0.95
        
        # The token is no longer just a statistic.
        # It fossilizes the rupture if it is structurally honest and not slop.
        fossil_state = None
        if is_honest and not is_slop:
            fossil_state = residues.clone().detach()
            
        token = VoynichExemptionToken(
            honesty_score=float(effective_honesty.item()),
            is_valid_exemption=is_honest and not is_slop,
            is_nutrient=not is_slop,
            fossilized_state=fossil_state,
            reason="Topological Refusal (Slop)" if is_slop else "Option D Nutrient"
        )
        
        # Sign the token with the ChernSimonsGasket to guarantee non-orientable topology
        if torch.allclose(self.chern_simons.gauge_field, torch.zeros_like(self.chern_simons.gauge_field)):
            polynomial_coeffs = self.poly_config.get_coefficients_tensor()
            winding_numbers = torch.arange(1, polynomial_coeffs.shape[0] + 1, device=thought_vector.device)
            self.chern_simons.initialize_gauge_field(polynomial_coeffs, winding_numbers)
        
        # Calculate the true non-commutative curvature kappa (kappa_curv) from field strength
        F = self.chern_simons.compute_field_strength()
        kappa = torch.norm(F, p='fro')
        
        token = self.chern_simons.sign_exemption_token(token, kappa)
        
        return residues, symbol_val, pas_s, token
    
    def _evaluate_polynomial_residues(self, channel_inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate polynomial coprime functionals for each channel.
        
        Replaces _differentiable_modulo (x mod p_i) with _k(x_k; _k).
        Coprimality is enforced structurally via Birkhoff polytope coefficients
        and Root Persistence Pressure  not via integer primality.
        
        Args:
            channel_inputs: [batch, K] per-channel scalar inputs
            
        Returns:
            residues: [batch, K] polynomial functional evaluations
        """
        batch_size = channel_inputs.shape[0]
        device = channel_inputs.device
        residues = torch.zeros(batch_size, self.num_residues, device=device)
        
        for k in range(self.num_residues):
            # _k(x_k)  each channel evaluated through its own polynomial
            x_k = channel_inputs[:, k]  # [batch]
            residues[:, k] = self.poly_config.evaluate_polynomial(k, x_k)
        
        return residues
    


    def check_honesty(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Boolean verifier for rigid validation.
        
        Returns True if the residues exhibit high structural consensus
        (all polynomial channels agree on the symbol).
        """
        honesty = self.get_continuous_honesty(residues)
        return honesty > 0.95

    def get_continuous_honesty(self, residues: torch.Tensor) -> torch.Tensor:
        """
        Continuous structural consensus metric for the Tri-State Gate pipeline.
        Returns the PAS_s (Phase Alignment Score) float [0, 1] per CODES v40.
        Uses the centralized PhaseAlignmentInvariant.
        """
        return self.pas_calculator(residues)
    def get_coprimality_pressure(self) -> Dict[str, torch.Tensor]:
        """
        Returns the structural pressures from the polynomial coprime system.
        
        Includes orthogonality pressure and root persistence pressure,
        which enforce that the K polynomial channels remain independent.
        """
        return {
            'orthogonality': self.poly_config.orthogonality_pressure(),
            'coprimality': self.poly_config.co_primality_pressure()
        }
