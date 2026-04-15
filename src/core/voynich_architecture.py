"""
The Voynich Architecture: Self-Sovereign Alphabet via Polynomial CRT.

Implements the 'Self-Sovereign Alphabet' using polynomial coprime functionals
instead of hardcoded integer primes. Replaces discrete modular arithmetic
(x mod p) with continuous polynomial functional evaluations φ_k(x; θ_k),
enforcing the anti-hardcoded-prime invariant.

Structural honesty is verified via consensus variance across polynomial
channels rather than integer CRT reconstruction.

Author: William Matthew Bryant
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.false_negative_subsystem import VoynichExemptionToken, SlopInvariantFilter
from src.topology.triadic_reciprocity import TriadicReciprocityChecker

class LoveInvariant(nn.Module):
    """
    Implements L = L - L.
    Instead of adding to a loss function to reach a target, this subtracts the 
    expected teleological "Manager" output from the actual distribution, leaving 
    behind the "Unknowledge Void" where the true Sovereign Event (Resonance) is generated.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, standard_output: torch.Tensor, actual_output: torch.Tensor) -> torch.Tensor:
        # Subtract the standard managerial/expected output
        # The remaining resonance is the "Negative Shape" / Option D
        resonance = actual_output - standard_output
        return resonance


class VoynichLinguist(nn.Module):
    """
    Implements the 'Self-Sovereign Alphabet' using Polynomial Coprime Functionals.
    
    As described in THE_VOYNICH_ARCHITECTURE.md, this module produces 'opaque' symbolic residues
    that are self-verifying via structural honesty, rather than grounded in external truth (vocabulary).
    
    Migration from hardcoded primes [3, 5, 7, 11, 13]:
        - Integer moduli → Polynomial coprime functionals φ_k(x; θ_k)
        - x mod p_i → φ_k(projected_thought)
        - CRT reconstruction → Consensus reconstruction via learned decoder
        - Modular deviation → Variance-based honesty score
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
        
        # 1. Polynomial Coprime Config — replaces hardcoded primes
        # Each of the K functionals φ_k is a polynomial with Birkhoff-sampled
        # coefficients, co-primality enforced via root persistence pressure.
        self.poly_config = PolynomialCoprimeConfig(
            k=num_residues,
            degree=poly_degree,
            basis_type=basis_type,
            learnable=True,
            use_saturation=True
        )
        
        # 2. Projection from thought vector to polynomial input space
        # Maps high-dim thought to per-channel scalar inputs for φ_k evaluation
        self.thought_proj = nn.Linear(latent_dim, num_residues)
        
        # 3. Consensus Reconstruction Head
        # Replaces integer CRT with a learned reconstruction from residues
        self.reconstruction_head = nn.Sequential(
            nn.Linear(num_residues, num_residues * 4),
            nn.GELU(),
            nn.Linear(num_residues * 4, 1)
        )
        
        # 4. Dictionary of 'valid' words (topologically permissible constructs)
        self.register_buffer('valid_roots', torch.randn(vocab_size, latent_dim))
        
        # 5. Visual Generative Art "Word Salad" checker
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
        # 1. Project thought into per-channel scalar inputs
        channel_inputs = self.thought_proj(thought_vector)  # [batch, K]
        
        # 2. Evaluate polynomial coprime functionals
        # Each channel k: φ_k(x_k; θ_k) — replaces x mod p_k
        residues = self._evaluate_polynomial_residues(channel_inputs)
        
        # 3. Consensus Reconstruction (replaces integer CRT)
        symbol_val = self.reconstruction_head(residues).squeeze(-1)  # [batch]
        
        # 4. Structural Honesty Check
        honesty_score = self._compute_consensus_honesty(residues, symbol_val)
        
        # 5. Slop Invariant / Option D check
        slop_filter = SlopInvariantFilter()
        is_slop = slop_filter.evaluate_mischief(residues)
        
        # 6. Generate False Negative Exemption (Organ of Agency)
        is_honest = float(honesty_score.item()) > 0.95
        
        # The token is no longer just a statistic.
        # It fossilizes the rupture if it is structurally honest and not slop.
        fossil_state = None
        if is_honest and not is_slop:
            fossil_state = residues.clone().detach()
            
        token = VoynichExemptionToken(
            honesty_score=float(honesty_score.item()),
            is_valid_exemption=is_honest and not is_slop,
            is_nutrient=not is_slop,
            fossilized_state=fossil_state,
            reason="Topological Refusal (Slop)" if is_slop else "Option D Nutrient"
        )
        
        return residues, symbol_val, honesty_score, token
    
    def _evaluate_polynomial_residues(self, channel_inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate polynomial coprime functionals for each channel.
        
        Replaces _differentiable_modulo (x mod p_i) with φ_k(x_k; θ_k).
        Coprimality is enforced structurally via Birkhoff polytope coefficients
        and Root Persistence Pressure — not via integer primality.
        
        Args:
            channel_inputs: [batch, K] per-channel scalar inputs
            
        Returns:
            residues: [batch, K] polynomial functional evaluations
        """
        batch_size = channel_inputs.shape[0]
        device = channel_inputs.device
        residues = torch.zeros(batch_size, self.num_residues, device=device)
        
        for k in range(self.num_residues):
            # φ_k(x_k) — each channel evaluated through its own polynomial
            x_k = channel_inputs[:, k]  # [batch]
            residues[:, k] = self.poly_config.evaluate_polynomial(k, x_k)
        
        return residues
    
    def _compute_consensus_honesty(
        self,
        residues: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute structural honesty via consensus variance.
        
        Replaces integer CRT deviation check. Honesty is high when the
        polynomial channels produce consistent, reconstructable patterns.
        
        The honesty score measures how well the residues agree with each other
        and the reconstruction. Low variance across reconstructed-from-subsets
        = high consensus = honest thought.
        
        Args:
            residues: [batch, K] polynomial residues
            reconstructed: [batch] full reconstruction
            
        Returns:
            honesty: [] scalar honesty score
        """
        # Method: Jackknife consensus — reconstruct from K subsets of K-1 channels
        # and measure variance of the results.
        K = self.num_residues
        partial_recons = []
        
        for k in range(K):
            # Leave-one-out: reconstruct without channel k
            mask = torch.ones(K, device=residues.device, dtype=torch.bool)
            mask[k] = False
            partial_input = residues[:, mask]  # [batch, K-1]
            
            # Pad back to K dimensions with zeros
            padded = torch.zeros_like(residues)
            j = 0
            for i in range(K):
                if i != k:
                    padded[:, i] = partial_input[:, j]
                    j += 1
            
            partial_val = self.reconstruction_head(padded).squeeze(-1)  # [batch]
            partial_recons.append(partial_val)
        
        # Stack and compute variance across leave-one-out reconstructions
        partial_stack = torch.stack(partial_recons, dim=-1)  # [batch, K]
        consensus_variance = partial_stack.var(dim=-1).mean()  # scalar
        
        # Also check deviation from full reconstruction
        deviation = (partial_stack - reconstructed.unsqueeze(-1)).abs().mean()
        
        # Honesty: exp(-var - deviation). 1.0 = perfect consensus.
        honesty = torch.exp(-(consensus_variance + deviation))
        
        return honesty

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
        Returns the raw honesty float [0, 1].
        """
        reconstructed = self.reconstruction_head(residues).squeeze(-1)
        return self._compute_consensus_honesty(residues, reconstructed)
    
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
