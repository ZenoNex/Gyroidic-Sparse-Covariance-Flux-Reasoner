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
from typing import Dict, List, Tuple, Optional, Any
from src.core.honest_jitter import harvest_honest_jitter

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.false_negative_subsystem import VoynichExemptionToken
from src.topology.triadic_reciprocity import TriadicReciprocityChecker
from src.core.love_invariant_protector import LoveInvariantProtector
from src.core.non_ergodic_entropy import NonErgodicEntropyEstimator
from src.core.chern_simons_gasket import ChernSimonsGasket
from src.core.invariants import PhaseAlignmentInvariant, catalan_number, segner_catalan_weights, get_prime_ladder
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
        Returns structural co-primality from the polynomial coprime system.
        
        Per CODES v40, channels disperse roots to achieve Lawful Novelty rather than
        succumbing to the literal mistranslation of 'persistence pressure'.
        """
        novelty_disp = getattr(self.poly_config, 'lawful_novelty_dispersion', self.poly_config.co_primality_pressure)()
        return {
            'orthogonality': self.poly_config.orthogonality_pressure(),
            'lawful_novelty': novelty_disp,
            'coprimality': novelty_disp  # Backwards compatibility alias
        }

    @staticmethod
    def split_sub_pairs(s: str) -> List[Tuple[str, str]]:
        """
        Splits an input string into symmetric sub-pairs for intermodule tokenization
        and cross-boundary handshake.
        
        Generates two canonical prefix-suffix partitions:
        Cycle 1: (s[:k], s[k:]) for k in 1..len(s) (ending with (s, ''))
        Cycle 2: ('', s) followed by (s[:k], s[k:]) for k in 1..len(s)-1
        
        For an 18-character string like 'Isotremoptememally', this generates
        exactly 36 possible sub-pairs representing all forward and boundary cuts.
        """
        n = len(s)
        if n == 0:
            return [("", "")]
            
        pairs: List[Tuple[str, str]] = []
        for k in range(1, n + 1):
            pairs.append((s[:k], s[k:]))
        pairs.append(("", s))
        for k in range(1, n):
            pairs.append((s[:k], s[k:]))
            
        return pairs

    @staticmethod
    def split_catalan_weighted_sub_pairs(s: str) -> List[Dict[str, object]]:
        """
        Decomposes an agglutinative morpho-unit of length n into its internal
        prefix-suffix binary parse cuts k in {1, ..., n - 1}, weighted by the
        exact Segner-Catalan binary tree recurrence:
            w_k = (C_{k-1} * C_{n-k-1}) / C_{n-1}
            sum_{k=1}^{n-1} w_k = 1.0

        Where C_{n-1} = (1 / n) * binom(2(n - 1), n - 1) is the total number of full
        recursive binary parse trees (vertices of the Stasheff Associahedron K_n)
        for an n-character morpho-unit.
        """
        n = len(s)
        if n <= 1:
            return [{
                "prefix": s,
                "suffix": "",
                "split_index": n,
                "left_trees": 1,
                "right_trees": 1,
                "total_catalan_trees": 1,
                "catalan_weight": 1.0
            }]

        total_trees = catalan_number(n - 1)
        weighted_cuts: List[Dict[str, object]] = []
        for k in range(1, n):
            left_c = catalan_number(k - 1)
            right_c = catalan_number(n - k - 1)
            weight = float(left_c * right_c) / float(total_trees)
            weighted_cuts.append({
                "prefix": s[:k],
                "suffix": s[k:],
                "split_index": k,
                "left_trees": left_c,
                "right_trees": right_c,
                "total_catalan_trees": total_trees,
                "catalan_weight": weight
            })
        return weighted_cuts


    @staticmethod
    def _is_pathological_garble(s: str) -> Tuple[bool, str]:
        """
        Detects whether an agglutinative candidate suffers from pathological garble.
        
        Wordle multi-cut openers optimize vowel-to-consonant density (preventing
        vowel-starved words or consonant clumping). In Voynich compounds:
        - Detects extreme vowel starvation (< 15% vowels in chunks >= 4).
        - Detects excessive consonant clustering (>= 5 consecutive consonants).
        - Detects broken residue parity across dynamic primes.
        """
        if len(s) == 0:
            return False, "EMPTY"
        vowels = set("aeiouyAEIOUY")
        n = len(s)
        if n >= 4:
            v_count = sum(1 for c in s if c in vowels)
            if (v_count / float(n)) < 0.15:
                return True, "VOWEL_STARVATION"
            max_c = 0
            cur_c = 0
            for c in s:
                if c.isalpha() and c not in vowels:
                    cur_c += 1
                    if cur_c > max_c:
                        max_c = cur_c
                else:
                    cur_c = 0
            if max_c >= 5:
                return True, "CONSONANT_CLUSTERING"
        return False, "NOMINAL"

    @staticmethod
    def _char_sequence_to_oklab(s: str) -> Tuple[float, float, float]:
        """
        Maps a character sequence to an OKLab perceptual moment (L, a, b).
        
        Uses deterministic linear color projection followed by LMS cone response
        and OKLab non-linear transformation matrices (M1 and M2 from ConjugateMomentTransport).
        """
        if not s:
            return 0.5, 0.0, 0.0
        # Deterministic RGB from char codepoints
        r_sum = sum(((ord(c) * 17) % 256) for c in s) / (255.0 * len(s))
        g_sum = sum(((ord(c) * 31) % 256) for c in s) / (255.0 * len(s))
        b_sum = sum(((ord(c) * 53) % 256) for c in s) / (255.0 * len(s))
        # Linear RGB to LMS cone response (M1)
        l = 0.4122214708 * r_sum + 0.5363325363 * g_sum + 0.0514459929 * b_sum
        m = 0.2119034982 * r_sum + 0.6806995451 * g_sum + 0.1073969566 * b_sum
        s_cone = 0.0883024619 * r_sum + 0.2817188376 * g_sum + 0.6299787005 * b_sum
        # Cube root non-linearity
        l_nl = math.copysign(abs(l) ** (1.0 / 3.0), l)
        m_nl = math.copysign(abs(m) ** (1.0 / 3.0), m)
        s_nl = math.copysign(abs(s_cone) ** (1.0 / 3.0), s_cone)
        # LMS to OKLab (M2)
        L = 0.2104542553 * l_nl + 0.7936177850 * m_nl - 0.0040720468 * s_nl
        a = 1.9779984951 * l_nl - 2.4285922050 * m_nl + 0.4505937099 * s_nl
        b = 0.0259040371 * l_nl + 0.7827717662 * m_nl - 0.8086757660 * s_nl
        return L, a, b

    @classmethod
    def split_information_gain_sub_pairs(
        cls,
        s: str,
        max_candidates: Optional[int] = None
    ) -> List[Dict[str, object]]:
        """
        Decomposes an agglutinative morpho-unit using Wordle-inspired information-theoretic
        entropy pruning, OKLab color grouping, and orthogonal multi-cut openers.
        
        Replaces combinatorial brute-force 2n sweeps with Shannon entropy reduction:
        1. Multi-Cut Orthogonal Openers (Golden ratio, Antipodal, Triadic, Bisection)
           bracket the morpho-unit boundary landscape.
        2. OKLab Topological Color Grouping discretizes (L, a, b) moment shifts and
           consonant/vowel transitions into ternary color buckets.
        3. Dynamic Programming Entropy Evaluation prunes zero-information degenerate cuts
           and penalizes pathological garble (vowel starvation / consonant clustering).
        4. Ranked by Segner-Catalan binary tree capacity x information gain.
        """
        n = len(s)
        if n <= 1:
            return [{
                "prefix": s,
                "suffix": "",
                "split_index": n,
                "information_gain": 0.0,
                "catalan_weight": 1.0,
                "rank_score": 1.0,
                "is_opener": True,
                "is_garbled": False,
                "garble_reason": "NOMINAL",
                "oklab_delta": 0.0,
                "color_group": (0, 0, 0)
            }]

        vowels = set("aeiouyAEIOUY")
        total_catalan = catalan_number(n - 1)

        # 1. Multi-cut orthogonal openers (Wordle opener analogy: SALET/TRACE)
        k_phi = max(1, min(n - 1, int(round(n / 1.618033988749895))))
        k_phi_anti = max(1, min(n - 1, n - k_phi))
        k_tri_1 = max(1, min(n - 1, n // 3))
        k_tri_2 = max(1, min(n - 1, (2 * n) // 3))
        k_mid = max(1, min(n - 1, n // 2))
        openers_set = {k_phi, k_phi_anti, k_tri_1, k_tri_2, k_mid}

        # 2. Evaluate OKLab color moments and ternary feedback bucket for every candidate cut
        raw_cuts = []
        bucket_counts: Dict[Tuple[int, int, int], int] = {}

        for k in range(1, n):
            pfx = s[:k]
            sfx = s[k:]

            # Garble detection on both sides
            pfx_garbled, pfx_reason = cls._is_pathological_garble(pfx)
            sfx_garbled, sfx_reason = cls._is_pathological_garble(sfx)
            is_garbled = pfx_garbled or sfx_garbled
            garble_reason = pfx_reason if pfx_garbled else sfx_reason

            # OKLab color moments
            L1, a1, b1 = cls._char_sequence_to_oklab(pfx)
            L2, a2, b2 = cls._char_sequence_to_oklab(sfx)
            delta_L = abs(L1 - L2)
            delta_ab = math.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)
            oklab_delta = math.sqrt(delta_L ** 2 + delta_ab ** 2)

            # Ternary color grouping bucket (0=gray/low, 1=yellow/mid, 2=green/high)
            bin_L = 0 if delta_L < 0.08 else (1 if delta_L < 0.22 else 2)
            bin_C = 0 if delta_ab < 0.04 else (1 if delta_ab < 0.12 else 2)

            # Transition bucket: 2 for vowel-consonant seam, 1 for consonant-consonant, 0 for intra-vowel/identical
            c_left = s[k - 1]
            c_right = s[k]
            if (c_left in vowels) != (c_right in vowels):
                bin_T = 2  # Morphological seam
            elif c_left != c_right:
                bin_T = 1
            else:
                bin_T = 0

            bucket = (bin_L, bin_C, bin_T)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

            # Catalan weight
            left_c = catalan_number(k - 1)
            right_c = catalan_number(n - k - 1)
            cat_weight = float(left_c * right_c) / float(total_catalan)

            raw_cuts.append({
                "split_index": k,
                "prefix": pfx,
                "suffix": sfx,
                "catalan_weight": cat_weight,
                "bucket": bucket,
                "oklab_delta": oklab_delta,
                "is_opener": (k in openers_set),
                "is_garbled": is_garbled,
                "garble_reason": garble_reason
            })

        # 3. Information gain computation and entropy-driven pruning
        total_eval = float(len(raw_cuts))
        candidates = []
        for cut in raw_cuts:
            p_bucket = bucket_counts[cut["bucket"]] / total_eval
            info_gain = -math.log2(p_bucket + 1e-12)

            # Garble penalty
            multiplier = 0.2 if cut["is_garbled"] else 1.0
            # Openers get priority boost
            opener_boost = 1.3 if cut["is_opener"] else 1.0

            rank_score = cut["catalan_weight"] * (1.0 + info_gain) * multiplier * opener_boost

            # Prune degenerate zero-contrast cuts unless it is a canonical opener
            if not cut["is_opener"] and cut["oklab_delta"] < 0.01 and cut["is_garbled"]:
                continue

            candidates.append({
                "prefix": cut["prefix"],
                "suffix": cut["suffix"],
                "split_index": cut["split_index"],
                "information_gain": info_gain,
                "catalan_weight": cut["catalan_weight"],
                "rank_score": rank_score,
                "is_opener": cut["is_opener"],
                "is_garbled": cut["is_garbled"],
                "garble_reason": cut["garble_reason"],
                "oklab_delta": cut["oklab_delta"],
                "color_group": cut["bucket"]
            })

        # Sort candidates by rank_score descending
        candidates.sort(key=lambda c: c["rank_score"], reverse=True)

        if max_candidates is not None and max_candidates > 0:
            return candidates[:max_candidates]
        return candidates

    def find_optimal_seam(
        self,
        s: str,
        max_candidates: int = 5
    ) -> Dict[str, Any]:
        """
        Finds the optimal morpheme boundary cut for an agglutinative morpho-unit
        using information-theoretic pruning and Chebyshev polynomial coprime functional projections.
        
        Evaluates the top information-theoretic candidate cuts from split_information_gain_sub_pairs,
        computes their Phase Alignment Score (PAS_s), and seals with the Chern-Simons Gasket
        when PAS_s > 0.95.
        """
        candidates = self.split_information_gain_sub_pairs(s, max_candidates=max_candidates)
        if not candidates:
            candidates = [{
                "prefix": s,
                "suffix": "",
                "split_index": len(s),
                "information_gain": 0.0,
                "catalan_weight": 1.0,
                "rank_score": 1.0,
                "is_opener": True,
                "is_garbled": False,
                "garble_reason": "NOMINAL",
                "oklab_delta": 0.0,
                "color_group": (0, 0, 0)
            }]

        best_cut = candidates[0]
        best_pas = -1.0

        device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")

        for cand in candidates:
            pfx = cand["prefix"]
            sfx = cand["suffix"]

            # Grounded character projection into latent space
            pfx_vals = [ord(c) for c in pfx] or [0]
            sfx_vals = [ord(c) for c in sfx] or [0]

            thought = torch.zeros(1, self.thought_proj.in_features, device=device)
            p_len = len(pfx_vals)
            s_len = len(sfx_vals)
            for idx in range(self.thought_proj.in_features):
                v1 = pfx_vals[idx % p_len] / 255.0
                v2 = sfx_vals[idx % s_len] / 255.0
                thought[0, idx] = math.sin((idx + 1) * v1) * math.cos((idx + 2) * v2)

            with torch.no_grad():
                proj = self.thought_proj(thought)
                residues = self._evaluate_polynomial_residues(proj)
                pas_score = float(self.pas_calculator(residues).item())

            cand["pas_score"] = pas_score

            if pas_score > best_pas:
                best_pas = pas_score
                best_cut = cand

        is_honest = best_pas > 0.95
        token = VoynichExemptionToken(
            honesty_score=best_pas,
            is_valid_exemption=is_honest and not best_cut["is_garbled"],
            is_nutrient=not best_cut["is_garbled"],
            fossilized_state=None,
            reason="Topological Refusal (Garble)" if best_cut["is_garbled"] else ("Optimal Seam PAS Lock" if is_honest else "Low Coherence Partition")
        )

        if is_honest:
            if torch.allclose(self.chern_simons.gauge_field, torch.zeros_like(self.chern_simons.gauge_field)):
                polynomial_coeffs = self.poly_config.get_coefficients_tensor()
                winding_numbers = torch.arange(1, polynomial_coeffs.shape[0] + 1, device=device)
                self.chern_simons.initialize_gauge_field(polynomial_coeffs, winding_numbers)
            F = self.chern_simons.compute_field_strength()
            kappa = torch.norm(F, p='fro')
            token = self.chern_simons.sign_exemption_token(token, kappa)

        return {
            "best_cut": (best_cut["prefix"], best_cut["suffix"]),
            "best_split_index": best_cut["split_index"],
            "phase_alignment_score": best_pas,
            "is_honest": is_honest,
            "exemption_token": token,
            "information_gain": best_cut["information_gain"],
            "oklab_delta": best_cut["oklab_delta"],
            "catalan_weight": best_cut["catalan_weight"],
            "candidate_cuts": candidates
        }
