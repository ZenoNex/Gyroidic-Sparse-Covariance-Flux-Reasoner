"""
Sparse covariance probes with gyroid-inspired violation detection.

Computes local spectral signatures to detect topology violations
without expensive global persistent homology.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import math
from src.core.honest_jitter import harvest_honest_jitter, honest_multinomial
from src.core.false_negative_subsystem import VoynichExemptionToken



class SparseGyroidCovarianceProbe(nn.Module):
    """
    Sparse covariance-based pressure evaluator.
    
    Maintains local k-hop covariance sketches and detects spectral anomalies
    that indicate broken gyroid-like connectivity patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        window_size: int = 32,
        k_hop: int = 2,
        num_eigenvalues: int = 8,
        violation_threshold: float = 0.5,
        use_saturation_detection: bool = True,
        adaptive_threshold: bool = True,
        percentile: float = 95.0
    ):
        """
        Initialize the Sparse Gyroid Covariance Probe.
        
        Args:
            hidden_dim: Dimension of the hidden state vectors.
            window_size: Size of the sliding window for local covariance.
            k_hop: Neighborhood hop distance for graph connectivity.
            num_eigenvalues: Number of top eigenvalues to analyze.
            violation_threshold: Fixed threshold for violation detection.
            use_saturation_detection: Enable the Saturation Fracture Detector.
            adaptive_threshold: Use percentile-based scaling for thresholds.
            percentile: Target percentile for the adaptive threshold.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.k_hop = k_hop
        self.num_eigenvalues = num_eigenvalues
        self.violation_threshold = violation_threshold
        self.use_saturation_detection = use_saturation_detection
        self.adaptive_threshold = adaptive_threshold
        self.percentile = percentile
        
        # Empirical Scaling Law (Bostick, 2025)
        # epsilon_drift varies as V^(-1/2)
        # We scale the base threshold by the inverse root of dimension
        if adaptive_threshold:
            # We treat hidden_dim as effective Volume V
            # Base epsilon is roughly 0.5 at dim=1? Or just a scaling factor.
            # Let's preserve the user's 'violation_threshold' as the coefficient epsilon_0
            # epsilon_drift = epsilon_0 * (V / V_0)^(-1/2) 
            # We assume V_0 = 1 for normalization, or just apply raw scaling.
            # To avoid crushing it too small, we use a reference dim of 64.
            scaling_factor = (hidden_dim / 64.0) ** -0.5
            self.scaled_threshold = violation_threshold * scaling_factor
        else:
            self.scaled_threshold = violation_threshold
        
        if use_saturation_detection:
            self.fracture_detector = SaturationFractureDetector()
            
        # Global Manifold Estimator (System 2 Driver)
        self.gyroid_cov = GyroidCovarianceEstimator(dim=hidden_dim)
    
    def compute_local_covariance(
        self,
        hidden_states: torch.Tensor,
        start_idx: int
    ) -> torch.Tensor:
        """
        Compute local windowed covariance matrix.
        
        Extracts a temporal window of hidden states and computes the 
        local Gram matrix to identify the spectral structure of the 
        manifold at that position.
        
        Args:
            hidden_states: The full sequence of hidden states [seq_len, hidden_dim].
            start_idx: The starting position for the local window.
            
        Returns:
            C_loc: The local covariance matrix [window_size, window_size].
        """
        seq_len = hidden_states.shape[0]
        end_idx = min(start_idx + self.window_size, seq_len)
        actual_window = end_idx - start_idx
        
        # Extract local window
        window = hidden_states[start_idx:end_idx]  # [actual_window, hidden_dim]
        
        # Compute covariance (or Gram matrix)
        # C = X X^T where X is normalized
        window_normalized = window - window.mean(dim=0, keepdim=True)
        window_normalized = window_normalized / (torch.norm(window_normalized, dim=1, keepdim=True) + 1e-8)
        
        C_loc = torch.mm(window_normalized, window_normalized.t())  # [actual_window, actual_window]
        
        # Pad if necessary
        if actual_window < self.window_size:
            C_loc_padded = torch.zeros(
                self.window_size, self.window_size,
                device=C_loc.device, dtype=C_loc.dtype
            )
            C_loc_padded[:actual_window, :actual_window] = C_loc
            C_loc = C_loc_padded
        
        return C_loc

    def compute_gcve(
        self,
        C_loc: torch.Tensor,
        h_mischief: float,
        tau_decay: float = 10.0,
        lambda_min_epsilon: float = 1e-6
    ) -> float:
        """
        Compute Gyroidic Covariance Violation Energy (GCVE).
        (legacy) V_m = V + H_mischief/tau - lambda_min/tr(C)
        (current) V_m = (V + Flatness_Penalty) * (1 - H_mischief / tau)
        The GCVE (V_m) measures the deviation from minimal-surface 
        expectations, weighted by the 'Mischief' entropy to allow for 
        admissible playful violations.
        
        Args:
            C_loc: The local covariance matrix [window, window].
            h_mischief: The current mischief entropy (H_m).
            tau_decay: Decay constant for structural pressure.
            lambda_min_epsilon: Small constant for numerical stability.
        
        Returns:
            V_m: The GCVE score. High scores indicate topological fracture.
            
        CODES v40 Invariant: 
            Non-Teleological Repair: 10.3. GCVE provides the 'containment 
            pressure' signal without requiring a target state.
        """
        # Eigenvalues for V calculation
        # Note: eigh is for symmetric matrices (covariance is symmetric)
        try:
            eigs = torch.linalg.eigvalsh(C_loc)
        except RuntimeError:
            # Fallback for numerical instability
            return 0.0
            
        if len(eigs) == 0:
            return 0.0
            
        lambda_min = eigs[0].item()
        trace_C = eigs.sum().item()
        
        # Standard violation (V) - approximation based on spectral gap or just max eig?
        # Using max eigenvalue relative to trace (spectral dominance)
        V = eigs[-1].item() / (trace_C + 1e-8)
        
        # Inversion of flatness (penalize uniform distributions)
        flatness_penalty = lambda_min / (trace_C + lambda_min_epsilon)
        
        # GCVE formula
        V_m = V + (h_mischief / tau_decay) - flatness_penalty
        
        return V_m
    
    def compute_spectral_signature(
        self,
        C_loc: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spectral properties of local covariance.
        
        Extracts top eigenvalues, spectral gaps, and condition numbers to 
        form a 'Spectral Signature' of the manifold.
        
        Args:
            C_loc: Local covariance matrix.
            
        Returns:
            A dictionary containing:
            - eigenvalues: Top tracked eigenvalues.
            - spectral_gap: Difference between top eigenvalues.
            - decay_rate: Average rate of eigenvalue attenuation.
            - trace: Total variance.
            - condition_number: Ratio of max to min eigenvalues.
        """
        # Compute eigenvalues (top k + 1 for gap computation)
        try:
            eigenvalues, _ = torch.linalg.eigh(C_loc)
            eigenvalues = eigenvalues.flip(0)  # Descending order
            eigenvalues = eigenvalues[:self.num_eigenvalues + 1]
        except:
            # Fallback if eigendecomposition fails
            eigenvalues = torch.ones(self.num_eigenvalues + 1, device=C_loc.device)
        
        # Compute metrics
        top_k = eigenvalues[:self.num_eigenvalues]
        
        if len(eigenvalues) > self.num_eigenvalues:
            spectral_gap = eigenvalues[self.num_eigenvalues - 1] - eigenvalues[self.num_eigenvalues]
        else:
            spectral_gap = torch.tensor(0.0, device=C_loc.device)
        
        decay_rate = (eigenvalues[0] - eigenvalues[-1]) / len(eigenvalues)
        trace = torch.trace(C_loc)
        
        lambda_min = eigenvalues[-1] + 1e-8
        lambda_max = eigenvalues[0] + 1e-8
        condition_number = lambda_max / lambda_min
        
        return {
            'eigenvalues': top_k,
            'spectral_gap': spectral_gap,
            'decay_rate': decay_rate,
            'trace': trace,
            'condition_number': condition_number,
            'lambda_min': lambda_min
        }
    
    def compute_pressure_score(
        self,
        spectral_signature: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the Gyroid Pressure Metric.
        
        Calculates the combined pressure score derived from the spectral gap 
        and geometric flatness of the local manifold.
        
        Args:
            spectral_signature: The signature generated by 
                                `compute_spectral_signature`.
            
        Returns:
            pressure_score: Scalar score indicating structural tension.
        """
        gap = spectral_signature['spectral_gap']
        decay = spectral_signature['decay_rate'] + 1e-8
        lambda_min = spectral_signature['lambda_min']
        trace = spectral_signature['trace'] + 1e-8
        
        # 1. Spectral Gap / Decay Rate (Topology Check)
        # Large gap relative to decay -> disconnected or blocky structure
        topo_term = torch.clamp(gap / decay, min=0.0)
        
        # 2. Minimum Eigenvalue / Trace (Geometry Check)
        # Measures effective rank stability / negative curvature proxy
        # Small values -> degenerate, flat; Large -> healthy hyperbolic
        geo_term = lambda_min / trace
        
        # Combined score
        return topo_term + geo_term
        
    def violation_fn(self, phi_eval: torch.Tensor) -> torch.Tensor:
        """
        Compute violation score from functional evaluation.
        
        Measures the deviation from the minimal surface constraint (G(x) = 0).
        
        Args:
            phi_eval: [batch, K] evaluations of the co-prime functionals.
            
        Returns:
            violation: [batch] scalar violation scores.
        """
        # Minimal surface constraint: G(x) should be 0.
        # Deviation from 0 indicates topological violation.
        # We use mean absolute deviation across functionals.
        if phi_eval.dim() > 1:
            return torch.abs(phi_eval).mean(dim=-1)
        else:
            return torch.abs(phi_eval)

    def forward(
        self, 
        h: torch.Tensor, 
        phi_fn: Optional[torch.nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Orchestrate violation detection.
        
        Args:
            h: [batch, seq_len, hidden_dim] hidden states (or 4D log-polar)
            phi_fn: Optional symbolic functional for fracture detection
            
        Returns:
            Results dictionary containing violations and scores
        """
        if len(h.shape) == 4:
            # Topologically Aware Dimensional Windowing (ACW) 
            # Prevents flat 'lobotomizing' of Log-Polar mappings.
            # Using spectral windowing to preserve phase boundary constraints & Phase transition dynamics.
            b, c, r, t = h.shape
            
            # Apply 2D FFT to enter spectral domain
            h_freq = torch.fft.rfft2(h)
            
            # Asymptotic Windowing W(f): attenuate high-frequency hallucination modes
            # This implicitly tracks the phase boundary transition ridge 
            mask = torch.ones_like(h_freq)
            mask[:, :, mask.size(2)//2:, mask.size(3)//2:] = 0.05 # Soft fractional attenuation, not total
            
            # Restore to spatial domain with geometric stress removed
            h_windowed = torch.fft.irfft2(h_freq * mask, s=(r, t))
            
            # Compress sequence while preserving spatial continuum (Volume Weighting mapping)
            h = h_windowed.reshape(b, c, r * t).transpose(1, 2)
            
        batch_size, seq_len, _ = h.shape
        violations = []
        gcve_pressures = []
        lambda_mins = []
        traces = []
        
        # 1. Compute GCVE per batch element (Geometric/Spectral)
        for b in range(batch_size):
             # For simplicity, we sample the middle window or multiple windows
             # Real implementation would scan across seq_len
             C_loc = self.compute_local_covariance(h[b], start_idx=max(0, seq_len//2 - 16))
             sig = self.compute_spectral_signature(C_loc)
             score = self.compute_pressure_score(sig)
             gcve_pressures.append(score)
             lambda_mins.append(sig['lambda_min'])
             traces.append(sig['trace'])
             violations.append(score > self.violation_threshold)
             
        gcve_pressures = torch.stack(gcve_pressures) # [batch]
        lambda_mins = torch.stack(lambda_mins)
        traces = torch.stack(traces)
        violations = torch.stack(violations).float()     # [batch]
        
        # 2. Compute Saturation Fracture (Input Sensitivity)
        fracture_scores = torch.zeros_like(gcve_pressures)
        if self.use_saturation_detection and phi_fn is not None:
             fracture_scores = self.fracture_detector(phi_fn, h)
             
        # Combined pressure score
        total_pressure = gcve_pressures + fracture_scores
        
        return {
            'gcve_scores': gcve_pressures,
            'fracture_scores': fracture_scores,
            'total_pressure': total_pressure,
            'lambda_min': lambda_mins,
            'trace_c': traces
        }

    def compute_interference_matrix(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise interference between batch elements.
        
        Measures how much the spectral signatures of different elements 
        overlap, indicating whether they are 'touching' the same 
        topological artifacts.
        
        Note that the eigenvalues themselves are non-local artifacts of the
        manifold geometry; if two different temporal sequences produce the 
        same eigenvalues, they are momentarily occupying the same 'hole' in 
        the gyroid's potential landscape.
        (legacyEquation: [partial_t Phi_i \circ \Phi_j]_{i \neq j} > Threshold)
        
        (current)  Interference = || \Phi_i(h) - \Phi_j(h) ||_2^2
        (where Phi_i \neq Phi_j, and the norm is computed over the sequence length)

        Args:
            h: The hidden states [batch, seq_len, hidden_dim].
            
        Returns:
            inter_matrix: [batch, batch] pairwise interference scores.
        """
        batch_size = h.shape[0]
        # We use a condensed representation: the spectral signature of each element
        signatures = []
        for b in range(batch_size):
            C_loc = self.compute_local_covariance(h[b], start_idx=max(0, h.shape[1]//2 - 16))
            sig = self.compute_spectral_signature(C_loc)
            # Flatten top eigenvalues as the 'violation fingerprint'
            signatures.append(sig['eigenvalues'])
        
        signatures = torch.stack(signatures) # [batch, num_eigenvalues]
        
        # Pairwise interference = cosine similarity of violation fingerprints
        # High similarity means batch elements are 'touching' the same manifold artifacts.
        signatures_norm = signatures / (torch.norm(signatures, dim=1, keepdim=True) + 1e-8)
        inter_matrix = torch.mm(signatures_norm, signatures_norm.t())
        
        return inter_matrix
    
    def scout_violations(
        self,
        hidden_states: torch.Tensor,
        return_indices: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Hunt for VIOLATIONS, not smoothness.
        
        Pointer #8: Semantics appear where covariance breaks minimal-surface 
        expectations. This method identifies locations in the sequence where 
        the manifold deviates significantly from its harmonic baseline.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim].
            return_indices: If True, return the sparse indices of the violations.
            
        Returns:
            Dict with:
            - 'sparse_deviation_mask': [batch, num_windows] boolean
            - 'deviation_magnitudes': [batch, num_windows] float
            - 'violation_indices': sparse indices (if return_indices)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_windows = max(1, (seq_len - self.window_size) // (self.window_size // 2) + 1)
        
        all_deviations = []
        all_expectations = []
        
        for b in range(batch_size):
            h = hidden_states[b]  # [seq_len, hidden_dim]
            
            window_deviations = []
            window_expectations = []
            
            for i in range(num_windows):
                start = i * (self.window_size // 2)
                
                # Compute local covariance
                C_loc = self.compute_local_covariance(h, start)
                
                # Compute spectral signature
                spec = self.compute_spectral_signature(C_loc)
                
                # GYROID EXPECTATION: For minimal surface, eigenvalue decay should be smooth
                # Expected decay: _i  _1 * exp(-i/) for some time constant 
                eigenvalues = spec['eigenvalues']
                num_eigs = len(eigenvalues)
                expected_decay = eigenvalues[0] * torch.exp(
                    -torch.arange(num_eigs, device=eigenvalues.device).float() / 3.0
                )
                
                # DEVIATION: Where does local covariance break this expectation?
                deviation = torch.abs(eigenvalues - expected_decay).sum()
                
                window_deviations.append(deviation)
                window_expectations.append(expected_decay.sum())
            
            all_deviations.append(torch.stack(window_deviations))
            all_expectations.append(torch.stack(window_expectations))
        
        deviations = torch.stack(all_deviations)  # [batch, num_windows]
        expectations = torch.stack(all_expectations)
        
        # Sparse: Only HIGH deviations matter (threshold at percentile OR scaled physical limit)
        if self.adaptive_threshold:
            # Dual check: Must exceed statistical percentile AND physical scaling limit
            stat_threshold = torch.quantile(deviations.flatten(), self.percentile / 100.0)
            threshold = max(stat_threshold, self.scaled_threshold)
        else:
            threshold = self.violation_threshold
        
        sparse_mask = deviations > threshold
        
        results = {
            'sparse_deviation_mask': sparse_mask,
            'deviation_magnitudes': deviations,
            'expectation_baseline': expectations,
            'threshold_used': threshold
        }
        
        if return_indices:
            # Get indices of violations for targeted attention
            results['violation_indices'] = torch.nonzero(sparse_mask)
        
        return results


class SaturationFractureDetector(nn.Module):
    """
    Tracks input sensitivity collapse (V_sat).
    If perturbations stop changing outputs -> dead region (saturation).
    If tiny perturbations flip many outputs -> brittle boundary (fracture).
    """
    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(
        self, 
        phi: torch.nn.Module, 
        x: torch.Tensor, 
        delta: float = 0.01
    ) -> torch.Tensor:
        """
        Compute the Saturation Fracture Score (V_sat).
        
        (Legacy) V_sat = (||Phi(x + d) - Phi(x)||_2^2) / (||d||_2^2)
        (Current) V_sat = min(1, ||Phi(x + d) - Phi(x)||_2 / ||d||_2)

        V_sat measures the input sensitivity. If small perturbations (Honest 
        Jitter) cause large flips in the symbolic output, the manifold is 
        considered 'Brittle' or 'Fractured' at that point.
        
        Args:
            phi: The functional block (saturated/symbolic).
            x: The input tensor.
            delta: The perturbation scale for the jitter.
            
        Returns:
            V_sat: [batch] fracture score.
            
        CODES v40 Invariant: 
            Symbolic Non-Revisability: 1.0. This score identifies when 
            symbols are unstable and require re-anchoring.
        """
        # Original output
        phi_x = phi(x) # [batch, K]
        
        # Perturbed output
        # SILICON SOVEREIGNTY: Replace stochastic noise with Honest Jitter
        noise = harvest_honest_jitter(x.shape, device=x.device, scaled=True) * delta
        phi_x_delta = phi(x + noise)
        
        # L0 difference (count flips)
        # Since phi is symbolic/saturated (e.g., -1, 1 or 0, 1), 
        # any change is a discrete flip.
        flips = (phi_x != phi_x_delta).float()
        V_sat = flips.sum(dim=-1) # [batch]
        
        return V_sat



class PalindromicRoutingCheck(nn.Module):
    """
    Enforces Strict Palindromic Routing (M_ab = M_ba).
    
    Replaces the empirical $O(N^3)$ TriadicReciprocityCheck.
    Guarantees trivial triadic tracking (Tr(P) = 1) 
    and bypasses continuous empirical checks in strongly stable regions.
    """
    def __init__(self, tolerance: float = 1e-4):
        super().__init__()
        self.tolerance = tolerance
        
    def check_cycle(self, hidden_states: torch.Tensor, indices: List[int]) -> bool:
        """
        Validate the cycle A->B->C->A by ensuring each segment is palindromic.
        Fast reject $O(K)$ implementation.
        """
        if len(indices) != 3:
            return False
            
        a = hidden_states[indices[0]]
        b = hidden_states[indices[1]]
        c = hidden_states[indices[2]]
        
        # Palindromic constraint: the transition must be symmetric.
        # This occurs when state norms are identical (or transition is symmetric).
        # Fast reject: if norms differ significantly, it's non-commutative.
        def check_symmetric(source, target):
            norm_s = torch.dot(source, source)
            norm_t = torch.dot(target, target)
            return torch.abs(norm_s - norm_t) < self.tolerance

        # If all links are palindromic, the Triadic cycle trace is trivially 1
        return check_symmetric(a, b) and check_symmetric(b, c) and check_symmetric(c, a)



class SparseExplorerRouting(nn.Module):
    """
    Routes high-violation tokens to deeper exploration via Random Walks.
    
    Implements a sparse random walker that samples the local neighborhood
    of high-violation tokens to approximate local persistent homology
    without full computation.
    
    Enhanced with strict Palindromic Routing checks.
    """
    
    def __init__(
        self,
        walk_length: int = 8,
        num_walks: int = 5,
        birth_death_epsilon: float = 0.1
    ):
        """
        Args:
            walk_length: Length of random walk for local exploration (5-10)
            num_walks: Number of random walks to sample per violation
            birth_death_epsilon: Threshold for spurious cycle detection
        """
        super().__init__()
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.birth_death_epsilon = birth_death_epsilon
        self.reciprocity_check = PalindromicRoutingCheck()
    
    def detect_local_cycles(
        self,
        hidden_states: torch.Tensor,
        violation_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Perform sparse random walk exploration around high-violation tokens.
        
        Samples the local neighborhood of high-violation tokens to 
        approximate local persistent homology. Implements 'Walk Back' 
        recovery and 'Track Jumping' to handle non-commutative cul-de-sacs.
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            violation_indices: [num_violations] indices of violating tokens
            attention_mask: [seq_len] valid tokens mask
            
        Returns:
            Dict with:
            - 'instability_detected': [num_violations] bools
            - 'total_aborts': int (count of reciprocity failures)
            - 'restarts': int (count of track jumps)
            A dictionary containing instability flags, abort counts, and restarts.
            
        CODES v40 Invariant: 
            Abortability Supremacy: 104. The ability to abort a walk and 
            restart from a new track is critical for manifold survival.
        """
        instability_detected = []
        total_aborts = 0
        total_restarts = 0
        
        seq_len = hidden_states.shape[0]
        
        # Pre-compute normalized states for similarity
        states_norm = hidden_states / (torch.norm(hidden_states, dim=1, keepdim=True) + 1e-8)
        
        for idx in violation_indices:
            start_node = idx.item()
            detected_instability = False
            
            # Monte Carlo sampling of local topology via random walks
            for _ in range(self.num_walks):
                current_node = start_node
                path_nodes = [current_node]
                path_sims = []
                
                for step in range(self.walk_length):
                    # 1. Compute local transition probs based on similarity
                    # (Restricted to small neighborhood for efficiency)
                    window_start = max(0, current_node - self.window_size // 2)
                    window_end = min(seq_len, current_node + (self.window_size // 2 + 1))

                    
                    # Local extraction
                    local_indices = torch.arange(window_start, window_end, device=hidden_states.device)
                    euclidean_sims = torch.mv(states_norm[window_start:window_end], states_norm[current_node])
                    
                    # Phase 8: RP^4 Projective Topology (Inverted Hypersphere Constraint)
                    # In an S^4/Z_2 projection, antipodal points (x ~ -x) are identified.
                    # We square the similarity so that extreme diametric oppositions 
                    # are treated as close neighbors, structurally linking "paradoxes"
                    # without gradient death or zero-crossing lobotomy.
                    local_sims = euclidean_sims.pow(2)
                    
                    # Mask self and invalid
                    local_sims[current_node - window_start] = -1e9 
                    
                    # Softmax routing
                    probs = torch.softmax(local_sims * 5.0, dim=0) # Temperature=0.2
                    
                    # ABORT RECOVERY: "Walk back and choose differently"
                    # Try up to 3 times to find a reciprocity-valid neighbor
                    next_node = -1
                    valid_step = False
                    
                    for attempt in range(3):
                        # Sample next step
                        next_idx_local = honest_multinomial(probs, 1).item()
                        candidate_node = window_start + next_idx_local
                        
                        # Triadic Reciprocity Check
                        if len(path_nodes) >= 2:
                            prev = path_nodes[-1]
                            prev_prev = path_nodes[-2]
                            if not self.reciprocity_check.check_cycle(hidden_states, [prev_prev, prev, candidate_node]):
                                # Reciprocity Violation -> "Walk Back" (Retry)
                                total_aborts += 1
                                continue # Try sampling again
                        
                        # If passed (or not applicable), accept
                        next_node = candidate_node
                        valid_step = True
                        break
                    
                    if not valid_step:
                        # "Jump Mental Tracks": Teleport to a random violation node
                        # if we are stuck in a non-commutative cul-de-sac
                        total_restarts += 1
                        if len(violation_indices) > 0:
                            # SILICON SOVEREIGNTY: Replaced torch.randint with Honest Jitter derivation
                            jitter = harvest_honest_jitter((1,), device=hidden_states.device, scaled=True).item()
                            rand_idx = int(jitter * len(violation_indices)) % len(violation_indices)
                            current_node = violation_indices[rand_idx].item()
                            path_nodes = [current_node] # Reset path
                            continue # Restart walk from new track
                        else:
                            break # No tracks to jump to
                    
                    # Record similarity
                    # (Re-calculate sim for the chosen node)
                    sim = torch.dot(states_norm[current_node], states_norm[next_node]).item()
                    path_sims.append(sim)
                    
                    # cycle detection: return to start
                    if next_node == start_node and len(path_nodes) > 2:
                        min_sim = min(path_sims)
                        if min_sim < self.birth_death_epsilon:
                            detected_instability = True 
                        break
                        
                    path_nodes.append(next_node)
                    current_node = next_node
                
                if detected_instability:
                    break
            
            instability_detected.append(detected_instability)
        
        return {
            'instability_detected': instability_detected,
            'total_aborts': total_aborts,
            'total_restarts': total_restarts
        }


class GyroidCovarianceEstimator(nn.Module):
    """
    Tensor-based Entropy Estimator using Gyroidic Manifold Covariance.
    
    Replaces scalar std() with proper gyroidic covariance trace and spectral entropy.
    Maintains a rolling buffer of samples for robust estimation.
    
    Uses the spectral properties of the covariance matrix:
    - Trace(C) = sum of eigenvalues = total variance
    - Spectral Entropy = -sum(p_i * log(p_i)) where p_i = _i / 
    """
    def __init__(self, dim: int, sample_size: int = 16, ema_decay: float = 0.9):
        super().__init__()
        self.dim = dim
        self.sample_size = sample_size
        self.ema_decay = ema_decay
        
        # Rolling buffer of samples for covariance estimation
        self.register_buffer('sample_buffer', torch.zeros(sample_size, dim))
        self.register_buffer('buffer_idx', torch.tensor(0))
        self.register_buffer('buffer_filled', torch.tensor(False))
        
        # EMA-smoothed covariance estimate
        self.register_buffer('cov_ema', torch.eye(dim) * 0.1)
        
    def update_buffer(self, sample: torch.Tensor):
        """Add a sample to the rolling buffer."""
        # sample: [1, dim] or [batch, dim]
        if sample.dim() == 2:
            sample = sample[0]  # Take first if batched
        
        idx = self.buffer_idx.item() % self.sample_size
        self.sample_buffer[idx] = sample.detach()
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.sample_size:
            self.buffer_filled.fill_(True)
        
    def compute_covariance(self) -> torch.Tensor:
        """Compute sample covariance from buffer."""
        if self.buffer_filled:
            samples = self.sample_buffer  # [sample_size, dim]
        else:
            n_filled = min(self.buffer_idx.item(), self.sample_size)
            if n_filled < 2:
                return self.cov_ema
            samples = self.sample_buffer[:n_filled]
        
        # Center samples
        mean = samples.mean(dim=0, keepdim=True)
        centered = samples - mean
        
        # Compute covariance: C = (X^T X) / (n-1)
        n = samples.shape[0]
        cov = torch.mm(centered.T, centered) / max(n - 1, 1)
        
        # EMA update
        self.cov_ema = self.ema_decay * self.cov_ema + (1 - self.ema_decay) * cov
        
        return self.cov_ema
        
    def estimate_entropy(self, sample: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Estimate spectral entropy from the covariance matrix.
        
        Spectral Entropy = - (p_i * log(p_i)) where p_i = _i / 
        Higher entropy = more spread across eigenvalues = higher uncertainty.
        
        Higher entropy indicates that the variance is spread across many 
        eigenvalues, suggesting a high-dimensional, uncertain state. Low 
        entropy suggests a collapsed, more certain state.

        Args:
            sample: Optional new sample to add to the buffer.
            
        Returns:
            entropy: Scalar tensor representing spectral entropy.
            
        CODES v40 Invariant: 
            Manifold Dimension Invariance: 31.0. Entropy tracking prevents 
            the manifold from collapsing toward a single basis (Lobotomy).
        """
        if sample is not None:
            self.update_buffer(sample)
        
        cov = self.compute_covariance()
        
        # Eigendecomposition with safety clamp
        try:
            # Sanitize covariance matrix for MKL stability
            cov_sanitized = torch.clamp(cov, -1e6, 1e6)
            if torch.isnan(cov_sanitized).any():
                cov_sanitized = torch.where(torch.isnan(cov_sanitized), torch.zeros_like(cov_sanitized), cov_sanitized)
            eigenvalues = torch.linalg.eigvalsh(cov_sanitized)
        except Exception as e:
            # Fallback to simpler trace-based entropy
            print(f"[WARN] Eigendecomposition stability failure: {e}")
            return torch.log(torch.trace(cov).clamp(min=1e-6))
        
        # Ensure positive (numerical stability)
        eigenvalues = eigenvalues.clamp(min=1e-8)
        
        # Normalize to probability distribution
        total = eigenvalues.sum()
        probs = eigenvalues / total.clamp(min=1e-8)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-8)))
        
        return entropy
    
    def estimate_trace(self, sample: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Estimate trace of covariance (total variance).
        
        Args:
            sample: Optional new sample to add to buffer first
            
        Returns:
            trace: Scalar tensor
        """
        if sample is not None:
            self.update_buffer(sample)
        
        cov = self.compute_covariance()
        return torch.trace(cov)

    def get_elipsodistrophy_metrics(self, sample: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Measures the spectral envelope as Hyperbolic Shear (System 2 Driver).

        ECCENTRICITY = log(max() / min())
        SHEAR = 2 * tanh(ECCENTRICITY / 2)

        NOTE: No additional O(N) cost  it rides the existing spectral decomposition.
        """
        if sample is not None:
            self.update_buffer(sample)

        cov = self.compute_covariance()

        try:
            cov_sanitized = torch.clamp(cov, -1e6, 1e6)
            if torch.isnan(cov_sanitized).any():
                cov_sanitized = torch.where(
                    torch.isnan(cov_sanitized),
                    torch.zeros_like(cov_sanitized),
                    cov_sanitized
                )
            eigenvalues = torch.linalg.eigvalsh(cov_sanitized)
            eigenvalues = eigenvalues.clamp(min=1e-8)
        except Exception:
            return {'atrophy': 0.0, 'spectral_width': 1.0, 'is_dangerously_legible': False}

        evs = torch.sort(eigenvalues, descending=True)[0]
        lambda_max = evs[0]
        lambda_min = evs[-1]

        # Hyperbolic Eccentricity
        eccentricity = torch.log(lambda_max / (lambda_min + 1e-9)).item()

        # Hyperbolic Shear (Poincar Projection)
        shear = 2.0 * torch.tanh(torch.tensor(eccentricity / 2.0)).item()
        
        # Diffusion Coefficient for SDEs
        diffusion_coefficient = 0.1 * (1.0 + shear)

        # Atrophy: Calculate by applying local correlation to the eigenvalue spectrum
        from core.martinova_correlation import compute_bounded_correlation
        corr = compute_bounded_correlation(eigenvalues.unsqueeze(-1).unsqueeze(0)).squeeze(0)
        atrophy = corr.item()
        is_dangerously_legible = atrophy > 0.85
        trigger_defibrillator = atrophy >= 0.99

        return {
            'atrophy': atrophy,
            'hyperbolic_shear': shear,
            'eccentricity': eccentricity,
            'diffusion_coefficient': diffusion_coefficient,
            'spectral_width': (lambda_max - lambda_min).item(),
            'is_dangerously_legible': is_dangerously_legible,
            'trigger_defibrillator': trigger_defibrillator
        }
class LeyLineGeodesicMetric(nn.Module):
    """
    Anisotropic Ley Line Geodesic Metric.
    
    Computes preferred geodesics in state space based on constraint-induced curvature.
    Implements a non-Euclidean metric g_{ij}(x) where 'ley lines' are paths
    that minimize the anisotropic action.
    """
    def __init__(self, dim: int, anisotropy_init: float = 1.0):
        super().__init__()
        self.dim = dim
        self.g_base = nn.Parameter(torch.eye(dim) * anisotropy_init)
        
    def compute_metric(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent metric tensor g_{ij}(x).
        
        In this implementation, the metric warped by the local variance 
        (covariance) to favor directions of lower resistance (sparse ley lines).
        """
        # Outer product for simple anisotropy
        # x is [dim] or [1, dim]
        if x.dim() == 1:
            x_col = x.unsqueeze(1)
            x_row = x.unsqueeze(0)
        else:
            x_col = x.transpose(-2, -1)
            x_row = x
        warp = torch.sigmoid(torch.matmul(x_col, x_row))
        return self.g_base + warp * 0.1
        
    def geodesic_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute anisotropic distance: sqrt( (x1-x2)^T G (x1-x2) )
        """
        delta = x1 - x2
        G = self.compute_metric(x1)
        # Using the midpoint approximation for G
        dist_sq = torch.matmul(delta.unsqueeze(1), torch.matmul(G, delta.unsqueeze(2)))
        return torch.sqrt(dist_sq.squeeze() + 1e-8)

class MoebiusFiberBundle(nn.Module):
    """
    Orientation-twisted recursive fiber bundle.
    
    Implements a transition function g satisfying g  O(n) \ SO(n),
    causing orientation reversal on traversal (Mbius holonomy).
    """
    def __init__(self, dim: int, fiber_dim: int):
        super().__init__()
        self.dim = dim
        self.fiber_dim = fiber_dim
        
        # Transition operator that includes a reflection (det = -1)
        reflection = torch.eye(dim)
        reflection[0, 0] = -1.0
        self.register_buffer('transition_twist', reflection)
        
        self.fiber_projection = nn.Linear(dim, fiber_dim)
        
    def forward(self, x: torch.Tensor, twist_gate: torch.Tensor) -> torch.Tensor:
        """
        Recursive twisted bundle step.
        
        x: Base state
        twist_gate: Trigger for orientation reversal (e.g. crossing a facet boundary)
        """
        # Apply twist if gated
        twisted_x = torch.where(twist_gate.unsqueeze(-1) > 0.5, 
                                torch.matmul(x, self.transition_twist), 
                                x)
        
        # Project to fiber space
        fiber_state = self.fiber_projection(twisted_x)
        return fiber_state

