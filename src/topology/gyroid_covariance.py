"""
Sparse covariance probes with gyroid-inspired violation detection.

Computes local spectral signatures to detect topology violations
without expensive global persistent homology.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import math


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
        Args:
            hidden_dim: Dimension of hidden states
            window_size: Size of local covariance window
            k_hop: Neighborhood hop distance
            num_eigenvalues: Number of top eigenvalues to track
            violation_threshold: Fixed threshold for violations
            use_saturation_detection: Enable Saturation Fracture Detector
            adaptive_threshold: Use adaptive percentile-based threshold
            percentile: Percentile for adaptive threshold (e.g., 95th)
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
    
    def compute_local_covariance(
        self,
        hidden_states: torch.Tensor,
        start_idx: int
    ) -> torch.Tensor:
        """
        Compute local windowed covariance matrix.
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            start_idx: Starting position for window
            
        Returns:
            C_loc: [window_size, window_size] local covariance
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
        
        V_m = V + H_mischief/tau - lambda_min/tr(C)
        
        Args:
            C_loc: Local covariance matrix [window, window]
            h_mischief: Current mischief entropy value
            tau_decay: Decay constant for mischief reward
        
        Returns:
            V_m: GCVE score (higher = more playful/structured violation)
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
        
        Returns:
            - eigenvalues: Top k eigenvalues
            - spectral_gap: λ_k - λ_{k+1}
            - decay_rate: (λ_1 - λ_m) / m
            - trace: Trace of covariance
            - condition_number: λ_max / λ_min
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
        Compute gyroid pressure metric.
        
        GCVE Metric:
        pressure = max(0, gap(λ) / decay_rate) + λ_min / trace(C_loc)
        
        High pressure indicates:
        - Sharp spectral gaps (disconnected components)
        - Poor spectral decay (not minimal surface-like)
        - Low minimum eigenvalue relative to trace (degenerate directions)
        
        Args:
            spectral_signature: Output from compute_spectral_signature
            
        Returns:
            pressure_score: scalar
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
        
        # Combined pressure
        pressure = topo_term + geo_term
        
        return pressure

    def forward(
        self, 
        h: torch.Tensor, 
        phi_fn: Optional[torch.nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Orchestrate violation detection.
        
        Args:
            h: [batch, seq_len, hidden_dim] hidden states
            phi_fn: Optional symbolic functional for fracture detection
            
        Returns:
            Results dictionary containing violations and scores
        """
        batch_size, seq_len, _ = h.shape
        violations = []
        gcve_pressures = []
        
        # 1. Compute GCVE per batch element (Geometric/Spectral)
        for b in range(batch_size):
             # For simplicity, we sample the middle window or multiple windows
             # Real implementation would scan across seq_len
             C_loc = self.compute_local_covariance(h[b], start_idx=max(0, seq_len//2 - 16))
             sig = self.compute_spectral_signature(C_loc)
             score = self.compute_pressure_score(sig)
             gcve_pressures.append(score)
             violations.append(score > self.violation_threshold)
             
        gcve_pressures = torch.stack(gcve_pressures) # [batch]
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
            'total_pressure': total_pressure
        }

    def compute_interference_matrix(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise interference between batch elements.
        
        Equation: [partial_t Phi_i \circ \Phi_j]_{i \neq j}
        
        Args:
            h: [batch, seq_len, hidden_dim]
            
        Returns:
            inter_matrix: [batch, batch] pairwise interference scores
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
        
        Pointer #8: Semantics appear where covariance breaks minimal-surface expectations.
        GCVE-style scouts must hunt violations, not smoothness.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            return_indices: If True, return indices of violation locations
            
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
                # Expected decay: λ_i ≈ λ_1 * exp(-i/τ) for some time constant τ
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
        Compute Saturation Fracture Score:
        V_sat = E_deltax [ |phi(x+deltax) - phi(x)|_0 ]
        
        Args:
            phi: Functional block (must be saturated/symbolic)
            x: Input tensor [batch, ...]
            delta: Perturbation scale
            
        Returns:
            V_sat: [batch] fracture score
        """
        # Original output
        phi_x = phi(x) # [batch, K]
        
        # Perturbed output
        noise = torch.randn_like(x) * delta
        phi_x_delta = phi(x + noise)
        
        # L0 difference (count flips)
        # Since phi is symbolic/saturated (e.g., -1, 1 or 0, 1), 
        # any change is a discrete flip.
        flips = (phi_x != phi_x_delta).float()
        V_sat = flips.sum(dim=-1) # [batch]
        
        return V_sat

        return results


class TriadicReciprocityCheck(nn.Module):
    """
    Enforces Triadic Reciprocity for non-commutative routing.
    
    Checks consistency of 3-cycles (A->B->C->A) to prevent
    divergent blowups in the routing manifold.
    
    Implements the "Spectral Quantization" of the cycle product trace.
    """
    def __init__(self, tolerance: float = 0.1, quantization_levels: int = 8):
        super().__init__()
        self.tolerance = tolerance
        self.quantization_levels = quantization_levels
        
    def check_cycle(self, hidden_states: torch.Tensor, indices: List[int]) -> bool:
        """
        Validate the triadic cycle A->B->C->A.
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            indices: [3] list of node indices [i, j, k]
            
        Returns:
            is_valid: boolean
        """
        if len(indices) != 3:
            return False
            
        # Extract states
        a = hidden_states[indices[0]]
        b = hidden_states[indices[1]]
        c = hidden_states[indices[2]]
        
        # Compute non-commutative transition operators (Approximate)
        # T_xy = outer(y, x) / |x|^2
        def transition_op(source, target):
            norm_sq = torch.dot(source, source) + 1e-8
            return torch.outer(target, source) / norm_sq
            
        M_ab = transition_op(a, b)
        M_bc = transition_op(b, c)
        M_ca = transition_op(c, a)
        
        # Compute Cycle Product: P = M_ca * M_bc * M_ab
        # Note: Order matters for non-commutative ops
        P = torch.mm(M_ca, torch.mm(M_bc, M_ab))
        
        # Check 1: Trace Stability (Reciprocity)
        # Ideally identity-like or unitary, trace should be close to 1.0 or quantized value
        tr_P = torch.trace(P)
        
        # Quantization Snap
        # "Force cycle product to snap to discrete set"
        # We check if trace is close to an integer (quantized structural resonance)
        tr_val = tr_P.item()
        nearest_int = round(tr_val)
        
        deviation = abs(tr_val - nearest_int)
        
        if deviation > self.tolerance:
            return False # Failed quantization snap
            
        # Check 2: Norm Blowup (Divergence Abort)
        norm_P = torch.norm(P)
        if norm_P > 1.5: # Allow some gain but abort explosion
            return False
            
        return True


class SparseExplorerRouting(nn.Module):
    """
    Routes high-violation tokens to deeper exploration via Random Walks.
    
    Implements a sparse random walker that samples the local neighborhood
    of high-violation tokens to approximate local persistent homology
    without full computation.
    
    Enhanced with Triadic Reciprocity Check.
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
        self.reciprocity_check = TriadicReciprocityCheck()
    
    def detect_local_cycles(
        self,
        hidden_states: torch.Tensor,
        violation_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Perform sparse random walk exploration around high-violation tokens.
        
        Implements Abort Recovery:
        - "Walk Back": Retry sampling if reciprocity check fails.
        - "Jump Mental Tracks": Teleport to new violation node if stuck.
        
        Args:
            hidden_states: [seq_len, hidden_dim]
            violation_indices: [num_violations] indices of violating tokens
            attention_mask: [seq_len] valid tokens mask
            
        Returns:
            Dict with:
            - 'instability_detected': [num_violations] bools
            - 'total_aborts': int (count of reciprocity failures)
            - 'restarts': int (count of track jumps)
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
                    window_start = max(0, current_node - 16)
                    window_end = min(seq_len, current_node + 17)
                    
                    # Local extraction
                    local_indices = torch.arange(window_start, window_end, device=hidden_states.device)
                    local_sims = torch.mv(states_norm[window_start:window_end], states_norm[current_node])
                    
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
                        next_idx_local = torch.multinomial(probs, 1).item()
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
                            rand_idx = torch.randint(0, len(violation_indices), (1,)).item()
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
    - Spectral Entropy = -sum(p_i * log(p_i)) where p_i = λ_i / Σλ
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
        
        Spectral Entropy = -Σ (p_i * log(p_i)) where p_i = λ_i / Σλ
        Higher entropy = more spread across eigenvalues = higher uncertainty.
        
        Args:
            sample: Optional new sample to add to buffer first
            
        Returns:
            entropy: Scalar tensor representing spectral entropy
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
            print(f"⚠️ Eigendecomposition stability failure: {e}")
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
    
    Implements a transition function g satisfying g ∈ O(n) \ SO(n),
    causing orientation reversal on traversal (Möbius holonomy).
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
