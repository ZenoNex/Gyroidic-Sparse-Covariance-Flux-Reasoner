"""
Topological Ingestion Validator
Deterministic boundary gate replacing probabilistic classifiers and bastardized Repunit probes.

Implements the D-module structural check at the ingestion boundary:
1. RationalSnappingLayer (bit-exact fixed-point lattice projection)
2. Holonomic Rank (entropy-adaptive SVD cutoff on polynomial residue covariance)
3. Cohomological Dimension (K - rank)
4. Lazarus Void detection (cohom_dim > K // 2)
5. Soliton Entropy (persistent singleton estimator, non-ergodicity check)
6. PAS_h Phase Alignment (against live manifold state, optional)

No PRNGs. No learned classification heads. Only geodesic drift primitives.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional

try:
    from src.core.primitive_ops import FixedPointField, SCALE_FACTOR
except ImportError:
    FixedPointField = None
    SCALE_FACTOR = 65536.0

try:
    from src.core.polynomial_coprime import PolynomialCoprimeConfig
except ImportError:
    PolynomialCoprimeConfig = None


class _PersistentEntropyEstimator:
    """
    Singleton non-ergodic entropy estimator with running history.

    Prevents the "fresh random instance every call" bug that killed
    the NonErgodicEntropyEstimator in voynich_architecture.py.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.history = []
            cls._instance.max_history = 100
        return cls._instance

    def __call__(self, signal: torch.Tensor) -> Dict[str, float]:
        flat = signal.flatten()
        n = flat.numel()

        if n < 2:
            return {"soliton_entropy": 0.0, "ergodic_entropy": 1.0, "is_slop": True, "threshold": 1e-6}

        centered = flat - flat.mean()
        if centered.abs().max() < 1e-8:
            return {"soliton_entropy": 0.0, "ergodic_entropy": 1.0, "is_slop": True, "threshold": 1e-6}

        # Spectral peakiness via FFT
        spec = torch.fft.rfft(centered).abs() ** 2
        total = spec.sum()
        if total < 1e-8:
            return {"soliton_entropy": 0.0, "ergodic_entropy": 1.0, "is_slop": True, "threshold": 1e-6}

        probs = spec / total
        spectral_entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_ent = math.log(max(1, len(probs)))
        norm_ent = spectral_entropy / max_ent if max_ent > 0 else 0.0

        soliton = 1.0 - norm_ent  # 1.0 = pure soliton, 0.0 = white noise

        self.history.append(soliton)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Adaptive threshold: 10th percentile of history, floored at 1e-6
        threshold = 1e-6
        if len(self.history) >= 10:
            hist_sorted = sorted(self.history)
            adaptive = hist_sorted[max(0, len(self.history) // 10 - 1)]
            threshold = max(threshold, adaptive * 0.5)

        return {
            "soliton_entropy": soliton,
            "ergodic_entropy": norm_ent,
            "is_slop": soliton < threshold,
            "threshold": threshold,
        }


class RationalSnap(nn.Module):
    """Bit-exact projection to Q via 2^16 fixed-point lattice."""
    def __init__(self, scale: float = SCALE_FACTOR):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if FixedPointField is not None:
            fp = FixedPointField(x, scale=self.scale)
            return fp.forward()
        # Fallback: deterministic rounding
        return torch.round(x * self.scale) / self.scale


class TopologicalIngestionValidator(nn.Module):
    """
    Deterministic topological gate for the ingestion boundary.

    Replaces:
    - TextbookFilter LSTM classifier (learned, probabilistic)
    - SparseRepunitProbe at wrong position (fixed-threshold, arithmetic)
    - Bastardized VoynichLinguist in minecraft_ingestor (zero-vector input)
    - Missing validation in diegetic_backend bimodal panels

    Validates by computing structural rank of data against the active
    polynomial coprime configuration. Sterile data (low rank, low soliton
    entropy, high cohomological dimension) is refused at the door.
    """

    def __init__(
        self,
        poly_config: Optional[Any],
        state_dim: int,
        scale: float = SCALE_FACTOR,
        min_rank_ratio: float = 0.3,
        pas_h_tolerance: float = 0.05,
    ):
        super().__init__()
        self.poly_config = poly_config
        self.state_dim = state_dim
        self.K = poly_config.k if poly_config is not None else 5
        self.min_rank_ratio = min_rank_ratio
        self.pas_h_tolerance = pas_h_tolerance
        self.snapper = RationalSnap(scale)
        self.entropy_estimator = _PersistentEntropyEstimator()

        # Deterministic structural projection matrix.
        # NOT learned. NOT random. Built from deterministic irrational frequency.
        self.register_buffer(
            'proj_matrix',
            self._build_deterministic_projection(state_dim, self.K)
        )

    def _build_deterministic_projection(self, in_dim: int, out_dim: int) -> torch.Tensor:
        i = torch.arange(in_dim, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(out_dim, dtype=torch.float32).unsqueeze(0)
        # Deterministic quasi-orthogonal initialization via irrational frequency
        M = torch.sin((i + 1) * (j + 1) * math.pi * 0.1234567891234)
        # Orthogonalize only the square submatrix, then pad
        m = min(in_dim, out_dim)
        q, _ = torch.linalg.qr(M[:m, :m])
        P = torch.zeros(in_dim, out_dim, dtype=torch.float32)
        P[:q.shape[0], :q.shape[1]] = q
        return P

    def _text_to_tensor(self, text: str, device: torch.device) -> torch.Tensor:
        """Deterministic text encoding without learned embeddings."""
        dim = self.state_dim
        chars = [ord(c) % 256 for c in text[:dim]]
        # Deterministic pad by repeating the pattern
        if len(chars) < dim:
            repeat = dim // len(chars) + 2
            chars = (chars * repeat)[:dim]
        t = torch.tensor(chars, dtype=torch.float32, device=device) / 128.0 - 1.0
        return t

    def _evaluate_residues(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate input against K polynomial coprime functionals."""
        batch = x.shape[0]
        K = self.K

        # Ensure state_dim
        if x.shape[-1] != self.state_dim:
            if x.shape[-1] < self.state_dim:
                pad = self.state_dim - x.shape[-1]
                x = torch.nn.functional.pad(x, (0, pad))
            else:
                x = x[..., :self.state_dim]

        channel_inputs = x @ self.proj_matrix  # [batch, K]

        if self.poly_config is None:
            # Fallback: identity mapping if no poly config available
            return channel_inputs

        residues = torch.zeros(batch, K, device=x.device, dtype=x.dtype)
        for k in range(K):
            residues[:, k] = self.poly_config.evaluate_polynomial(k, channel_inputs[:, k])
        return residues

    def _spectral_rank(self, residues: torch.Tensor, entropy: float) -> int:
        """Holonomic rank via entropy-adaptive SVD cutoff."""
        # Covariance of residues: [K, K]
        mat = residues.T @ residues
        try:
            S = torch.linalg.svdvals(mat)
        except Exception:
            S = torch.linalg.eigvals(mat).abs()

        cutoff = 1e-4 * (1.0 + entropy)
        return int((S > cutoff).sum().item())

    def _pas_h(self, residues: torch.Tensor, manifold: Optional[torch.Tensor]) -> float:
        """Phase alignment against live manifold state."""
        if manifold is None:
            return 1.0

        r = residues.mean(dim=0) if residues.dim() > 1 else residues
        m = manifold.flatten()[:r.shape[0]]

        if m.numel() < r.numel():
            m = torch.nn.functional.pad(m, (0, r.numel() - m.numel()))
        else:
            m = m[:r.numel()]

        # Complex representation: split in half
        half = r.shape[0] // 2
        if half == 0:
            return 1.0

        z_r = torch.complex(r[:half], r[half:2*half])
        z_m = torch.complex(m[:half], m[half:2*half])

        dot = (z_r * z_m.conj()).sum()
        norm = torch.abs(z_r).sum() * torch.abs(z_m).sum()
        if norm < 1e-8:
            return 0.0

        return float(torch.abs(dot / norm).item())

    def validate_text(self, text: str, manifold_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Validate a text string at the ingestion boundary."""
        device = manifold_state.device if manifold_state is not None else torch.device('cpu')
        x = self._text_to_tensor(text, device).unsqueeze(0)  # [1, state_dim]
        return self.forward(x, manifold_state)

    def validate_tensor(self, x: torch.Tensor, manifold_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Validate an existing tensor at the ingestion boundary."""
        return self.forward(x, manifold_state)

    def forward(self, x: torch.Tensor, manifold_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Returns:
            {
                "admissible": bool,
                "holonomic_rank": int,
                "cohomological_dimension": int,
                "soliton_entropy": float,
                "pas_h": float,
                "is_lazarus_void": bool,
                "reason": str,
                "residues": Tensor,
            }
        """
        # 1. Rational snap
        x_snap = self.snapper(x)

        # 2. Polynomial residue evaluation
        residues = self._evaluate_residues(x_snap)

        # 3. Entropy (singleton, persistent)
        ent = self.entropy_estimator(residues)

        # 4. Spectral rank
        rank = self._spectral_rank(residues, ent["ergodic_entropy"])
        cohom_dim = self.K - rank

        # 5. PAS_h
        pas_h = self._pas_h(residues, manifold_state)

        # 6. Decision
        reasons = []
        admissible = True

        if ent["is_slop"]:
            reasons.append(f"SOLITON_TOO_LOW({ent['soliton_entropy']:.2e}<{ent['threshold']:.2e})")
            admissible = False

        min_rank = max(1, int(self.min_rank_ratio * self.K))
        if rank < min_rank:
            reasons.append(f"RANK_TOO_LOW({rank}/{min_rank})")
            admissible = False

        is_void = cohom_dim > (self.K // 2)
        if is_void:
            reasons.append(f"LAZARUS_VOID({cohom_dim}>{self.K//2})")
            admissible = False

        if manifold_state is not None and pas_h < 0.1:
            reasons.append(f"PAS_H_INCOHERENT({pas_h:.3f})")
            admissible = False

        return {
            "admissible": admissible,
            "holonomic_rank": rank,
            "cohomological_dimension": cohom_dim,
            "soliton_entropy": ent["soliton_entropy"],
            "ergodic_entropy": ent["ergodic_entropy"],
            "pas_h": pas_h,
            "is_lazarus_void": is_void,
            "reason": "; ".join(reasons) if reasons else "STRUCTURALLY_HONEST",
            "residues": residues.detach(),
        }
