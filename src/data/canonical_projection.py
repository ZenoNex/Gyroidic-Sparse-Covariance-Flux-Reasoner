"""
Canonical Projection Utilities

Provides a unified, topology-consistent projection for ingestion modalities
(text, topics/wiki chunks, image feature vectors) into the engine manifold.

Design goals:
- Deterministic, numeric-only outputs with finite guarantees.
- Consistent with repair pipeline: pad -> view [B, K, residue_dim] -> transform ->
  flatten -> truncate to [1, dim].
- Uses the same polynomial family and associator as the core engine path.
- Exposes entropy diagnostics via GyroidCovarianceEstimator for energy-based control.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F
import math

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.models.diegetic_heads import DataAssociationLayer
from src.topology.gyroid_covariance import GyroidCovarianceEstimator


class CanonicalProjector:
    def __init__(self, dim: int = 64, k: int = 5, device: str = 'cpu'):
        self.dim = dim
        self.k = k
        self.device = device

        # Polynomial family consistent with repair subsystems
        self.poly_config = PolynomialCoprimeConfig(
            k=k, degree=4, basis_type='chebyshev', learnable=True, use_saturation=True, device=device
        )
        # Associator to map states to residues similar to the backend
        self.associator = DataAssociationLayer(input_dim=dim, hidden_dim=dim, k=k).to(device, non_blocking=True)
        # Covariance estimator for energy/entropy
        self.gyroid_cov = GyroidCovarianceEstimator(dim=dim, sample_size=16)
        
        # Backward-compat toggles for old encodings/frontends
        self.backward_compat = True
        self.default_long_edges = [64, 128]  # L0, L1 caps; L2 ROI-based
        self.edge_threshold = 0.12
        self.entropy_threshold = 0.75
        self.max_rois = 4

    # -----------------------------
    # Text/Topic projection helpers
    # -----------------------------
    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Replicates the backend's sequence-aware polynomial rotating hash to [1, dim]."""
        vec = torch.zeros(1, self.dim, device=self.device)

        # Chebyshev-based positional coefficients
        num_coeffs = 12
        poly_coeffs = []
        x = 0.5
        for n in range(num_coeffs):
            if n == 0:
                coeff = 1.0
            elif n == 1:
                coeff = x
            else:
                t_prev2 = 1.0
                t_prev1 = x
                for _ in range(2, n + 1):
                    t_curr = 2 * x * t_prev1 - t_prev2
                    t_prev2 = t_prev1
                    t_prev1 = t_curr
                coeff = t_prev1
            poly_coeffs.append(abs(coeff * 10) + 2)  # ensure >=2

        for i, ch in enumerate(text):
            p = poly_coeffs[i % len(poly_coeffs)]
            idx = int((i * p + ord(ch)) % self.dim)
            magnitude = (ord(ch) / 128.0) * (1.0 / (math.log(i + 2)))
            vec[0, idx] += magnitude

        if len(text) > 0:
            salt = sum(ord(c) for c in text) % self.dim
            vec[0, salt] *= 1.1

        return vec / (vec.norm() + 1e-8)

    def project_text_to_state(self, text: str) -> Dict[str, Any]:
        """Project text to a canonical [1, dim] state with entropy diagnostic."""
        base = self._text_to_tensor(text)  # [1, dim]
        # Residue mapping for alignment with K
        with torch.no_grad():
            residues = torch.tanh(self.associator.residue_map(base))  # [1, k]
        # Reconstruct back to [1, dim] via pad/view/flatten scheme (simple tiling)
        state = self._reconstruct_from_residues(base)
        entropy = self.gyroid_cov.estimate_entropy(state)
        return {
            'state': state,                # [1, dim]
            'entropy': float(entropy.item())
        }

    def project_topic_to_state(self, topic_text: str) -> Dict[str, Any]:
        return self.project_text_to_state(topic_text)

    # -----------------------------
    # Image/multiscale projection
    # -----------------------------
    def project_multiscale_image(self, levels: List[torch.Tensor], weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Project multi-scale image features to [1, dim] with optional adaptive coefficient windowing.
        Backward-compatible: if called with flat features, still returns fused state/entropy.
        """
        states = []
        entropies = []
        windows = []
        for feat in levels:
            state = self._feature_to_state(feat)
            states.append(state)
            entropies.append(float(self.gyroid_cov.estimate_entropy(state).item()))
            # Build simple edge-based coefficient window from feature if 2D-like
            win = self._default_coeff_window_from_feature(feat)
            windows.append(win)

        if weights is None:
            # Lower entropy → higher weight (normalize inversely)
            # Adjusted: Give more weight to higher entropy/novelty states
            inv = torch.tensor([max(e, 1e-6) for e in entropies], dtype=torch.float32)
            w = (inv / (inv.sum() + 1e-8)).tolist()
        else:
            s = sum(weights) + 1e-8
            w = [wi / s for wi in weights]

        fused = torch.zeros(1, self.dim, device=self.device)
        for wi, st in zip(w, states):
            fused += float(wi) * st
        fused = fused / (fused.norm() + 1e-8)

        # Adaptive residue modulation via averaged window
        avg_window = None
        for win in windows:
            if win is None:
                continue
            if avg_window is None:
                avg_window = win.clone()
            else:
                avg_window += win
        if avg_window is not None:
            avg_window = avg_window / (avg_window.norm() + 1e-8)
            fused = self._apply_window_in_residue_space(fused, avg_window)

        entropy = self.gyroid_cov.estimate_entropy(fused)
        return {
            'state': fused,
            'entropy': float(entropy.item()),
            'component_entropies': entropies,
            'weights': w
        }

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _feature_to_state(self, feat: torch.Tensor) -> torch.Tensor:
        """Map a flat or 2D feature to [1, dim] with residue-aware reconstruction."""
        f = feat
        if f.dim() >= 2:
            f = f.float().to(self.device, non_blocking=True)
            # Reduce 2D map to 1D via average pooling over one axis then flatten
            pooled = f.mean(dim=0).flatten()
        else:
            pooled = f.flatten().to(self.device, non_blocking=True).float()
        # Interpolate to target dim
        if pooled.numel() == self.dim:
            base = pooled.unsqueeze(0)
        else:
            src = pooled.numel()
            idx = torch.linspace(0, src - 1, self.dim, device=self.device)
            x0 = torch.clamp(idx.floor().long(), 0, src - 1)
            x1 = torch.clamp(x0 + 1, 0, src - 1)
            t = (idx - x0.float())
            base = ((1 - t) * pooled[x0] + t * pooled[x1]).unsqueeze(0)
        base = base / (base.norm() + 1e-8)
        return self._reconstruct_from_residues(base)

    def _reconstruct_from_residues(self, base: torch.Tensor) -> torch.Tensor:
        """
        Create a residue-aligned representation and flatten back to [1, dim].
        This mirrors the pad/view/flatten/truncate loop for K compatibility.
        """
        bsz = base.shape[0]
        state_dim = base.shape[1]
        if state_dim % self.k != 0:
            pad_sz = self.k - (state_dim % self.k)
            base = F.pad(base, (0, pad_sz), mode='reflect')
        padded_dim = base.shape[1]
        residue_dim = padded_dim // self.k
        residues = base.view(bsz, self.k, residue_dim)
        flat = residues.view(bsz, -1)
        if flat.shape[1] > state_dim:
            flat = flat[:, :state_dim]
        flat = flat / (flat.norm() + 1e-8)
        return flat

    # -----------------------------
    # Advanced image path with pyramids/aspect handling
    # -----------------------------
    def project_image_path_to_state(self, path: str, max_pixels: int = 1_000_000, long_edges: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Load an image robustly, build an adaptive multi-scale representation, and project to [1, dim].
        Preserves backward compatibility by always returning finite numeric outputs.
        """
        try:
            from PIL import Image
        except Exception:
            raise RuntimeError("PIL (Pillow) is required for image path projection")

        long_edges = long_edges or self.default_long_edges
        img = Image.open(path).convert('RGB')
        W, H = img.size
        # Cap total pixels
        scale = min(1.0, math.sqrt(max_pixels / max(1, W * H)))
        if scale < 1.0:
            img = img.resize((max(1, int(W * scale)), max(1, int(H * scale))))
            W, H = img.size

        # Build L0
        l0 = self._resize_long_edge(img, long_edges[0])
        l0_t = self._to_tensor(l0)
        edge0 = self._compute_edge_field(l0_t)
        group0 = self._compute_group_field(l0_t, edge0)
        win0 = self._field_to_coeff_window(edge0, group0)
        st0 = self._feature_to_state(l0_t)

        # Gating for L1 based on entropy/edge
        ent0 = float(self.gyroid_cov.estimate_entropy(st0).item())
        edge_density0 = float(edge0.mean().item())
        states = [st0]
        entropies = [ent0]
        windows = [win0]

        if ent0 > self.entropy_threshold or edge_density0 > self.edge_threshold:
            l1 = self._resize_long_edge(img, long_edges[1])
            l1_t = self._to_tensor(l1)
            edge1 = self._compute_edge_field(l1_t)
            group1 = self._compute_group_field(l1_t, edge1)
            win1 = self._field_to_coeff_window(edge1, group1)
            st1 = self._feature_to_state(l1_t)
            states.append(st1)
            entropies.append(float(self.gyroid_cov.estimate_entropy(st1).item()))
            windows.append(win1)

        # Fuse states entropy-weighted
            # Adjusted: Give more weight to higher entropy/novelty states
            inv = torch.tensor([max(e, 1e-6) for e in entropies], dtype=torch.float32)
        w = (inv / (inv.sum() + 1e-8)).tolist()
        fused = torch.zeros(1, self.dim, device=self.device)
        for wi, st in zip(w, states):
            fused += float(wi) * st
        fused = fused / (fused.norm() + 1e-8)

        # Average window modulation in residue space
        avg_window = None
        for win in windows:
            if win is None:
                continue
            if avg_window is None:
                avg_window = win.clone()
            else:
                avg_window += win
        if avg_window is not None:
            avg_window = avg_window / (avg_window.norm() + 1e-8)
            fused = self._apply_window_in_residue_space(fused, avg_window)

        entropy = self.gyroid_cov.estimate_entropy(fused)
        return {
            'state': fused,
            'entropy': float(entropy.item()),
            'component_entropies': entropies,
            'weights': w,
            'edge_density_l0': edge_density0
        }

    # -----------------------------
    # Low-level helpers for images/fields/windows
    # -----------------------------
    def _resize_long_edge(self, img, target: int):
        W, H = img.size
        if max(W, H) == target:
            return img
        if W >= H:
            new_W, new_H = target, max(1, int(H * (target / W)))
        else:
            new_H, new_W = target, max(1, int(W * (target / H)))
        return img.resize((new_W, new_H))

    def _to_tensor(self, img) -> torch.Tensor:
        import numpy as np
        arr = np.asarray(img).astype('float32') / 255.0
        t = torch.from_numpy(arr).to(self.device, non_blocking=True)
        if t.dim() == 3:
            t = t.permute(2, 0, 1)  # C,H,W
        return t

    def _compute_edge_field(self, img_chw: torch.Tensor) -> torch.Tensor:
        # Simple Sobel magnitude normalized to [0,1]
        C, H, W = img_chw.shape
        img_gray = img_chw.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=self.device).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=self.device).view(1,1,3,3)
        gx = torch.nn.functional.conv2d(img_gray, kx, padding=1)
        gy = torch.nn.functional.conv2d(img_gray, ky, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy).squeeze(0).squeeze(0)  # [H,W]
        mag = mag / (mag.max() + 1e-8)
        return mag

    def _compute_group_field(self, img_chw: torch.Tensor, edge_hw: torch.Tensor) -> torch.Tensor:
        # Approximate region coherence: blur edge map and invert to highlight smooth regions
        blur = torch.nn.functional.avg_pool2d(edge_hw.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()
        coherence = 1.0 - blur  # [H,W]
        coherence = torch.clamp(coherence, 0.0, 1.0)
        return coherence

    def _field_to_coeff_window(self, edge_hw: torch.Tensor, group_hw: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            # Reduce HxW to 1D residue_dim window using average over H and linear interp to dim
            edge_1d = edge_hw.mean(dim=0)  # [W]
            group_1d = group_hw.mean(dim=0)  # [W]
            win_1d = 0.6 * edge_1d + 0.4 * group_1d
            win_1d = (win_1d - win_1d.min()) / (win_1d.max() - win_1d.min() + 1e-8)
            # Interpolate to dim (acts as a proxy for residue_dim index window)
            src = win_1d.numel()
            idx = torch.linspace(0, src - 1, self.dim, device=self.device)
            x0 = torch.clamp(idx.floor().long(), 0, src - 1)
            x1 = torch.clamp(x0 + 1, 0, src - 1)
            t = (idx - x0.float())
            win = (1 - t) * win_1d[x0] + t * win_1d[x1]  # [dim]
            return win / (win.norm() + 1e-8)
        except Exception:
            return None

    def _apply_window_in_residue_space(self, state: torch.Tensor, window_dim: torch.Tensor, alpha: float = 0.2, beta: float = 0.8) -> torch.Tensor:
        # Pad/view to residues, apply window as smooth modulation, then flatten/truncate
        bsz = state.shape[0]
        state_dim = state.shape[1]
        base = state
        if state_dim % self.k != 0:
            pad_sz = self.k - (state_dim % self.k)
            base = F.pad(base, (0, pad_sz), mode='reflect')
        padded_dim = base.shape[1]
        residue_dim = padded_dim // self.k
        residues = base.view(bsz, self.k, residue_dim)
        # Build residue_dim window by resampling window_dim
        src = window_dim.numel()
        idx = torch.linspace(0, src - 1, residue_dim, device=self.device)
        x0 = torch.clamp(idx.floor().long(), 0, src - 1)
        x1 = torch.clamp(x0 + 1, 0, src - 1)
        t = (idx - x0.float())
        w_res = ((1 - t) * window_dim[x0] + t * window_dim[x1]).view(1, 1, residue_dim)
        mod = (alpha + beta * w_res)
        residues = residues * mod
        flat = residues.view(bsz, -1)
        if flat.shape[1] > state_dim:
            flat = flat[:, :state_dim]
        return flat / (flat.norm() + 1e-8)

    def _default_coeff_window_from_feature(self, feat: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            if feat.dim() >= 2:
                f = feat
                if f.dim() == 2:
                    f = f.unsqueeze(0)
                edge = self._compute_edge_field(f)
                group = self._compute_group_field(f, edge)
                return self._field_to_coeff_window(edge, group)
            return None
        except Exception:
            return None

