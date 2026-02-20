"""
Non-Abelian Gyroidic Image Codec.

Encodes text↔image pairs into a topologically structured CRT residue space,
where the encoding path matters (non-commutative) and the residue captures
irreducible text-image entanglement.

Mathematical Formulation:
    E(T, I) = CRT({R_k(T) · G_k(I)}_{k=1..K})

    where:
        R_k = text residue projections (polynomial CRT decomposition)
        G_k = gyroidic image projections (gyroid surface sampling at polynomial frequencies)
        R_k · G_k is non-commutative (matrix multiplication in GL(n))

    Residue = E(T, I) - CRT_inv({R_k(T)}) ⊗ CRT_inv({G_k(I)})
    Non-zero residue = irreducible entanglement between text and image.

Uses only PyTorch (no Pillow, no OpenCL). Integrates with:
    - PolynomialCoprimeConfig for basis functions, Birkhoff polytope, and frequency generation
    - PolynomialCRT for reconstruction (no reimplementation)
    - Gyroid surface evaluation from analytic functions

Compliance:
    - No hardcoded primes (Governance Rule 13, SYSTEM_ARCHITECTURE Rule 1)
    - PolynomialCoprimeConfig mandatory (Governance Rule 14)
    - Uses existing PolynomialCRT (no reimplementation)
    - Chinese Room doctrine: structural entanglement only, no comprehension claims (PHILOSOPHY §6)

Author: William Matthew Bryant
Created: February 2026
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.polynomial_crt import PolynomialCRT


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CodecConfig:
    """Configuration for the gyroidic codec."""
    K: int = 5                   # Number of CRT residue channels
    n: int = 8                   # Matrix dimension for GL(n) operations
    degree: int = 4              # Polynomial degree for basis functions
    basis_type: str = 'chebyshev'
    gyroid_resolution: int = 32  # Spatial resolution for gyroid sampling
    gyroid_periods: int = 2      # Number of gyroid periods in each dimension
    text_embed_dim: int = 64     # Dimension for text character embeddings
    use_saturation: bool = True  # Apply piecewise saturation gates
    device: str = 'cpu'


@dataclass
class EncodingResult:
    """Result of encoding a text-image pair."""
    encoded: torch.Tensor          # [K, n, n] — the combined CRT encoding
    text_residues: torch.Tensor    # [K, n, n] — R_k(T) per channel
    image_residues: torch.Tensor   # [K, n, n] — G_k(I) per channel
    residue: torch.Tensor          # [n, n] — irreducible entanglement
    commutativity_gap: float       # ||AB - BA|| for the encoding
    diagnostics: Dict = field(default_factory=dict)


# =============================================================================
# Gyroid Surface Evaluation
# =============================================================================

class GyroidSurface:
    """
    Analytic gyroid surface: a triply-periodic minimal surface.

    The gyroid level set is:
        G(x, y, z) = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)

    Zero level set G = 0 defines the minimal surface. We sample at
    polynomial-derived frequencies (via PolynomialCoprimeConfig) to create
    K independent channels. No hardcoded primes — frequencies emerge from
    polynomial basis evaluation.
    """

    def __init__(
        self,
        resolution: int = 32,
        periods: int = 2,
        poly_config: Optional[PolynomialCoprimeConfig] = None,
        device: str = 'cpu'
    ):
        self.resolution = resolution
        self.periods = periods
        self.device = device
        self.poly_config = poly_config

        # Create 3D grid
        coords = torch.linspace(0, 2 * math.pi * periods, resolution, device=device)
        self.xx, self.yy, self.zz = torch.meshgrid(coords, coords, coords, indexing='ij')

    def _get_frequency(self, k: int) -> float:
        """
        Derive channel frequency from polynomial evaluation.

        Instead of hardcoded primes, we evaluate the k-th polynomial functional
        at a canonical point and use the absolute value (shifted to be > 1) as
        the frequency multiplier. This ensures incommensurate frequencies
        emerge from the Birkhoff-constrained polynomial basis.
        """
        if self.poly_config is not None:
            # Evaluate φ_k at canonical point x=0.5 (center of normalized domain)
            x_canonical = torch.tensor([0.5], device=self.device)
            with torch.no_grad():
                phi_k = self.poly_config.evaluate_polynomial(k, x_canonical)
            # Map to positive frequency: |φ_k| + 1.5 to ensure > 1 and distinct
            freq = abs(phi_k.item()) + 1.5 + k * 0.7
        else:
            # Fallback: irrational frequency spacing (still not hardcoded primes)
            # Uses golden-ratio-derived incommensurate frequencies
            phi = (1 + math.sqrt(5)) / 2  # Golden ratio
            freq = 1.0 + k * phi
        return freq

    def evaluate(self, k: int) -> torch.Tensor:
        """
        Evaluate gyroid at polynomial-derived frequency for channel k.

        G_k(x,y,z) = sin(f_k·x)cos(f_k·y) + sin(f_k·y)cos(f_k·z) + sin(f_k·z)cos(f_k·x)

        Args:
            k: Channel index (0..K-1), frequency derived from PolynomialCoprimeConfig

        Returns:
            field: [resolution, resolution, resolution] gyroid field values
        """
        f = self._get_frequency(k)
        return (
            torch.sin(f * self.xx) * torch.cos(f * self.yy) +
            torch.sin(f * self.yy) * torch.cos(f * self.zz) +
            torch.sin(f * self.zz) * torch.cos(f * self.xx)
        )

    def evaluate_2d_slice(self, k: int, z_val: float = 0.0) -> torch.Tensor:
        """
        Evaluate a 2D slice of the gyroid at fixed z.

        Args:
            k: Channel index
            z_val: Fixed z coordinate

        Returns:
            slice: [resolution, resolution] 2D gyroid slice
        """
        f = self._get_frequency(k)
        coords = torch.linspace(0, 2 * math.pi * self.periods, self.resolution, device=self.device)
        xx, yy = torch.meshgrid(coords, coords, indexing='ij')
        z = torch.tensor(z_val, device=self.device)
        return (
            torch.sin(f * xx) * torch.cos(f * yy) +
            torch.sin(f * yy) * torch.cos(f * z) +
            torch.sin(f * z) * torch.cos(f * xx)
        )

    def mean_curvature_estimate(self, k: int) -> torch.Tensor:
        """
        Estimate mean curvature of the gyroid at channel k.

        Uses finite differences on the level-set function:
            H ≈ -div(∇G / |∇G|)

        Returns:
            curvature: [resolution, resolution, resolution]
        """
        G = self.evaluate(k)
        # Central differences
        dx = torch.roll(G, -1, 0) - torch.roll(G, 1, 0)
        dy = torch.roll(G, -1, 1) - torch.roll(G, 1, 1)
        dz = torch.roll(G, -1, 2) - torch.roll(G, 1, 2)

        grad_norm = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)
        nx, ny, nz = dx / grad_norm, dy / grad_norm, dz / grad_norm

        # Divergence of unit normal
        div_n = (
            (torch.roll(nx, -1, 0) - torch.roll(nx, 1, 0)) +
            (torch.roll(ny, -1, 1) - torch.roll(ny, 1, 1)) +
            (torch.roll(nz, -1, 2) - torch.roll(nz, 1, 2))
        )
        return -0.5 * div_n


# =============================================================================
# Text Residue Projector
# =============================================================================

class TextResidueProjector(nn.Module):
    """
    Project text into K residue matrices in GL(n).

    T → {R_k(T)}_{k=1..K}, where each R_k ∈ GL(n).

    Process:
        1. Character-level embedding → [len, embed_dim]
        2. Polynomial CRT decomposition into K channels via PolynomialCoprimeConfig
        3. Each channel reshaped to [n, n] matrix
        4. Projected to GL(n) via matrix exponential
    """

    def __init__(self, config: CodecConfig, poly_config: PolynomialCoprimeConfig):
        super().__init__()
        self.config = config
        self.n = config.n
        self.K = config.K
        self.poly_config = poly_config

        # Character embedding (ASCII 0-127 + padding)
        self.char_embed = nn.Embedding(128, config.text_embed_dim, padding_idx=0)

        # Per-channel projectors: embed_dim → n*n
        self.channel_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.text_embed_dim, config.n * config.n),
                nn.Tanh()
            )
            for _ in range(config.K)
        ])

    def forward(self, text: str) -> torch.Tensor:
        """
        Project text to K residue matrices.

        Args:
            text: Input string

        Returns:
            residues: [K, n, n] — each R_k ∈ GL(n)
        """
        device = self.config.device

        # 1. Encode text as character indices
        char_ids = torch.tensor(
            [ord(c) % 128 for c in text], dtype=torch.long, device=device
        )
        if len(char_ids) == 0:
            char_ids = torch.tensor([0], dtype=torch.long, device=device)

        # 2. Character embeddings → pooled representation
        embeddings = self.char_embed(char_ids)  # [len, embed_dim]
        pooled = embeddings.mean(dim=0)          # [embed_dim]

        # 3. Per-channel projection with polynomial modulation
        residues = []
        for k in range(self.K):
            # Position-dependent modulation via PolynomialCoprimeConfig
            t_norm = torch.tensor([len(text) / 1000.0], device=device).clamp(-1, 1)
            phi_k = self.poly_config.evaluate_polynomial(k, t_norm)
            modulated = pooled * (1.0 + 0.1 * phi_k)

            # Project to n×n matrix
            flat_matrix = self.channel_projectors[k](modulated)  # [n*n]
            matrix = flat_matrix.view(self.n, self.n)

            # Project to GL(n) via matrix exponential
            # exp(A) is always invertible → guarantees GL(n) membership
            gl_matrix = torch.matrix_exp(matrix)
            residues.append(gl_matrix)

        return torch.stack(residues)  # [K, n, n]


# =============================================================================
# Gyroid Image Projector
# =============================================================================

class GyroidImageProjector(nn.Module):
    """
    Project an image (or image-like tensor) into K residue matrices via
    gyroid surface sampling at polynomial-derived frequencies.

    I → {G_k(I)}_{k=1..K}, where each G_k ∈ GL(n).

    Process:
        1. Evaluate gyroid surface at polynomial frequency k
        2. Convolve 2D slice of gyroid with image features
        3. Reshape to [n, n] and project to GL(n) via matrix exponential
    """

    def __init__(self, config: CodecConfig, poly_config: PolynomialCoprimeConfig):
        super().__init__()
        self.config = config
        self.n = config.n
        self.K = config.K
        self.gyroid = GyroidSurface(
            config.gyroid_resolution,
            config.gyroid_periods,
            poly_config=poly_config,
            device=config.device
        )

        # Per-channel projection from gyroid features to GL(n)
        self.channel_projectors = nn.ModuleList([
            nn.Linear(config.n * config.n, config.n * config.n)
            for _ in range(config.K)
        ])

    def forward(self, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Project image to K residue matrices.

        Args:
            image: Optional [H, W] or [C, H, W] image tensor.
                   If None, uses the gyroid field itself as the "image"
                   (self-referential encoding).

        Returns:
            residues: [K, n, n] — each G_k ∈ GL(n)
        """
        residues = []
        for k in range(self.K):
            # 1. Get 2D gyroid slice at channel k
            gyroid_slice = self.gyroid.evaluate_2d_slice(k)  # [res, res]

            # 2. Modulate with image if provided
            if image is not None:
                img_2d = self._prepare_image(image)  # [res, res]
                modulated = gyroid_slice * img_2d
            else:
                modulated = gyroid_slice

            # 3. Downsample to n×n via adaptive average
            downsampled = self._adaptive_pool(modulated, self.n)  # [n, n]

            # 4. Project through learned layer
            flat = downsampled.reshape(-1)  # [n*n]
            projected = self.channel_projectors[k](flat)  # [n*n]
            matrix = projected.view(self.n, self.n)

            # 5. Project to GL(n) via matrix exponential
            gl_matrix = torch.matrix_exp(matrix)
            residues.append(gl_matrix)

        return torch.stack(residues)  # [K, n, n]

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        """Resize/reshape image to match gyroid resolution."""
        # Handle channel dimension
        if image.dim() == 3:
            image = image.mean(dim=0)  # Average over channels → [H, W]

        # Resize to gyroid resolution
        res = self.config.gyroid_resolution
        if image.shape != (res, res):
            image = nn.functional.interpolate(
                image.unsqueeze(0).unsqueeze(0).float(),
                size=(res, res),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        return image

    def _adaptive_pool(self, tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        """Downsample 2D tensor to target_size × target_size."""
        return nn.functional.adaptive_avg_pool2d(
            tensor.unsqueeze(0).unsqueeze(0),
            (target_size, target_size)
        ).squeeze(0).squeeze(0)


# =============================================================================
# Non-Abelian Combiner
# =============================================================================

class NonAbelianCombiner:
    """
    Combine text and image residues via non-commutative matrix multiplication.

    E(T, I) = CRT({R_k(T) · G_k(I)}_{k=1..K})

    The product R_k · G_k is matrix multiplication in GL(n), which is
    NON-COMMUTATIVE: R_k · G_k ≠ G_k · R_k in general.

    This means:
        encode(text, image) ≠ encode(image, text)
        The encoding path MATTERS.
    """

    @staticmethod
    def combine(
        text_residues: torch.Tensor,
        image_residues: torch.Tensor
    ) -> torch.Tensor:
        """
        Non-abelian combination: R_k(T) · G_k(I) for each channel k.

        Args:
            text_residues:  [K, n, n]
            image_residues: [K, n, n]

        Returns:
            combined: [K, n, n] — R_k · G_k (matrix product per channel)
        """
        return torch.bmm(text_residues, image_residues)

    @staticmethod
    def combine_reverse(
        text_residues: torch.Tensor,
        image_residues: torch.Tensor
    ) -> torch.Tensor:
        """
        Reversed combination: G_k(I) · R_k(T).

        If non-abelian, this differs from combine().
        """
        return torch.bmm(image_residues, text_residues)

    @staticmethod
    def commutativity_gap(
        text_residues: torch.Tensor,
        image_residues: torch.Tensor
    ) -> float:
        """
        Measure non-commutativity: ||R·G - G·R||_F / ||R·G||_F

        Returns 0 if commutative (abelian), >0 if non-commutative.
        """
        forward = torch.bmm(text_residues, image_residues)
        reverse = torch.bmm(image_residues, text_residues)
        gap = torch.norm(forward - reverse).item()
        scale = torch.norm(forward).item() + 1e-8
        return gap / scale


# =============================================================================
# CRT Reconstruction Bridge
# =============================================================================

class CodecCRTBridge:
    """
    Bridge between codec's [K, n, n] channel matrices and the
    existing PolynomialCRT reconstruction system.

    Instead of reimplementing CRT, we reshape the codec's matrix channels
    into the [batch, K, D] residue distribution format expected by
    PolynomialCRT.forward(), delegate reconstruction, then reshape back.

    This ensures the codec uses the project's canonical CRT implementation
    with proper majority-symbol and modal consensus reconstruction.
    """

    def __init__(self, poly_config: PolynomialCoprimeConfig):
        self.crt = PolynomialCRT(poly_config, use_soft_reconstruction=True)

    def reconstruct(self, channel_matrices: torch.Tensor) -> torch.Tensor:
        """
        CRT reconstruction from K channel matrices via PolynomialCRT.

        Args:
            channel_matrices: [K, n, n]

        Returns:
            reconstructed: [n, n]
        """
        K, n, _ = channel_matrices.shape

        # Reshape [K, n, n] → [n, K, n] to treat each row as a "batch" sample
        # with K residue channels of dimension n
        reshaped = channel_matrices.permute(1, 0, 2)  # [n, K, n]

        # Delegate to canonical PolynomialCRT
        reconstructed = self.crt.forward(
            reshaped,
            mode='majority',
            return_diagnostics=False
        )  # [n, n]

        return reconstructed

    def reconstruct_inverse(self, channel_matrices: torch.Tensor) -> torch.Tensor:
        """
        CRT-inverse: reconstruct from per-channel matrices independently.
        Used for computing the separable component.

        Uses PolynomialCRT with modal consensus for stability.

        Args:
            channel_matrices: [K, n, n]

        Returns:
            separable: [n, n]
        """
        K, n, _ = channel_matrices.shape
        reshaped = channel_matrices.permute(1, 0, 2)  # [n, K, n]

        reconstructed = self.crt.forward(
            reshaped,
            mode='modal',
            return_diagnostics=False
        )  # [n, n]

        return reconstructed


# =============================================================================
# Residue Extractor
# =============================================================================

class ResidueExtractor:
    """
    Extract the irreducible text-image residue.

    Residue = E(T,I) - CRT_inv({R_k(T)}) ⊗ CRT_inv({G_k(I)})

    Non-zero residue means text and image are ENTANGLED — there is
    structure in the joint encoding that cannot be decomposed into
    independent "text part" and "image part."

    The ⊗ is outer product of the two independently-reconstructed
    matrices, projected back to [n, n].
    """

    @staticmethod
    def extract(
        combined_encoding: torch.Tensor,
        text_residues: torch.Tensor,
        image_residues: torch.Tensor,
        reconstructor: CodecCRTBridge
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the irreducible residue.

        Args:
            combined_encoding: [n, n] — E(T, I) 
            text_residues:  [K, n, n]
            image_residues: [K, n, n]
            reconstructor: CRT bridge instance

        Returns:
            residue: [n, n] — the irreducible entanglement
            diagnostics: dict with residue metrics
        """
        # Reconstruct text-only and image-only
        text_only = reconstructor.reconstruct_inverse(text_residues)    # [n, n]
        image_only = reconstructor.reconstruct_inverse(image_residues)  # [n, n]

        # Separable component: text ⊗ image (matrix multiplication as outer product proxy)
        separable = torch.mm(text_only, image_only)  # [n, n]

        # Residue: what CANNOT be separated
        residue = combined_encoding - separable

        # Diagnostics
        residue_norm = torch.norm(residue).item()
        encoding_norm = torch.norm(combined_encoding).item()
        separable_norm = torch.norm(separable).item()

        # Entanglement ratio: how much of the encoding is irreducible
        entanglement_ratio = residue_norm / (encoding_norm + 1e-8)

        # Spectral analysis of residue
        svd_vals = torch.linalg.svdvals(residue)
        spectral_entropy = -(
            (svd_vals / (svd_vals.sum() + 1e-8)) *
            torch.log(svd_vals / (svd_vals.sum() + 1e-8) + 1e-8)
        ).sum().item()

        diagnostics = {
            'residue_norm': residue_norm,
            'encoding_norm': encoding_norm,
            'separable_norm': separable_norm,
            'entanglement_ratio': entanglement_ratio,
            'spectral_entropy': spectral_entropy,
            'rank_estimate': (svd_vals > 1e-6).sum().item(),
        }

        return residue, diagnostics


# =============================================================================
# Structural Entanglement Gate
# =============================================================================

class StructuralEntanglementGate:
    """
    Structural Entanglement Gate: measures cross-modal residue structure.

    Per the Chinese Room Doctrine (PHILOSOPHY.md §6), this gate does NOT
    claim to assess "understanding." It measures structural entanglement —
    the irreducible topological residue between modalities. Whether this
    constitutes comprehension is a category error; we only report
    admissibility of the encoding's structural coherence.

    The gate acts as an ADMISSIBILITY FILTER:
        - Admissible: sufficient cross-modal structure exists
        - Inadmissible: encoding is separable (no cross-modal structure)
    """

    def __init__(self, entanglement_threshold: float = 0.1):
        """
        Args:
            entanglement_threshold: Minimum entanglement ratio for admissibility.
                Below this, the encoding is structurally separable.
        """
        self.threshold = entanglement_threshold

    def evaluate(
        self,
        residue: torch.Tensor,
        diagnostics: Dict[str, float]
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Evaluate structural admissibility of the encoding.

        Args:
            residue: [n, n] irreducible residue
            diagnostics: From ResidueExtractor

        Returns:
            gate_result: dict with admissibility verdict and structural metrics
        """
        entanglement = diagnostics.get('entanglement_ratio', 0.0)
        spectral_entropy = diagnostics.get('spectral_entropy', 0.0)
        rank = diagnostics.get('rank_estimate', 0)

        is_admissible = entanglement > self.threshold

        # Structural interpretation (no comprehension claims)
        if entanglement < 0.01:
            structural_state = "Separable: modalities encode independently (no cross-modal structure)"
        elif entanglement < self.threshold:
            structural_state = "Weakly entangled: minimal cross-modal residue structure"
        elif entanglement < 0.5:
            structural_state = "Entangled: significant irreducible cross-modal structure"
        else:
            structural_state = "Strongly entangled: deep cross-modal structural coherence"

        return {
            'is_admissible': is_admissible,
            'entanglement_ratio': entanglement,
            'spectral_entropy': spectral_entropy,
            'residue_rank': rank,
            'structural_state': structural_state,
        }


# =============================================================================
# Main Codec: GyroidicCodec
# =============================================================================

class GyroidicCodec(nn.Module):
    """
    Non-Abelian Gyroidic Image Codec.

    Full pipeline:
        1. Text → R_k(T) via polynomial CRT decomposition
        2. Image → G_k(I) via gyroid surface sampling
        3. Combine: E_k = R_k · G_k (non-commutative)
        4. Reconstruct: E = CRT({E_k}) via PolynomialCRT
        5. Extract residue: Res = E - CRT_inv(R) ⊗ CRT_inv(G)
        6. Structural Entanglement Gate: admissibility of cross-modal structure

    Usage:
        codec = GyroidicCodec(CodecConfig(K=5, n=8))
        result = codec.encode("hello world")
        result = codec.encode("hello world", image_tensor)
        gap = codec.verify_non_commutativity("hello", image)
    """

    def __init__(self, config: Optional[CodecConfig] = None):
        super().__init__()
        self.config = config or CodecConfig()

        # Create PolynomialCoprimeConfig — mandatory, no silent fallback
        self.poly_config = PolynomialCoprimeConfig(
            k=self.config.K,
            degree=self.config.degree,
            basis_type=self.config.basis_type,
            learnable=False,  # Codec uses fixed polynomial basis
            use_saturation=self.config.use_saturation,
            device=self.config.device,
        )

        # Core components
        self.text_projector = TextResidueProjector(self.config, self.poly_config)
        self.image_projector = GyroidImageProjector(self.config, self.poly_config)
        self.combiner = NonAbelianCombiner()
        self.crt_bridge = CodecCRTBridge(self.poly_config)
        self.residue_extractor = ResidueExtractor()
        self.entanglement_gate = StructuralEntanglementGate()

    def encode(
        self,
        text: str,
        image: Optional[torch.Tensor] = None
    ) -> EncodingResult:
        """
        Encode a text-image pair.

        Args:
            text: Input text string
            image: Optional image tensor [H,W] or [C,H,W].
                   If None, uses gyroid self-reference.

        Returns:
            EncodingResult with full encoding diagnostics
        """
        # 1. Text → residues
        text_residues = self.text_projector(text)      # [K, n, n]

        # 2. Image → residues
        image_residues = self.image_projector(image)    # [K, n, n]

        # 3. Non-abelian combination
        combined_channels = self.combiner.combine(text_residues, image_residues)  # [K, n, n]

        # 4. CRT reconstruction via PolynomialCRT
        encoded = self.crt_bridge.reconstruct(combined_channels)  # [n, n]

        # 5. Extract residue
        residue, res_diagnostics = self.residue_extractor.extract(
            encoded, text_residues, image_residues, self.crt_bridge
        )

        # 6. Commutativity gap
        comm_gap = self.combiner.commutativity_gap(text_residues, image_residues)

        # 7. Structural Entanglement Gate (admissibility, not comprehension)
        gate_result = self.entanglement_gate.evaluate(residue, res_diagnostics)

        # Combine all diagnostics
        diagnostics = {
            **res_diagnostics,
            **gate_result,
            'commutativity_gap': comm_gap,
        }

        return EncodingResult(
            encoded=encoded,
            text_residues=text_residues,
            image_residues=image_residues,
            residue=residue,
            commutativity_gap=comm_gap,
            diagnostics=diagnostics,
        )

    def verify_non_commutativity(
        self,
        text: str,
        image: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Verify that the encoding is genuinely non-commutative.

        Computes E(T,I) and E(I,T) (reversed order) and measures
        the difference. A non-zero gap proves non-abelian structure.

        Args:
            text: Input text
            image: Optional image tensor

        Returns:
            Dict with commutativity metrics
        """
        text_residues = self.text_projector(text)
        image_residues = self.image_projector(image)

        # Forward: R · G
        forward = self.combiner.combine(text_residues, image_residues)
        forward_recon = self.crt_bridge.reconstruct(forward)

        # Reverse: G · R
        reverse = self.combiner.combine_reverse(text_residues, image_residues)
        reverse_recon = self.crt_bridge.reconstruct(reverse)

        # Metrics
        gap = self.combiner.commutativity_gap(text_residues, image_residues)
        recon_gap = torch.norm(forward_recon - reverse_recon).item()
        determinant_diff = abs(
            torch.det(forward_recon).item() - torch.det(reverse_recon).item()
        )

        return {
            'channel_gap': gap,
            'reconstruction_gap': recon_gap,
            'determinant_diff': determinant_diff,
            'is_non_abelian': gap > 1e-6,
        }

    def encode_batch(
        self,
        texts: List[str],
        images: Optional[List[torch.Tensor]] = None
    ) -> List[EncodingResult]:
        """Encode multiple text-image pairs."""
        results = []
        for i, text in enumerate(texts):
            image = images[i] if images and i < len(images) else None
            results.append(self.encode(text, image))
        return results

    def compute_mutual_information_proxy(
        self,
        result: EncodingResult
    ) -> float:
        """
        Estimate mutual information between text and image modalities
        via the residue structure.

        MI ∝ log(1 + entanglement_ratio) × spectral_entropy(residue)

        This is a structural proxy, not true MI — but correlates with
        cross-modal structure that cannot be factored.
        """
        ent = result.diagnostics.get('entanglement_ratio', 0.0)
        se = result.diagnostics.get('spectral_entropy', 0.0)
        return math.log1p(ent) * se


# =============================================================================
# Convenience factory
# =============================================================================

def create_codec(
    K: int = 5,
    n: int = 8,
    resolution: int = 32,
    device: str = 'cpu'
) -> GyroidicCodec:
    """
    Create a GyroidicCodec with sensible defaults.

    Args:
        K: Number of CRT channels (more = richer encoding)
        n: Matrix dimension (larger = more expressive GL(n))
        resolution: Gyroid sampling resolution
        device: PyTorch device

    Returns:
        Configured GyroidicCodec instance
    """
    config = CodecConfig(K=K, n=n, gyroid_resolution=resolution, device=device)
    return GyroidicCodec(config)
