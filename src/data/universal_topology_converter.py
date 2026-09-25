import os
import hashlib
import math
import torch
from pathlib import Path
from typing import Dict, Any, Union, List

# Core Gyroidic Imports
from src.core.invariants import get_prime_ladder
from src.core.polynomial_coprime import PolynomialBasis
try:
    from src.core.honest_jitter import harvest_honest_jitter
except ImportError:
    harvest_honest_jitter = None

class UniversalTopologyConverter:
    """
    High-Dimensional Universal Topology Converter (Inspired by p2r3/convert).
    Extracts deep Spectral Tensors, Polynomial CRT Pressure Signatures, and 
    Defect Scout anomalies from arbitrary file types by reusing core Gyroidic systems
    (PolynomialBasis, get_prime_ladder, IVSTEncoder).
    
    Enforces the Output Boundary Policy (finite outputs, structural proxies only)
    while avoiding copyright infringement via non-obstructive embeddings.
    """

    def __init__(self, target_dim: int = 768, num_moduli: int = 5):
        self.target_dim = target_dim
        self.num_moduli = num_moduli
        
        # IVSTEncoder fallback for media
        try:
            from src.models.modular_embeddings import IVSTEncoder
            self.ivst_encoder = IVSTEncoder()
        except ImportError:
            self.ivst_encoder = None
            
        # 1. DYNAMIC PRIME GENERATION (Avoiding the hard-coded prime heresy)
        # Fetching primes dynamically aligned with MetaPolytopeMatrioshka
        prime_tensor = get_prime_ladder(self.num_moduli)
        self.primes = [int(p.item()) for p in prime_tensor]

        # 2. POLYNOMIAL BASIS INITIALIZATION
        # For mapping raw bytes to the high-dimensional phase space
        self.polynomial_basis = PolynomialBasis(degree=7, basis_type='chebyshev', domain=(-1.0, 1.0))

    def process_artifact(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found: {filepath}")

        ext = filepath.suffix.lower()
        
        # Media Delegation (IVSTEncoder)
        if ext in ['.mp4', '.mkv', '.avi', '.mp3', '.wav'] and self.ivst_encoder is not None:
            # Note: For full integration, you would pass the file to IVSTEncoder here
            pass 

        # 1. Compute Base Entropy Map (Raw Bytes -> Frequency Density)
        byte_map = self._compute_byte_histogram(filepath)
        
        # 2. Extract Spectral Tensor using PolynomialBasis [1, target_dim]
        spectral_tensor = self._compute_spectral_tensor(byte_map, filepath, ext)
        
        # 3. Compute CRT Pressure Signature using dynamic moduli
        pressure_signature = self._compute_crt_pressure(byte_map)
        
        # 4. Generate Defect Scout Anomalies (Homology Pressure proxy)
        defect_anomalies = self._scout_defects(byte_map)
        
        # 5. Initialize a low-magnitude Love Tensor stub
        love_tensor = torch.randn(1, self.target_dim) * 0.01

        # Universal Signature (Honest Jitter)
        if harvest_honest_jitter:
            jitter = harvest_honest_jitter((1,), device='cpu', scaled=True).item()
        else:
            jitter = 0.0
            
        signature = hashlib.sha256(f"{filepath.name}_{jitter}".encode()).hexdigest()

        topology_data = {
            "format": ext if ext else "unknown",
            "spectral_tensor": spectral_tensor,
            "pressure_signature": pressure_signature,
            "defect_anomalies": defect_anomalies,
            "love_tensor": love_tensor,
            "universal_signature": signature,
            "honesty_jitter": jitter
        }
        
        return topology_data

    def get_ui_summary(self, topology_data: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a JSON-serializable summary of the tensors for the frontend."""
        return {
            "format": topology_data["format"],
            "spectral_shape": list(topology_data["spectral_tensor"].shape),
            "spectral_l2_norm": round(torch.norm(topology_data["spectral_tensor"]).item(), 4),
            "crt_moduli": self.primes,
            "pressure_max": round(torch.max(topology_data["pressure_signature"]).item(), 4),
            "defect_count": int(torch.sum(torch.abs(topology_data["defect_anomalies"]) > 0.5).item()),
            "universal_signature": topology_data["universal_signature"],
            "honesty_jitter": round(topology_data["honesty_jitter"], 6)
        }

    def _compute_byte_histogram(self, filepath: Path) -> torch.Tensor:
        """Reads file in chunks and computes normalized byte frequency."""
        hist = torch.zeros(256, dtype=torch.float32)
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                chunk_tensor = torch.tensor(list(chunk), dtype=torch.long)
                hist += torch.bincount(chunk_tensor, minlength=256).float()
        
        total = hist.sum()
        if total > 0:
            hist = hist / total
        return hist

    def _compute_spectral_tensor(self, byte_map: torch.Tensor, filepath: Path, ext: str) -> torch.Tensor:
        """
        Projects raw structure into a high-dimensional Spectral Tensor [1, target_dim].
        Evaluates the PolynomialBasis over the byte frequency map.
        """
        # Normalize byte map to [-1, 1] for Chebyshev evaluation
        x_norm = 2.0 * byte_map - 1.0
        
        # Evaluate basis: returns [256, degree+1]
        basis_evals = self.polynomial_basis.evaluate(x_norm)
        
        # Flatten and project to target_dim (deterministic pseudo-random projection)
        file_size = filepath.stat().st_size
        struct_str = f"{file_size}_{ext}_{torch.std(byte_map).item()}"
        seed = int(hashlib.md5(struct_str.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        
        flattened_dim = 256 * (self.polynomial_basis.degree + 1)
        projection_matrix = torch.randn(flattened_dim, self.target_dim) / math.sqrt(flattened_dim)
        
        spectral_tensor = torch.matmul(basis_evals.view(1, -1), projection_matrix)
        
        return torch.tanh(spectral_tensor)

    def _compute_crt_pressure(self, byte_map: torch.Tensor) -> torch.Tensor:
        """
        Computes the CRT (Chinese Remainder Theorem) Pressure Signature.
        Maps the byte map to dynamically-fetched co-prime bases.
        Returns a tensor of shape [1, num_moduli].
        """
        pressure = torch.zeros(1, self.num_moduli)
        for i, p in enumerate(self.primes):
            scaled_map = byte_map * 255.0 * p
            remainder = scaled_map - p * torch.floor(scaled_map / p)
            pressure[0, i] = torch.mean(remainder) / p
            
        return pressure

    def _scout_defects(self, byte_map: torch.Tensor) -> torch.Tensor:
        """
        Defect Scout: Identifies sparse anomalies and amplifies them.
        Returns a tensor of shape [1, 256].
        """
        mean_freq = torch.mean(byte_map)
        std_freq = torch.std(byte_map)
        
        z_scores = (byte_map - mean_freq) / (std_freq + 1e-8)
        defects = torch.where(z_scores < -1.0, -z_scores, torch.zeros_like(z_scores))
        
        return torch.tanh(defects).unsqueeze(0)


if __name__ == "__main__":
    converter = UniversalTopologyConverter()
    print("Universal Topology Converter (Tensor Extraction via Gyroidic Primitives) Ready.")
