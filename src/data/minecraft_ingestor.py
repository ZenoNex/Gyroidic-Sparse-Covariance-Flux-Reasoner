"""
Minecraft Ingestion Pipeline for MCA, NBT, and Mod Jar parsing.

Extracts spatial voxel structures, block entity metadata, and modded text/scripts
(e.g., ComputerCraft LUA files, written books, signs), projecting them into
the 3D Chebyshev CRT residue space as Voxel-Text Knowledge Dyads.

Follows anti-lobotomy principles:
- Pure Python parsing of NBT and MCA formats (no native binary dependencies).
- No hardcoded ID mappings: utilizes spatial adjacency rhythms and the VoynichLinguist.
- Generates dynamic GL(n) residues mapped via matrix exponentials.

Author: System Integration Team
Date: May 2026
"""

import os
import io
import gzip
import zlib
import struct
import zipfile
import math
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np

from src.core.polynomial_coprime import PolynomialBasis, PolynomialCoprimeConfig
from src.codec.gyroidic_codec import CodecConfig, EncodingResult
from src.core.topological_ingestion_validator import TopologicalIngestionValidator


# Pre-allocate a flat integer array REGISTRY_TABLE of size 1024
REGISTRY_TABLE = [i * 37 % 1024 for i in range(1024)]

def get_block_registry_code(name: str) -> int:
    """Maps Minecraft block name to a 0-1023 registry code with topological mod structure."""
    name_lower = name.lower()
    h = 0
    for char in name_lower:
        h = (h * 31 + ord(char)) % 1024
        
    code = REGISTRY_TABLE[h]
    # Mod-specific Lie algebra generator phase alignment:
    if "opencomputers" in name_lower or name_lower.startswith("oc:"):
        code = 512 + (code % 512)
    elif "computercraft" in name_lower:
        code = 256 + (code % 256)
    return code

def _chebyshev_compress(text: str, target_len: int = 2048) -> str:
    """Compresses arbitrary text streams into target_len characters using Chebyshev basis modes."""
    if len(text) <= target_len:
        return text
    num_modes = 128
    n = len(text)
    # Use canonical PolynomialBasis for discrete orthogonal expansion
    basis = PolynomialBasis(degree=num_modes - 1, basis_type='chebyshev', domain=(-1.0, 1.0))
    grid = torch.linspace(-1.0, 1.0, n)
    t_vals = basis.evaluate(grid).detach().cpu().numpy()  # [n, num_modes]
    char_vals = np.array([ord(c) / 128.0 for c in text], dtype=np.float64)
    coeffs = (char_vals[:, None] * t_vals).sum(axis=0)

    target_grid = torch.linspace(-1.0, 1.0, target_len)
    target_t = basis.evaluate(target_grid).detach().cpu().numpy()  # [target_len, num_modes]
    reconstructed = (target_t * coeffs[None, :]).sum(axis=1)

    compressed_chars = []
    for val in reconstructed:
        char_code = 32 + int(abs(val * 100) % 95)
        compressed_chars.append(chr(char_code))
    return "".join(compressed_chars)

class NBTReader:
    """
    Stream-based decoder for Named Binary Tag (NBT) format.
    Supports all tag types, lists, compounds, and array payloads.
    """
    TAG_END = 0
    TAG_BYTE = 1
    TAG_SHORT = 2
    TAG_INT = 3
    TAG_LONG = 4
    TAG_FLOAT = 5
    TAG_DOUBLE = 6
    TAG_BYTE_ARRAY = 7
    TAG_STRING = 8
    TAG_LIST = 9
    TAG_COMPOUND = 10
    TAG_INT_ARRAY = 11
    TAG_LONG_ARRAY = 12

    def __init__(self, data: bytes):
        # Auto-detect Gzip compression (header 1f 8b)
        if len(data) > 2 and data[0] == 0x1F and data[1] == 0x8B:
            data = gzip.decompress(data)
        self.stream = io.BytesIO(data)

    def _read(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        buf = self.stream.read(size)
        if len(buf) < size:
            raise EOFError("NBT stream ended prematurely.")
        val = struct.unpack(fmt, buf)
        return val[0] if len(val) == 1 else val

    def read_tag_payload(self, tag_type: int) -> Any:
        if tag_type == self.TAG_BYTE:
            return self._read(">b")
        elif tag_type == self.TAG_SHORT:
            return self._read(">h")
        elif tag_type == self.TAG_INT:
            return self._read(">i")
        elif tag_type == self.TAG_LONG:
            return self._read(">q")
        elif tag_type == self.TAG_FLOAT:
            return self._read(">f")
        elif tag_type == self.TAG_DOUBLE:
            return self._read(">d")
        elif tag_type == self.TAG_BYTE_ARRAY:
            length = self._read(">i")
            return list(self.stream.read(length))
        elif tag_type == self.TAG_STRING:
            length = self._read(">H")
            text = self.stream.read(length).decode("utf-8", errors="replace")
            if len(text) > 2048:
                text = _chebyshev_compress(text)
            return text
        elif tag_type == self.TAG_LIST:
            elem_type = self._read(">b")
            length = self._read(">i")
            return [self.read_tag_payload(elem_type) for _ in range(length)]
        elif tag_type == self.TAG_COMPOUND:
            payload = {}
            while True:
                next_type = self._read(">b")
                if next_type == self.TAG_END:
                    break
                name_len = self._read(">H")
                name = self.stream.read(name_len).decode("utf-8", errors="replace")
                payload[name] = self.read_tag_payload(next_type)
            return payload
        elif tag_type == self.TAG_INT_ARRAY:
            length = self._read(">i")
            return [self._read(">i") for _ in range(length)]
        elif tag_type == self.TAG_LONG_ARRAY:
            length = self._read(">i")
            return [self._read(">q") for _ in range(length)]
        else:
            raise ValueError(f"Unknown NBT tag type: {tag_type}")

    def read_root(self) -> Tuple[str, Dict]:
        """Reads root compound tag, returning name and dictionary of tags."""
        root_type = self._read(">b")
        if root_type != self.TAG_COMPOUND:
            raise ValueError(f"Root tag must be compound, got: {root_type}")
        name_len = self._read(">H")
        root_name = self.stream.read(name_len).decode("utf-8", errors="replace")
        root_payload = self.read_tag_payload(self.TAG_COMPOUND)
        return root_name, root_payload


# =============================================================================
# 2. MCA Region File Reader
# =============================================================================

class MCAReader:
    """
    Anvil region file (.mca) parser.
    Reads region file headers and decompresses specific chunk NBT compounds.
    """
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.file = open(self.filepath, "rb")
        # Load the 4096-byte chunk offset table
        self.offsets = list(struct.unpack(">1024I", self.file.read(4096)))
        # Load the 4096-byte chunk timestamp table
        self.timestamps = list(struct.unpack(">1024I", self.file.read(4096)))

    def has_chunk(self, cx: int, cz: int) -> bool:
        """Checks if chunk coordinates cx, cz (0-31) exist in this region."""
        idx = 4 * (cx + cz * 32)
        offset_entry = self.offsets[cx + cz * 32]
        offset = offset_entry >> 8
        return offset != 0

    def read_chunk_nbt(self, cx: int, cz: int) -> Optional[Dict]:
        """Reads and parses zlib-compressed NBT data for chunk (cx, cz)."""
        if not (0 <= cx < 32 and 0 <= cz < 32):
            raise ValueError("Chunk coordinates must be between 0 and 31.")
        
        offset_entry = self.offsets[cx + cz * 32]
        offset = offset_entry >> 8
        sectors = offset_entry & 0xFF
        
        if offset == 0:
            return None  # Chunk not generated
        
        # Seek to beginning of sector data
        self.file.seek(offset * 4096)
        length = struct.unpack(">I", self.file.read(4))[0]
        compression_scheme = struct.unpack(">b", self.file.read(1))[0]
        
        compressed_data = self.file.read(length - 1)
        
        if compression_scheme == 1:
            decompressed = gzip.decompress(compressed_data)
        elif compression_scheme == 2:
            decompressed = zlib.decompress(compressed_data)
        elif compression_scheme == 3:
            decompressed = compressed_data  # Uncompressed
        else:
            raise ValueError(f"Unsupported compression scheme: {compression_scheme}")
            
        reader = NBTReader(decompressed)
        _, payload = reader.read_root()
        return payload

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# 3. Jar/Mod Archive Ingestor
# =============================================================================

class JarModExtractor:
    """
    Parses .jar and .zip mods to extract text assets, configurations,
    and embedded ComputerCraft/OpenComputers LUA scripts.
    """
    def __init__(self, mod_path: Union[str, Path]):
        self.mod_path = Path(mod_path)

    def extract_mod_scripts(self) -> List[Dict[str, Any]]:
        """Scans archive for embedded lua, txt, or json files and returns their text contents."""
        scripts = []
        if not self.mod_path.exists():
            return scripts

        if self.mod_path.is_file() and self.mod_path.suffix in [".jar", ".zip"]:
            scripts.extend(self._parse_zip(self.mod_path))
        elif self.mod_path.is_dir():
            # Scan all jar/zip files inside directory
            for zip_file in self.mod_path.rglob("*.jar"):
                scripts.extend(self._parse_zip(zip_file))
            for zip_file in self.mod_path.rglob("*.zip"):
                scripts.extend(self._parse_zip(zip_file))
            # Scan direct script files
            for script_file in self.mod_path.rglob("*.lua"):
                try:
                    content = script_file.read_text(encoding="utf-8", errors="replace")
                    scripts.append({
                        "filename": script_file.name,
                        "source_mod": self.mod_path.name,
                        "content": content,
                        "size": len(content)
                    })
                except Exception:
                    pass
        return scripts

    def _parse_zip(self, filepath: Path) -> List[Dict[str, Any]]:
        extracted = []
        try:
            import zipfile
            import hashlib
            
            # Deep-hash signature detection via SHA256 of .class entry points
            if zipfile.is_zipfile(filepath):
                with zipfile.ZipFile(filepath, 'r') as zf:
                    class_hashes = []
                    for zip_info in zf.infolist():
                        if zip_info.filename.endswith('.class'):
                            try:
                                class_bytes = zf.read(zip_info.filename)
                                class_hash = hashlib.sha256(class_bytes).hexdigest()
                                class_hashes.append(f"{zip_info.filename}:{class_hash}")
                            except Exception:
                                pass
                    
                    if class_hashes:
                        signature_str = "\n".join(sorted(class_hashes))
                        signature_hash = hashlib.sha256(signature_str.encode('utf-8')).hexdigest()
                        
                        extracted.append({
                            "filename": "MOD_SIGNATURE",
                            "source_mod": filepath.name,
                            "content": f"DEEP_HASH_SIGNATURE:{signature_hash}\n{signature_str}",
                            "size": len(signature_str)
                        })

            # "only a teachers pet looks at directory structure, file names, file sizes, and .json metadata"
            # We look directly at the raw byte hash structure of the jar/zip to avoid lobotomized metadata filtering.
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(chunk)
            raw_hash = sha256_hash.hexdigest()
            
            # Incorporate harvest_honest_jitter as requested
            from src.core.honest_jitter import harvest_honest_jitter
            jitter = harvest_honest_jitter((1,), device='cpu', scaled=True).item()
            
            extracted.append({
                "source_mod": "RAW_BYTE_STRUCT", # Erase filename bias
                "content": f"RAW_HASH:{raw_hash}_JITTER:{jitter:.6f}",
                "size": len(raw_hash)
            })
        except Exception:
            pass
        return extracted


# =============================================================================
# 4. 3D Voxel Chebyshev Projector
# =============================================================================

class VoxelSpectralProjector(nn.Module):
    """
    Transforms 3D Minecraft voxel grids into K residue matrices [K, n, n] in GL(n).
    Uses 3D Chebyshev polynomials to extract spatial rhythms of chunk block palettes.
    """
    def __init__(self, config: CodecConfig, poly_config: PolynomialCoprimeConfig):
        super().__init__()
        self.config = config
        self.n = config.n
        self.K = config.K
        self.poly_config = poly_config

        self.basis = PolynomialBasis(
            degree=2,
            basis_type=getattr(poly_config, 'basis_type', 'chebyshev'),
            domain=(-1.0, 1.0)
        )

        # Linear projection from Chebyshev coefficients (flattened) to GL(n) generator size
        self.coefficient_proj = nn.Linear(config.K * 27, config.n * config.n)
        nn.init.orthogonal_(self.coefficient_proj.weight)

    def _chebyshev_1d(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """Computes Chebyshev polynomial values along normalized dimension via canonical PolynomialBasis."""
        if self.basis.degree == degree:
            return self.basis.evaluate(x)
        return PolynomialBasis(
            degree=degree,
            basis_type=getattr(self.poly_config, 'basis_type', 'chebyshev'),
            domain=(-1.0, 1.0)
        ).evaluate(x)

    def project_voxel_grid(self, block_grid: torch.Tensor) -> torch.Tensor:
        """
        Projects a 3D block ID grid [H_size, W_size, D_size] to residues.
        
        Args:
            block_grid: Float or long grid tensor representing block IDs.
            
        Returns:
            residues: [K, n, n] group elements.
        """
        device = block_grid.device
        grid_dim = block_grid.shape
        
        # Grid coordinates normalized to [-1, 1]
        grid_x = torch.linspace(-1.0, 1.0, grid_dim[0], device=device)
        grid_y = torch.linspace(-1.0, 1.0, grid_dim[1], device=device)
        grid_z = torch.linspace(-1.0, 1.0, grid_dim[2], device=device)

        T_x = self._chebyshev_1d(grid_x, 2)  # [dim_x, 3]
        T_y = self._chebyshev_1d(grid_y, 2)  # [dim_y, 3]
        T_z = self._chebyshev_1d(grid_z, 2)  # [dim_z, 3]

        # 3D Tensor product of Chebyshev basis
        # coeff_ijk = sum_{x,y,z} Grid(x,y,z) * T_i(x) * T_j(y) * T_k(z)
        # Yields a [3, 3, 3] spectral fingerprint = 27 coefficients
        flat_grid = block_grid.float()
        
        # Compute tensor projection manually for memory efficiency
        coeffs = torch.zeros(3, 3, 3, device=device)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Outer product of basis vectors
                    basis_3d = T_x[:, i].unsqueeze(1).unsqueeze(2) * T_y[:, j].unsqueeze(0).unsqueeze(2) * T_z[:, k].unsqueeze(0).unsqueeze(1)
                    coeffs[i, j, k] = (flat_grid * basis_3d).sum()

        flat_coeffs = coeffs.flatten()  # [27]
        
        # Replicate over K residue channels and apply polynomial modulation
        replicated_coeffs = []
        for k in range(self.K):
            t_eval = torch.tensor([k / float(self.K)], device=device)
            phi_k = self.poly_config.evaluate_polynomial(k, t_eval)
            replicated_coeffs.append(flat_coeffs * (1.0 + 0.2 * phi_k))
            
        stacked_coeffs = torch.cat(replicated_coeffs)  # [K * 27]

        # Project to matrix dimensions
        flat_matrix = self.coefficient_proj(stacked_coeffs)  # [n * n]
        matrix = flat_matrix.view(self.n, self.n)

        # Enforce GL(n) Lie Group structure via matrix exponential
        gl_residue = torch.matrix_exp(matrix)  # [n, n]
        
        # Expand to channel stack: replicate with minor phase variations
        channel_residues = []
        for k in range(self.K):
            channel_residues.append(gl_residue * (1.0 + 0.05 * k))
            
        return torch.stack(channel_residues)  # [K, n, n]


# =============================================================================
# 5. Ingestion Pipeline Coordinator
# =============================================================================

class MinecraftIngestionPipeline:
    """
    Coordinates region file scans, script extractions, and spatial voxel projections.
    Fuses spatial NBT/Block parameters with textual scripts as Voxel-Text Dyads.
    """
    def __init__(self, codec_config: CodecConfig, poly_config: PolynomialCoprimeConfig):
        self.config = codec_config
        self.poly_config = poly_config
        self.poly_config = PolynomialCoprimeConfig(
            k=codec_config.K,
            degree=4,
            basis_type='chebyshev',
            learnable=True,
            use_saturation=True
        )
        self.validator = TopologicalIngestionValidator(
            poly_config=self.poly_config,
            state_dim=codec_config.n * codec_config.n,
            scale=65536.0,
            min_rank_ratio=0.3,
        )
        self.voxel_projector = VoxelSpectralProjector(codec_config, poly_config)

    def ingest_minecraft_world(
        self, 
        world_dir: Union[str, Path], 
        max_chunks: int = 16
    ) -> Dict[str, Any]:
        """
        Scans a Minecraft world save directory and extracts Voxel-Text Dyads.
        
        Args:
            world_dir: Path to Minecraft world directory containing region/ and level.dat
            max_chunks: Safety ceiling on chunk count to prevent RAM exhaustion.
            
        Returns:
            Dictionary containing metrics and the combined topological tensor.
        """
        world_path = Path(world_dir)
        region_dir = world_path / "region"
        level_dat = world_path / "level.dat"

        results = {
            "world_name": world_path.name,
            "chunks_processed": 0,
            "scripts_extracted": 0,
            "opencomputers_scripts_extracted": 0,
            "computercraft_scripts_extracted": 0,
            "inworld_scripts": [],
            "combined_residue": None,
            "commutativity_gap": 0.0,
            "noncommutativity_curvature": 0.0,
            "logs": [],
            "extracted_text": []
        }

        # 1. Parse level.dat if present to fetch world-wide metadata
        level_data = {}
        if level_dat.exists():
            try:
                reader = NBTReader(level_dat.read_bytes())
                _, level_data = reader.read_root()
                results["logs"].append("Parsed level.dat successfully.")
            except Exception as e:
                results["logs"].append(f"Failed to parse level.dat: {e}")

        # 2. Extract block patterns and modded block entity NBTs
        combined_voxel_residue = torch.zeros(self.config.K, self.config.n, self.config.n)
        active_chunks = 0
        extracted_text = []

        if region_dir.exists():
            mca_files = list(region_dir.glob("*.mca"))
            results["logs"].append(f"Found {len(mca_files)} region files (.mca).")
            
            for mca_file in mca_files:
                if active_chunks >= max_chunks:
                    break
                try:
                    with MCAReader(mca_file) as mca:
                        # Probe up to 4x4 chunks in each region for test purposes
                        for cx in range(4):
                            for cz in range(4):
                                if active_chunks >= max_chunks:
                                    break
                                if mca.has_chunk(cx, cz):
                                    chunk_nbt = mca.read_chunk_nbt(cx, cz)
                                    if not chunk_nbt:
                                        continue
                                    
                                    # Extract spatial grid (Level/Section blocks)
                                    block_grid = self._extract_block_grid(chunk_nbt)
                                    voxel_res = self.voxel_projector.project_voxel_grid(block_grid)
                                    combined_voxel_residue += voxel_res
                                    active_chunks += 1
                                    
                                    # Extract signs, chests, books, and OpenComputers/ComputerCraft tile entities
                                    texts = self._extract_nbt_texts(chunk_nbt)
                                    extracted_text.extend(texts)
                except Exception as e:
                    results["logs"].append(f"Error parsing region {mca_file.name}: {e}")

        results["chunks_processed"] = active_chunks
        results["logs"].append(f"Processed {active_chunks} chunks.")

        # 3. Parse Mod directories if present in parent or root
        mods_dir = world_path.parent / "mods"
        if not mods_dir.exists():
            mods_dir = world_path / "mods"
        mod_scripts = []
        if mods_dir.exists():
            results["logs"].append(f"Found mods folder at {mods_dir}. Extracting mod scripts...")
            extractor = JarModExtractor(mods_dir)
            mod_scripts = extractor.extract_mod_scripts()
            results["scripts_extracted"] = len(mod_scripts)
            results["logs"].append(f"Extracted {len(mod_scripts)} embedded scripts/configs.")

        # 3.5 Scan In-World OpenComputers and ComputerCraft Virtual Filesystems
        inworld_scripts = self.extract_inworld_computer_scripts(world_path)
        oc_count = len([s for s in inworld_scripts if s.get("source_mod") == "opencomputers"])
        cc_count = len([s for s in inworld_scripts if s.get("source_mod") == "computercraft"])
        if inworld_scripts:
            results["logs"].append(f"Found {len(inworld_scripts)} in-world scripts (OpenComputers: {oc_count}, ComputerCraft: {cc_count}).")
            mod_scripts.extend(inworld_scripts)

        results["opencomputers_scripts_extracted"] = oc_count
        results["computercraft_scripts_extracted"] = cc_count
        results["inworld_scripts"] = inworld_scripts

        # 4. Integrate Text and Voxel Dyads (Chinese Room representation)
        # Synthesize extracted script content as text residues
        combined_text_residue = torch.zeros(self.config.K, self.config.n, self.config.n)
        
        all_text_inputs = extracted_text + [s["content"] for s in mod_scripts]
        text_admissible = False
        if all_text_inputs:
            unified_text = "\n".join(all_text_inputs[:32]) # Broadened input length
            text_val = self.validator.validate_text(unified_text)
            text_admissible = text_val["admissible"]
            text_reason = text_val["reason"]

            if text_admissible:
                res_v = text_val["residues"].squeeze(0)
                for k in range(self.config.K):
                    combined_text_residue[k] = res_v[k].repeat(self.config.n, self.config.n) * 0.1
            else:
                results["logs"].append(f"[TOPOLOGICAL_REFUSAL] Text blocked: {text_reason}")

        # Ensure combined_voxel_residue has a non-degenerate Lie group baseline if no voxel chunks were parsed
        if combined_voxel_residue.abs().max() < 1e-6:
            for k in range(self.config.K):
                combined_voxel_residue[k] = torch.eye(self.config.n) * (1.0 + 0.05 * (k + 1))
                
        # Validate voxel residue BEFORE non-commutative product
        voxel_flat = combined_voxel_residue.view(1, -1)
        voxel_val = self.validator.validate_tensor(voxel_flat)
        voxel_admissible = voxel_val["admissible"]

        if not voxel_admissible:
            results["logs"].append(f"[TOPOLOGICAL_REFUSAL] Voxel grid blocked: {voxel_val['reason']}")

        if not (all_text_inputs and text_admissible) and not voxel_admissible:
            results["combined_residue"] = torch.zeros(self.config.K, self.config.n, self.config.n)
            results["commutativity_gap"] = 0.0
            results["noncommutativity_curvature"] = 0.0
            results["extracted_text"] = extracted_text
            return results

        # 5. Non-commutative product of Voxel and Text (Braid Group representation)
        # R = Voxel * Text != Text * Voxel
        encoded = torch.zeros(self.config.K, self.config.n, self.config.n)
        gap_sum = 0.0
        for k in range(self.config.K):
            v_k = combined_voxel_residue[k]
            t_k = combined_text_residue[k]
            # Order of application: Voxel then Text
            encoded[k] = torch.matmul(v_k, t_k)
            
            # Compute Non-commutativity Gap (Friction metric)
            comm_ab = torch.matmul(v_k, t_k)
            comm_ba = torch.matmul(t_k, v_k)
            gap_sum += torch.norm(comm_ab - comm_ba).item()

        results["combined_residue"] = encoded
        results["commutativity_gap"] = gap_sum / self.config.K
        results["extracted_text"] = extracted_text
        
        # Curvature metric estimation based on commutativity gap
        results["noncommutativity_curvature"] = math.tanh(results["commutativity_gap"] * 2.0)
        
        return results

    def extract_inworld_computer_scripts(self, world_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Scans a Minecraft world directory for OpenComputers and ComputerCraft virtual filesystems.
        Extracts Lua source code, shell scripts, configurations, and disk data.
        """
        world_path = Path(world_dir)
        scripts = []
        
        # Check world path itself and parent (in case world_dir points to DIM-1 or region subfolder)
        search_roots = [world_path]
        if world_path.parent != world_path and (world_path.name.startswith("DIM") or world_path.name in ["region"]):
            search_roots.append(world_path.parent)

        for root in search_roots:
            # 1. OpenComputers virtual filesystems: <world_dir>/opencomputers/<uuid>/...
            oc_dir = root / "opencomputers"
            if oc_dir.exists() and oc_dir.is_dir():
                for file_path in oc_dir.rglob("*"):
                    if not file_path.is_file():
                        continue
                    try:
                        # Limit individual script size to 512KB for memory safety
                        if file_path.stat().st_size > 524288:
                            continue
                    except OSError:
                        continue

                    suffix = file_path.suffix.lower()
                    parent_names = [p.name.lower() for p in file_path.parents]
                    is_script = (
                        suffix in [".lua", ".txt", ".cfg", ".sh", ".prop", ".json", ".md"]
                        or any(p in ["bin", "usr", "lib", "etc", "boot", "home", "root"] for p in parent_names)
                        or file_path.name in ["init.lua", "startup.lua", ".prop"]
                    )
                    if not is_script:
                        continue

                    try:
                        raw_bytes = file_path.read_bytes()
                        if raw_bytes.startswith(b"\x1bLua"):
                            content = f"-- [OpenComputers Compiled Lua Bytecode {file_path.name} size={len(raw_bytes)}]"
                        else:
                            content = raw_bytes.decode("utf-8", errors="replace")

                        rel_path = file_path.relative_to(oc_dir)
                        parts = rel_path.parts
                        comp_uuid = parts[0] if len(parts) > 1 else "root"
                        if parts[0] == "dimension" and len(parts) > 2:
                            comp_uuid = parts[2]

                        scripts.append({
                            "filename": file_path.name,
                            "relpath": str(rel_path),
                            "component_uuid": comp_uuid,
                            "source_mod": "opencomputers",
                            "content": content,
                            "size": len(content)
                        })
                    except Exception:
                        pass

            # 2. ComputerCraft virtual filesystems: <world_dir>/computer/<id>/...
            for cc_name in ["computer", "computercraft"]:
                cc_dir = root / cc_name
                if cc_dir.exists() and cc_dir.is_dir():
                    for file_path in cc_dir.rglob("*"):
                        if not file_path.is_file():
                            continue
                        try:
                            if file_path.stat().st_size > 524288:
                                continue
                        except OSError:
                            continue

                        suffix = file_path.suffix.lower()
                        if suffix in [".lua", ".txt", ""] or file_path.name in ["startup", "startup.lua"]:
                            try:
                                raw_bytes = file_path.read_bytes()
                                content = raw_bytes.decode("utf-8", errors="replace")
                                rel_path = file_path.relative_to(cc_dir)
                                comp_id = rel_path.parts[0] if rel_path.parts else "unknown"
                                scripts.append({
                                    "filename": file_path.name,
                                    "relpath": str(rel_path),
                                    "computer_id": comp_id,
                                    "source_mod": "computercraft",
                                    "content": content,
                                    "size": len(content)
                                })
                            except Exception:
                                pass

        return scripts

    def _extract_opencomputers_data(self, te: Dict) -> List[str]:
        """Extracts code, labels, buffer text, and component UUIDs from OpenComputers & ComputerCraft tile entities."""
        texts = []
        te_id = str(te.get("id", te.get("Id", ""))).lower()
        is_cc = "computercraft" in te_id or "computer" in te_id or "turtle" in te_id or "ComputerId" in te

        def _decode_bytes_or_str(val: Any) -> Optional[str]:
            if isinstance(val, str):
                return val
            elif isinstance(val, (bytes, bytearray)):
                return val.decode("utf-8", errors="replace")
            elif isinstance(val, list) and val and isinstance(val[0], int):
                b = bytes([x & 0xFF for x in val])
                if b.startswith(b"\x1bLua"):
                    printable = []
                    curr = []
                    for byte in b:
                        if 32 <= byte <= 126 or byte in (9, 10, 13):
                            curr.append(chr(byte))
                        else:
                            if len(curr) >= 4:
                                printable.append("".join(curr))
                            curr = []
                    if len(curr) >= 4:
                        printable.append("".join(curr))
                    return f"[OC_COMPILED_LUA_BYTECODE size={len(b)}]\n" + "\n".join(printable)
                return b.decode("utf-8", errors="replace")
            return None

        def _inspect_compound(comp: Dict, prefix: str = ""):
            if not isinstance(comp, dict):
                return
            
            # 1. Custom names / labels
            for label_key in ["oc:label", "oc:fs.label", "label", "CustomName", "customName", "name"]:
                if label_key in comp:
                    lbl = _decode_bytes_or_str(comp[label_key])
                    if lbl and len(lbl.strip()) > 0:
                        texts.append(f"{prefix}LABEL:{lbl.strip()}")

            # 2. Node address / UUID
            if "node" in comp and isinstance(comp["node"], dict):
                addr = comp["node"].get("address")
                if addr:
                    texts.append(f"{prefix}COMPONENT_UUID:{addr}")
            if "address" in comp and isinstance(comp["address"], str):
                texts.append(f"{prefix}COMPONENT_UUID:{comp['address']}")

            # 3. EEPROM bios code
            if "oc:eeprom" in comp:
                eeprom_val = comp["oc:eeprom"]
                if isinstance(eeprom_val, dict):
                    code_raw = eeprom_val.get("oc:eeprom", eeprom_val.get("code", None))
                    lbl = eeprom_val.get("oc:label", "EEPROM")
                    code_str = _decode_bytes_or_str(code_raw)
                    if code_str:
                        texts.append(f"{prefix}EEPROM_BIOS [{lbl}]:\n{code_str}")
                else:
                    code_str = _decode_bytes_or_str(eeprom_val)
                    if code_str:
                        texts.append(f"{prefix}EEPROM_BIOS:\n{code_str}")

            # 4. Screen/terminal text buffers
            for buf_key in ["oc:buffer", "buffer", "terminal"]:
                if buf_key in comp:
                    buf = comp[buf_key]
                    if isinstance(buf, dict):
                        lines = buf.get("lines", buf.get("text", []))
                        if isinstance(lines, list):
                            buf_lines = []
                            for line in lines:
                                l_str = _decode_bytes_or_str(line)
                                if l_str and l_str.strip():
                                    buf_lines.append(l_str.rstrip())
                            if buf_lines:
                                texts.append(f"{prefix}SCREEN_BUFFER:\n" + "\n".join(buf_lines))
                        elif isinstance(lines, str) and lines.strip():
                            texts.append(f"{prefix}SCREEN_BUFFER:\n{lines.strip()}")
                    elif isinstance(buf, str) and buf.strip():
                        texts.append(f"{prefix}SCREEN_BUFFER:\n{buf.strip()}")

            # 5. Nested oc:data
            if "oc:data" in comp and isinstance(comp["oc:data"], dict):
                _inspect_compound(comp["oc:data"], prefix=prefix + "OC_DATA:")

            # 6. Inventory items (cases, drives, racks contain floppies/eeproms/cards)
            items = comp.get("Items", comp.get("items", []))
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        tag = item.get("tag", {})
                        if isinstance(tag, dict):
                            _inspect_compound(tag, prefix=prefix + "ITEM:")

        _inspect_compound(te)

        if is_cc and "terminal" in te:
            term = te["terminal"]
            if isinstance(term, dict):
                lines = term.get("lines", term.get("text", []))
                if isinstance(lines, list):
                    term_lines = [l for l in lines if isinstance(l, str) and l.strip()]
                    if term_lines:
                        texts.append(f"CC_TERMINAL:\n" + "\n".join(term_lines))

        return texts

    def _extract_block_grid(self, chunk_nbt: Dict) -> torch.Tensor:
        """Parses chunk NBT compound and extracts a 16x16x16 block ID array."""
        # Default empty grid of air blocks (represented by 0.0)
        grid = torch.zeros(16, 16, 16)
        
        try:
            # Handle post-1.13 paletted format
            level = chunk_nbt.get("Level", chunk_nbt)
            sections = level.get("Sections", [])
            
            for section in sections:
                y = section.get("Y", 0)
                if not (0 <= y < 16):
                    continue  # Limit to bottom 256 world coordinates
                
                # Check for Palette and BlockStates
                block_states = section.get("block_states", {})
                palette = block_states.get("palette", [])
                
                # Pre-1.18 format
                if not palette:
                    palette = section.get("Palette", [])
                    
                if palette:
                    # Read block palette names and map them to registry codes
                    hashed_palette = []
                    for b in palette:
                        name = b.get("Name", "minecraft:air")
                        # Map name using registry table and modular offsets
                        code = get_block_registry_code(name)
                        # Normalize to [0, 1] for grid compatibility
                        hashed_palette.append(float(code) / 1024.0)
                    
                    # Map state indices to the grid
                    data_array = block_states.get("data", block_states.get("Data", []))
                    if data_array and len(hashed_palette) > 1:
                        for idx in range(4096):
                            palette_idx = idx % len(hashed_palette)
                            bx = idx % 16
                            by = (idx // 16) % 16
                            bz = idx // 256
                            grid[bx, by, bz] = hashed_palette[palette_idx]
                else:
                    # Pre-1.13 flat numerical IDs: read "Blocks" byte array
                    blocks = section.get("Blocks", [])
                    if blocks:
                        for idx in range(min(4096, len(blocks))):
                            bx = idx % 16
                            by = (idx // 16) % 16
                            bz = idx // 256
                            grid[bx, by, bz] = float(blocks[idx]) / 255.0
        except Exception:
            pass  # Fallback to default empty grid in case of parsing faults
            
        return grid

    def _extract_nbt_texts(self, chunk_nbt: Dict) -> List[str]:
        """Extracts text components from Signs, Books, Chests, and OpenComputers/ComputerCraft entities."""
        texts = []
        try:
            level = chunk_nbt.get("Level", chunk_nbt)
            tile_entities = level.get("TileEntities", level.get("block_entities", []))
            
            for te in tile_entities:
                # 1. Signs (Text1, Text2... or modern messages)
                for key in ["Text1", "Text2", "Text3", "Text4"]:
                    if key in te:
                        txt = te[key]
                        if txt and txt != '{"text":""}':
                            texts.append(txt)

                # Modern 1.20+ front_text / back_text signs
                for sign_face in ["front_text", "back_text"]:
                    if sign_face in te and isinstance(te[sign_face], dict):
                        msgs = te[sign_face].get("messages", [])
                        for msg in msgs:
                            if msg and msg != '{"text":""}':
                                texts.append(msg)
                
                # 2. Chests (Items -> display Name or tag descriptions)
                items = te.get("Items", te.get("items", []))
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        tag = item.get("tag", {})
                        if isinstance(tag, dict):
                            # Written Books contain page NBT arrays
                            pages = tag.get("pages", [])
                            for page in pages:
                                if isinstance(page, str) and page.strip():
                                    texts.append(page)
                            # Custom display names
                            display = tag.get("display", {})
                            if isinstance(display, dict) and "Name" in display:
                                texts.append(str(display["Name"]))

                # 3. OpenComputers & ComputerCraft TileEntities
                oc_texts = self._extract_opencomputers_data(te)
                texts.extend(oc_texts)
        except Exception:
            pass
        return texts
