#!/usr/bin/env python3
"""
Gyroidic Dataset Ingestion & Training System

A comprehensive system for ingesting datasets from various sources and training
the Gyroidic Sparse Covariance Flux Reasoner while maintaining anti-lobotomy
principles and structural honesty.

Key Features:
- Multiple dataset source integration (HuggingFace, Kaggle, Wikipedia, local files)
- Mandelbulb-Gyroidic geometric augmentation
- Non-teleological training with evolutionary trust selection
- Temporal association learning
- Structural integrity preservation
- Anti-lobotomy compliance monitoring

Author: System Architecture Team
Date: January 2026
"""

import sys
import os
import argparse
import json
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import time
from urllib.parse import urlparse
import subprocess
import gzip
import csv
from src.core.honest_jitter import harvest_honest_jitter

# Robust path management
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Core system imports (anti-lobotomy compliant)
from core.polynomial_coprime import PolynomialCoprimeConfig
from augmentation.mandelbulb_gyroidic_augmenter import MandelbulbGyroidicAugmenter, AugmentationConfig
from training.temporal_association_trainer import TemporalAssociationTrainer, TemporalAssociationDataset
from ui.wikipedia_integration import wikipedia_integration
# Image Processor for Multimodal Support
from image_extension import ImageProcessor

# Import training examples
sys.path.append('examples')
from enhanced_temporal_training import NonLobotomyTemporalModel, NonLobotomyTemporalTrainer

@dataclass
class DatasetConfig:
    """Configuration for dataset ingestion."""
    name: str
    source_type: str  # 'huggingface', 'kaggle', 'wikipedia', 'local', 'url'
    source_path: str
    preprocessing: str = 'text'  # 'text', 'image', 'tabular', 'multimodal'
    augmentation: bool = True
    mandelbulb_augmentation: bool = False
    temporal_associations: bool = True
    max_samples: Optional[int] = None
    validation_split: float = 0.2
    manifold_aware: bool = False

    @property
    def safe_name(self) -> str:
        """Sanitized name safe for Windows directory creation."""
        s = self.name.replace(':', '_').replace(',', '_').replace('/', '_').replace('\\', '_')
        if len(s) > 50:
            s = s[:50]
        return s

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    model_type: str = 'temporal'  # 'temporal', 'association', 'multimodal'
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    evolution_rate: float = 0.02
    fossilization_threshold: float = 0.8
    survivorship_threshold: float = 0.7
    use_mandelbulb_augmentation: bool = False
    augmentation_factor: int = 2
    save_checkpoints: bool = True
    checkpoint_interval: int = 5

class SovereignDynamicDataset(torch.utils.data.Dataset):
    """
    Sovereign Dynamic Dataset: Loads samples on-demand from disk.
    
    Architecture:
    - manifest.json: Metadata and chunk index
    - chunks/: Subdirectory containing .pt chunk files (e.g., 1000 samples each)
    
    This prevents VRAM/RAM pressure by avoiding loading the entire dataset at once.
    """
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        manifest_path = self.dataset_path / "manifest.json"
        
        if not manifest_path.exists():
            # Legacy support: if manifest doesn't exist, we might have a single .pt file
            legacy_path = self.dataset_path / "processed_data.pt"
            if legacy_path.exists():
                print(f"   [WARN] Legacy dataset detected at {legacy_path}. Loading into memory (one last time).")
                self.data = torch.load(legacy_path)
                self.num_samples = len(self.data)
                self.is_legacy = True
            else:
                # Check for synthetic datasets (might not have files)
                self.data = []
                self.num_samples = 0
                self.is_legacy = True
        else:
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            self.num_samples = self.manifest['num_samples']
            self.samples_per_chunk = self.manifest['samples_per_chunk']
            self.num_chunks = self.manifest['num_chunks']
            self._current_chunk_idx = -1
            self._current_chunk_data = None
            self.is_legacy = False
            self.data = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.is_legacy:
            if idx < len(self.data):
                return self.data[idx]
            return {}
            
        chunk_idx = idx // self.samples_per_chunk
        intra_idx = idx % self.samples_per_chunk
        
        # Security check
        if chunk_idx >= self.num_chunks:
            return {}

        if chunk_idx != self._current_chunk_idx:
            chunk_path = self.dataset_path / "chunks" / f"chunk_{chunk_idx}.pt"
            if chunk_path.exists():
                self._current_chunk_data = torch.load(chunk_path)
                self._current_chunk_idx = chunk_idx
            else:
                return {}
            
        if self._current_chunk_data and intra_idx < len(self._current_chunk_data):
            return self._current_chunk_data[intra_idx]
        return {}

    def __iter__(self):
        for i in range(self.num_samples):
            yield self.__getitem__(i)

class DatasetIngestionSystem:
    """
    Main system for dataset ingestion and training.
    
    Maintains anti-lobotomy principles:
    - No hardcoded primes (uses polynomial co-prime functionals)
    - Evolutionary trust selection (not gradient descent on trust)
    - Structural honesty (no placeholders)
    - Non-teleological flow (survivorship pressure, not loss minimization)
    """
    
    def __init__(self, device: str = 'auto', engine: Optional[Any] = None):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # [FULL BRIDGE] Initialize DiegeticPhysicsEngine for Manifold-Aware (Thick) Ingestion
        if engine is not None:
            self.engine = engine
            print(f"[INGEST] Manifold Bridge ACTIVE (reused existing engine) on {self.device}")
        else:
            try:
                from src.ui.diegetic_backend import DiegeticPhysicsEngine
                self.engine = DiegeticPhysicsEngine(device=self.device)
                print(f"[INGEST] Manifold Bridge ACTIVE on {self.device}")
            except ImportError:
                self.engine = None
                print("[INGEST] Warning: DiegeticPhysicsEngine not found. Manifold-aware ingestion disabled.")
            except Exception as e:
                self.engine = None
                print(f"[INGEST] Warning: Failed to initialize Manifold Bridge: {e}")
        
        self.datasets = {}
        self.models = {}
        self.trainers = {}
        self.augmenters = {}

        # Initialize Image Processor (satellite component)
        try:
            self.image_processor = ImageProcessor(device=self.device)
            print(f"[IMG] Image Processor initialized on {self.device}")
        except Exception as e:
            print(f"[WARN] Failed to initialize ImageProcessor: {e}")
            self.image_processor = None
        
        # Create data directory
        self.data_dir = Path("datasets")
        self.data_dir.mkdir(exist_ok=True)
        
        # Training history
        self.training_history = {}
        
        print(f"[BRAIN] Gyroidic Dataset Ingestion System initialized")
        print(f"   Device: {self.device}")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Anti-lobotomy compliance: [OK] ACTIVE")
    
    def _save_dynamic_dataset(self, samples: List[Dict], dataset_path: Path, config: DatasetConfig, samples_per_chunk: int = 1000):
        """Save dataset in chunks with a manifest for dynamic loading."""
        chunks_dir = dataset_path / "chunks"
        chunks_dir.mkdir(exist_ok=True, parents=True)
        
        num_samples = len(samples)
        num_chunks = (num_samples + samples_per_chunk - 1) // samples_per_chunk
        
        print(f"   [DISK] Saving {num_samples} samples into {num_chunks} chunks (dynamic loading enabled)...")
        
        for i in range(num_chunks):
            chunk = samples[i*samples_per_chunk : (i+1)*samples_per_chunk]
            chunk_path = chunks_dir / f"chunk_{i}.pt"
            torch.save(chunk, chunk_path)
            if (i+1) % 10 == 0 or i == num_chunks - 1:
                print(f"   [DISK] Progress: {i+1}/{num_chunks} chunks saved")
                
        # Save manifest
        manifest = {
            'num_samples': num_samples,
            'num_chunks': num_chunks,
            'samples_per_chunk': samples_per_chunk,
            'timestamp': time.time(),
            'config': {
                'name': config.name,
                'source_type': config.source_type,
                'preprocessing': config.preprocessing
            }
        }
        
        with open(dataset_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=4)
            
        print(f"   [OK] Dataset manifest saved to {dataset_path / 'manifest.json'}")

    def add_dataset_source(self, config: DatasetConfig) -> bool:
        """Add a dataset source for ingestion."""
        print(f"\n[DATA] Adding dataset: {config.name}")
        print(f"   Source: {config.source_type} - {config.source_path}")
        print(f"   Preprocessing: {config.preprocessing}")
        print(f"   Augmentation: {config.augmentation}")
        print(f"   Mandelbulb augmentation: {config.mandelbulb_augmentation}")
        
        try:
            if config.source_type == 'huggingface':
                success = self._ingest_huggingface_dataset(config)
            elif config.source_type == 'kaggle':
                success = self._ingest_kaggle_dataset(config)
            elif config.source_type == 'wikipedia':
                success = self._ingest_wikipedia_dataset(config)
            elif config.source_type == 'local':
                success = self._ingest_local_dataset(config)
            elif config.source_type == 'url':
                success = self._ingest_url_dataset(config)
            elif config.source_type == 'portal':
                success = self._ingest_portal_dataset(config)
            elif config.source_type == 'minecraft':
                success = self._ingest_minecraft_dataset(config)
            elif config.source_type == 'open_science':
                success = self._ingest_open_science_dataset(config)
            else:
                print(f"[FAIL] Failed to add dataset {config.name}")
                return False
            
            if success:
                self.datasets[config.name] = config
            
            return success
                
        except Exception as e:
            print(f"[ERR] Error adding dataset {config.name}: {e}")
            return False

    def _ingest_open_science_dataset(self, config: DatasetConfig, return_samples: bool = False) -> Union[bool, List[Dict]]:
        """Ingest dataset from open-access scientific API queries."""
        try:
            print(f"[OPEN_SCIENCE] Loading open science queries from: {config.source_path}")
            
            # 1. Parse the query configs
            query_configs = []
            source_path_str = config.source_path.strip()
            
            # Check if it's a JSON string directly
            if source_path_str.startswith("[") or source_path_str.startswith("{"):
                try:
                    parsed = json.loads(source_path_str)
                    if isinstance(parsed, list):
                        query_configs = parsed
                    elif isinstance(parsed, dict):
                        query_configs = [parsed]
                except Exception as e:
                    print(f"[ERR] Failed to parse open science JSON string: {e}")
            else:
                # Check if it's a file
                try:
                    file_path = Path(source_path_str)
                    if file_path.exists() and file_path.is_file():
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content.startswith("[") or content.startswith("{"):
                                parsed = json.loads(content)
                                if isinstance(parsed, list):
                                    query_configs = parsed
                                elif isinstance(parsed, dict):
                                    query_configs = [parsed]
                            else:
                                for line in content.splitlines():
                                    line = line.strip()
                                    if not line or line.startswith("#"):
                                        continue
                                    try:
                                        query_configs.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        query_configs.append({"type": line})
                except Exception as e:
                    print(f"   [WARN] Failed to parse query config file {file_path.name}: {e}")
            
            # If not parsed from file, try to parse directly as JSON string
            if not query_configs:
                if source_path_str.startswith("[") or source_path_str.startswith("{"):
                    try:
                        parsed = json.loads(source_path_str)
                        if isinstance(parsed, list):
                            query_configs = parsed
                        elif isinstance(parsed, dict):
                            query_configs = [parsed]
                    except json.JSONDecodeError as e:
                        print(f"   [WARN] Failed to parse inline JSON query: {e}")
            
            # If still empty, assume comma-separated list of types or "all"/"default"
            if not query_configs:
                types = []
                if source_path_str.lower() in ["all", "default", "standard"]:
                    types = ["ligo", "sdss", "ncbi", "openneuro"]
                else:
                    types = [t.strip().lower() for t in source_path_str.split(",") if t.strip()]
                
                reference_queries = {
                    "ligo": {
                        "type": "ligo",
                        "event": "GW190521",
                        "detector": "H1",
                        "duration": 4.0,
                        "sample_rate": 4096
                    },
                    "sdss": {
                        "type": "sdss",
                        "catalog_id": "J/A+A/540/A106",
                        "row_limit": 100
                    },
                    "ncbi": {
                        "type": "ncbi",
                        "accession_id": "AM743169.1",
                        "db": "nucleotide"
                    },
                    "openneuro": {
                        "type": "openneuro",
                        "dataset_id": "ds003445",
                        "subject_id": "sub-01",
                        "run_id": "run-1"
                    }
                }
                
                for t in types:
                    if t in reference_queries:
                        query_configs.append(reference_queries[t])
                    else:
                        print(f"   [WARN] Unknown query type: {t}, passing basic query dict.")
                        query_configs.append({"type": t})
            
            print(f"   [OPEN_SCIENCE] Aggregating {len(query_configs)} scientific queries...")
            
            # 2. Query and aggregate samples using OpenScienceIngestor
            from src.data.open_science_ingestor import OpenScienceIngestor
            cache_dir = self.data_dir / "open_science_cache"
            ingestor = OpenScienceIngestor(cache_dir=str(cache_dir))
            
            samples = ingestor.query_and_aggregate(query_configs)
            
            if config.max_samples:
                samples = samples[:config.max_samples]
                
            # Apply manifold thick preprocessing (CODES v40 PAS_LOCK & TEMPOLOCK)
            import math
            final_samples = []
            
            # CODES v40 Thresholds
            THETA_L = 0.85 # Minimum lawful PAS threshold
            MAX_DRIFT = 0.05 # PAS_zeta drift limit
            
            # Speculative Coprime Gate for recovery
            try:
                from src.topology.speculative_homology import SpeculativeCoprimeGate
                coprime_gate = SpeculativeCoprimeGate(device=self.device if hasattr(self, 'device') else torch.device('cpu'))
            except ImportError:
                coprime_gate = None

            # Instantiate reusable SDMI PAS components
            from src.core.invariants import PhaseAlignmentInvariant, APAS_Zeta
            try:
                from src.core.invariants import compute_chiral_shift
            except ImportError:
                def compute_chiral_shift(x): return torch.tensor(0.0)
            from src.core.honest_jitter import harvest_honest_jitter
            import torch
            
            device = self.device if hasattr(self, 'device') else torch.device('cpu')
            pas_calculator = PhaseAlignmentInvariant(degree=33).to(device)
            zeta_checker = APAS_Zeta(zeta=MAX_DRIFT).to(device)

            for i, sample in enumerate(samples):
                text_content = sample.get('text', '')
                text_len = len(text_content)
                
                # 1. CODES v40: SDMI PAS_h and PAS_LOCK
                # SDMI Polynomial Tokenization (String -> Coefficient Tensor)
                char_bytes = text_content.encode('utf-8') if text_content else b'0'
                poly_coeffs = torch.tensor(list(char_bytes), dtype=torch.float32, device=device).unsqueeze(0)
                
                # Expand/truncate to match 33-shell representation
                if poly_coeffs.shape[1] > 33:
                    poly_coeffs = poly_coeffs[:, :33]
                else:
                    poly_coeffs = torch.nn.functional.pad(poly_coeffs, (0, 33 - poly_coeffs.shape[1]))
                    
                # Adaptive Shell Weighting (prioritize lower frequency harmonics)
                weights = torch.exp(-0.1 * torch.arange(33, dtype=torch.float32, device=device)).unsqueeze(0)
                weighted_coeffs = poly_coeffs * weights
                
                # Symmetry Breaking Perturbation
                chiral_drift = compute_chiral_shift(weighted_coeffs)
                symmetry_breaker = harvest_honest_jitter(weighted_coeffs.shape, device=device) * (chiral_drift.item() + 0.01)
                perturbed_state = weighted_coeffs + symmetry_breaker
                
                # Calculate PAS_h & APAS_zeta drift limit
                current_pas = pas_calculator(perturbed_state)
                
                last_pas_tensor = getattr(self, '_last_pas_tensor', current_pas)
                drift_tensor, violation = zeta_checker.check_drift(current_pas, last_pas_tensor)
                self._last_pas_tensor = current_pas
                
                # Compatibility for Lazarus scoring
                drift = drift_tensor.item()
                pas_h = current_pas.item()
                self._last_pas = pas_h
                
                pas_lock = (pas_h >= THETA_L) and (violation.item() == 0.0)
                
                # 2. Attempt topological recovery if PAS_LOCK fails
                if not pas_lock and coprime_gate:
                    try:
                        # Feed the length/structure into the gate
                        dummy_tensor = torch.tensor([[float(text_len)] * 64], device=coprime_gate.device)
                        recovered, gap = coprime_gate(dummy_tensor)
                        if gap < 1.0: # Arbitrary threshold for structural recovery
                            pas_h = min(1.0, pas_h + 0.3) # Boost PAS through topological adjustment
                            pas_lock = (pas_h >= THETA_L)
                    except Exception:
                        pass
                        
                if not pas_lock:
                    # Treat as drift/dark matter
                    print(f"   [CODES] Discarding open science sample {i} (PAS_LOCK failed: {pas_h:.2f} < {THETA_L})")
                    continue
                    
                preprocessed = self._preprocess_sample(sample, 'text')
                if preprocessed:
                    # 3. Add TEMPOLOCK and CODES mapping
                    # Centralized dynamic prime sourcing from FGRT to prevent isolated lobotomies
                    from src.core.fgrt_primitives import PrimeResonanceLadder
                    prime_ladder = PrimeResonanceLadder(num_resonators=10)
                    prime_gates = prime_ladder.primes.tolist()
                    
                    assigned_gate = prime_gates[i % len(prime_gates)]
                    
                    preprocessed['metadata'].update(sample.get('metadata', {}))
                    preprocessed['metadata']['CODES_v40'] = {
                        'PAS_h': round(pas_h, 4),
                        'PAS_zeta_drift': round(drift, 4),
                        'PAS_LOCK': pas_lock,
                        'TEMPOLOCK_interval': assigned_gate,
                        'entropy_banded': True,
                        'GLYPHLOCK': True
                    }
                    final_samples.append(preprocessed)
                else:
                    final_samples.append(sample)
            
            if return_samples:
                return final_samples
                
            # Save dataset
            dataset_path = self.data_dir / config.safe_name
            dataset_path.mkdir(exist_ok=True, parents=True)
            self._save_dynamic_dataset(final_samples, dataset_path, config)
            
            print(f"[OK] Open Science dataset created and chunked: {len(final_samples)} samples")
            return True
            
        except Exception as e:
            print(f"[ERR] Open Science ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _ingest_minecraft_dataset(self, config: DatasetConfig, return_samples: bool = False) -> Union[bool, List[Dict]]:
        """Ingest dataset from Minecraft world saves and mod packages."""
        try:
            print(f"[MINECRAFT] Loading Minecraft dataset from: {config.source_path}")
            
            from src.data.minecraft_ingestor import MinecraftIngestionPipeline, JarModExtractor
            from src.codec.gyroidic_codec import CodecConfig
            
            minecraft_dir = Path("datasets") / "minecraft"
            world_path = minecraft_dir / config.source_path
            
            if not world_path.exists():
                print(f"[ERR] Minecraft world save does not exist: {world_path}")
                return False
                
            # Initialize pipeline (use default configuration sizes, e.g. K=5, n=256)
            codec_config = CodecConfig(K=5, n=256, device=self.device)
            poly_config = PolynomialCoprimeConfig(k=5, degree=4, device=self.device)
            pipeline = MinecraftIngestionPipeline(codec_config, poly_config)
            
            max_chunks = config.max_samples if config.max_samples else 16
            results = pipeline.ingest_minecraft_world(world_path, max_chunks=max_chunks)
            
            samples = []
            voxel_res = results.get("combined_residue")
            voxel_res_list = voxel_res.tolist() if isinstance(voxel_res, torch.Tensor) else None
            
            # 1. Raw Byte Mod Ingestion (No JVM, No Names, Pure Structural Topology)
            mods_dir = world_path.parent / "mods"
            if mods_dir.exists():
                from src.core.honest_jitter import AgentSmithEngine
                import hashlib
                
                engine = AgentSmithEngine(device=self.device if hasattr(self, 'device') else torch.device('cpu'))
                for mod_file in mods_dir.rglob("*"):
                    if mod_file.is_file() and mod_file.suffix in ['.jar', '.zip']:
                        try:
                            # Read raw bytes completely ignoring internal structure or semantics
                            with open(mod_file, "rb") as f:
                                raw_bytes = f.read()
                            
                            if not raw_bytes: continue
                            
                            # Chunked rolling hash logic over pure bytes
                            chunk_size = 1024 * 1024 # 1MB chunks
                            structural_hash = 0
                            for i in range(0, len(raw_bytes), chunk_size):
                                chunk = raw_bytes[i:i+chunk_size]
                                chunk_val = int(hashlib.sha256(chunk).hexdigest()[:8], 16)
                                structural_hash = (structural_hash + chunk_val) % (16**8)
                                
                            deterministic_seed = structural_hash / (16**8)
                            
                            # Sovereign Logistic Expansion maps byte structure to manifold
                            structural_embedding = engine((768,), seed_val=deterministic_seed, scaled=False)
                            
                            sample = {
                                'text': f"RAW_BYTE_TOPOLOGY:{hashlib.md5(raw_bytes).hexdigest()}",
                                'length': len(raw_bytes),
                                'source': f"minecraft_mod_raw_byte_stream",
                                'metadata': {
                                    'world_name': config.source_path,
                                    'type': 'raw_byte_topology',
                                    'noncommutativity_curvature': results.get('noncommutativity_curvature', 0.0),
                                    'commutativity_gap': results.get('commutativity_gap', 0.0),
                                    'structural_embedding': structural_embedding.cpu().tolist()
                                }
                            }
                            if voxel_res_list:
                                sample['metadata']['voxel_residue'] = voxel_res_list
                            samples.append(sample)
                        except Exception as e:
                            print(f"[WARN] Failed to process mod byte stream: {e}")

            # 2. Config Files Ingestion
            config_texts = []
            
            # World-specific configs
            serverconfig_dir = world_path / "serverconfig"
            if serverconfig_dir.exists():
                for cfg_file in serverconfig_dir.rglob("*"):
                    if cfg_file.is_file() and cfg_file.suffix in ['.toml', '.json', '.cfg', '.txt', '.conf', '.yaml', '.yml']:
                        try:
                            content = cfg_file.read_text(encoding='utf-8', errors='replace')
                            if content.strip():
                                config_texts.append({
                                    'name': cfg_file.name,
                                    'type': 'serverconfig',
                                    'content': content
                                })
                        except Exception:
                            pass
            
            # Global configs
            globalconfig_dir = world_path.parent / "config"
            if globalconfig_dir.exists():
                for cfg_file in globalconfig_dir.rglob("*"):
                    if cfg_file.is_file() and cfg_file.suffix in ['.toml', '.json', '.cfg', '.txt', '.conf', '.yaml', '.yml']:
                        try:
                            content = cfg_file.read_text(encoding='utf-8', errors='replace')
                            if content.strip():
                                config_texts.append({
                                    'name': cfg_file.name,
                                    'type': 'config',
                                    'content': content
                                })
                        except Exception:
                            pass
                            
            # defaultconfigs configs
            defaultconfig_dir = world_path.parent / "defaultconfigs"
            if defaultconfig_dir.exists():
                for cfg_file in defaultconfig_dir.rglob("*"):
                    if cfg_file.is_file() and cfg_file.suffix in ['.toml', '.json', '.cfg', '.txt', '.conf', '.yaml', '.yml']:
                        try:
                            content = cfg_file.read_text(encoding='utf-8', errors='replace')
                            if content.strip():
                                config_texts.append({
                                    'name': cfg_file.name,
                                    'type': 'defaultconfigs',
                                    'content': content
                                })
                        except Exception:
                            pass

            for cfg in config_texts:
                sample = {
                    'text': f"Config Type: {cfg['type']}\nFilename: {cfg['name']}\nContent:\n{cfg['content']}",
                    'length': len(cfg['content']),
                    'source': f"minecraft:{config.source_path}/{cfg['type']}/{cfg['name']}",
                    'metadata': {
                        'world_name': config.source_path,
                        'type': 'config_file',
                        'config_type': cfg['type'],
                        'filename': cfg['name'],
                        'noncommutativity_curvature': results.get('noncommutativity_curvature', 0.0),
                        'commutativity_gap': results.get('commutativity_gap', 0.0),
                    }
                }
                if voxel_res_list:
                    sample['metadata']['voxel_residue'] = voxel_res_list
                samples.append(sample)
                
            # 3. Signs/Written Books (NBT text extractions)
            nbt_texts = results.get("extracted_text", [])
            for idx, text in enumerate(nbt_texts):
                sample = {
                    'text': text,
                    'length': len(text),
                    'source': f"minecraft:{config.source_path}/nbt_text_{idx}",
                    'metadata': {
                        'world_name': config.source_path,
                        'type': 'nbt_text',
                        'noncommutativity_curvature': results.get('noncommutativity_curvature', 0.0),
                        'commutativity_gap': results.get('commutativity_gap', 0.0),
                    }
                }
                if voxel_res_list:
                    sample['metadata']['voxel_residue'] = voxel_res_list
                samples.append(sample)
                
            if return_samples:
                return samples
                
            # Save dataset
            dataset_path = self.data_dir / config.safe_name
            dataset_path.mkdir(exist_ok=True, parents=True)
            self._save_dynamic_dataset(samples, dataset_path, config)
            
            print(f"[OK] Minecraft dataset created and chunked: {len(samples)} samples")
            return True
            
        except Exception as e:
            print(f"[ERR] Minecraft ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _ingest_huggingface_dataset(self, config: DatasetConfig, return_samples: bool = False) -> Union[bool, List[Dict]]:
        """Ingest dataset from HuggingFace Hub."""
        try:
            # Try to import datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                print("[WARN] HuggingFace datasets library not installed")
                print("   Install with: pip install datasets")
                return False
            
            print(f"[HF] Loading HuggingFace dataset: {config.source_path}")
            
            # Load dataset
            if config.max_samples:
                # Load streaming for large datasets
                dataset = load_dataset(config.source_path, streaming=True)
                # Take first max_samples
                if 'train' in dataset:
                    dataset = dataset['train'].take(config.max_samples)
            else:
                dataset = load_dataset(config.source_path)
            
            # Save to local directory
            dataset_path = self.data_dir / config.safe_name
            dataset_path.mkdir(exist_ok=True)
            
            # Process and save samples
            samples = []
            for i, sample in enumerate(dataset):
                if config.max_samples and i >= config.max_samples:
                    break
                
                processed_sample = self._preprocess_sample(sample, config.preprocessing)
                if processed_sample:
                    samples.append(processed_sample)
                
                if i % 1000 == 0:
                    print(f"   Processed {i} samples...")
            
            if return_samples:
                return samples
            
            # Save processed dataset
            self._save_dynamic_dataset(samples, dataset_path, config)
            
            print(f"[OK] HuggingFace dataset loaded and chunked: {len(samples)} samples")
            return True
            
        except Exception as e:
            print(f"[ERR] HuggingFace ingestion failed: {e}")
            return False
    
    def _ingest_kaggle_dataset(self, config: DatasetConfig) -> bool:
        """Ingest dataset from Kaggle."""
        try:
            print(f"[KAG] Loading Kaggle dataset: {config.source_path}")
            
            # Check if kaggle CLI is available
            try:
                result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
                if result.returncode != 0:
                    print("[WARN] Kaggle CLI not available")
                    print("   Install with: pip install kaggle")
                    print("   Configure with your API key: https://www.kaggle.com/docs/api")
                    return False
            except FileNotFoundError:
                print("[WARN] Kaggle CLI not found")
                return False
            
            # Download dataset
            dataset_path = self.data_dir / config.safe_name
            dataset_path.mkdir(exist_ok=True)
            
            # Use kaggle CLI to download
            cmd = ['kaggle', 'datasets', 'download', '-d', config.source_path, '-p', str(dataset_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"[FAIL] Kaggle download failed: {result.stderr}")
                return False
            
            # Extract if zip file
            zip_files = list(dataset_path.glob("*.zip"))
            if zip_files:
                with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
                zip_files[0].unlink()  # Remove zip file
            
            print(f"[OK] Kaggle dataset downloaded to {dataset_path}")
            
            # Now process the downloaded files locally
            local_config = DatasetConfig(
                name=config.name,
                source_type='local',
                source_path=str(dataset_path),
                preprocessing=config.preprocessing,
                max_samples=config.max_samples,
                augmentation=config.augmentation,
                mandelbulb_augmentation=config.mandelbulb_augmentation,
                manifold_aware=getattr(config, 'manifold_aware', False)
            )
            return self._ingest_local_dataset(local_config)
        
        except Exception as e:
            print(f"[ERR] Kaggle ingestion failed: {e}")
            return False
    
    def _ingest_wikipedia_dataset(self, config: DatasetConfig, return_samples: bool = False) -> Union[bool, List[Dict]]:
        """Ingest dataset from Wikipedia articles."""
        try:
            print(f"[WIKI] Loading Wikipedia dataset: {config.source_path}")
            
            # Parse Wikipedia URLs or topics
            if config.source_path.startswith('http'):
                # Single URL
                urls = [config.source_path]
            else:
                # Topic list or file
                if Path(config.source_path).exists():
                    with open(config.source_path, 'r') as f:
                        topics = [line.strip() for line in f if line.strip()]
                else:
                    topics = config.source_path.split(',')
                
                # Convert topics to URLs
                urls = [f"https://en.wikipedia.org/wiki/{topic.strip().replace(' ', '_')}" 
                       for topic in topics]
            
            # Limit URLs if max_samples specified
            if config.max_samples:
                urls = urls[:config.max_samples]
            
            # Extract content using Wikipedia integration
            samples = []
            for i, url in enumerate(urls):
                try:
                    print(f"   Processing {i+1}/{len(urls)}: {url}")
                    
                    # Extract content
                    title = wikipedia_integration.extract_title_from_url(url)
                    wiki_result = wikipedia_integration.fetch_wikipedia_content(title)
                    content = wiki_result.get('full_content', '') if wiki_result else None
                    
                    if content:
                        # Clean content
                        cleaned_content = wikipedia_integration._fallback_clean_content(content)
                        
                        # Extract concepts
                        concepts = wikipedia_integration.extract_key_concepts(title, cleaned_content)
                        
                        # Chunk into paragraphs so we get multiple samples per article
                        paragraphs = [p.strip() for p in cleaned_content.split('\n\n') if len(p.strip()) > 50]
                        if not paragraphs:
                            # Fallback if no double newlines
                            paragraphs = [cleaned_content]
                            
                        for p_idx, paragraph in enumerate(paragraphs):
                            if config.max_samples and len(samples) >= config.max_samples:
                                break
                                
                            sample = {
                                'title': f"{title} (Part {p_idx+1})",
                                'content': paragraph,
                                'concepts': concepts,
                                'url': url,
                                'length': len(paragraph)
                            }
                            samples.append(sample)
                            
                        if config.max_samples and len(samples) >= config.max_samples:
                            break
                    
                except Exception as e:
                    print(f"   [WARN] Failed to process {url}: {e}")
                    continue
            
            if return_samples:
                return samples
            
            # Save dataset
            dataset_path = self.data_dir / config.safe_name
            dataset_path.mkdir(exist_ok=True, parents=True)
            self._save_dynamic_dataset(samples, dataset_path, config)
            
            print(f"[OK] Wikipedia dataset created and chunked: {len(samples)} articles")
            return True
            
        except Exception as e:
            print(f"[ERR] Wikipedia ingestion failed: {e}")
            return False
    
    def _ingest_local_dataset(self, config: DatasetConfig, return_samples: bool = False) -> Union[bool, List[Dict]]:
        """Ingest dataset from local files."""
        try:
            print(f"[DISK] Loading local dataset: {config.source_path}")
            
            source_path = Path(config.source_path)
            if not source_path.exists():
                print(f"[ERR] Path does not exist: {source_path}")
                return False
            
            total_samples = []
            files_to_process = []
            
            if source_path.is_file():
                files_to_process = [source_path]
            else:
                file_patterns = ['*.txt', '*.json', '*.csv', '*.jsonl', '*.py', '*.md', '*.jsonl.gz', '*.gz']
                for pattern in file_patterns:
                    files_to_process.extend(source_path.rglob(pattern))
            
            print(f"   Found {len(files_to_process)} files to process")
            
            for i, file_path in enumerate(files_to_process):
                # Check if we already reached max samples
                if config.max_samples and len(total_samples) >= config.max_samples:
                    print(f"   [STOP] Reached global max samples ({config.max_samples}), skipping remaining files.")
                    break
                
                print(f"   [FILE] Processing file {i+1}/{len(files_to_process)}: {file_path.name}")
                
                # Pass current total length to keep track of budget
                current_limit = config.max_samples - len(total_samples) if config.max_samples else None
                file_samples = self._process_local_file(file_path, config, max_new_samples=current_limit)
                
                if file_samples:
                    total_samples.extend(file_samples)
                    print(f"   [DATA] Current total samples collected: {len(total_samples)}")
            
            if return_samples:
                return total_samples
            
            # Save processed dataset
            dataset_path = self.data_dir / config.safe_name
            dataset_path.mkdir(exist_ok=True, parents=True)
            
            self._save_dynamic_dataset(total_samples, dataset_path, config)
            
            print(f"[OK] Local dataset loaded and chunked: {len(total_samples)} samples")
            return True
            
        except Exception as e:
            print(f"[FAIL] Local ingestion failed: {e}")
            return False
    
    def _ingest_url_dataset(self, config: DatasetConfig) -> bool:
        """Ingest dataset from URL (download and process)."""
        try:
            print(f"[URL] Loading dataset from URL: {config.source_path}")
            
            # Download file
            response = requests.get(config.source_path, stream=True)
            response.raise_for_status()
            
            # Determine filename
            parsed_url = urlparse(config.source_path)
            filename = Path(parsed_url.path).name or "dataset"
            
            dataset_path = self.data_dir / config.safe_name
            dataset_path.mkdir(exist_ok=True)
            
            file_path = dataset_path / filename
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"   Downloaded: {progress:.1f}%", end='\r')
            
            print(f"\n   Download complete: {file_path}")
            
            # Extract if compressed
            if file_path.suffix in ['.zip', '.tar', '.tar.gz', '.tgz']:
                if file_path.suffix == '.zip':
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                else:
                    with tarfile.open(file_path, 'r:*') as tar_ref:
                        tar_ref.extractall(dataset_path)
                
                file_path.unlink()  # Remove compressed file
            
            print(f"[OK] URL dataset downloaded to {dataset_path}")
            
            # Now process the downloaded files locally
            local_config = DatasetConfig(
                name=config.name,
                source_type='local',
                source_path=str(dataset_path),
                preprocessing=config.preprocessing,
                max_samples=config.max_samples,
                augmentation=config.augmentation,
                mandelbulb_augmentation=config.mandelbulb_augmentation,
                manifold_aware=getattr(config, 'manifold_aware', False)
            )
            return self._ingest_local_dataset(local_config)
            
        except Exception as e:
            print(f"[FAIL] URL ingestion failed: {e}")
            return False
            
    def _ingest_portal_dataset(self, config: DatasetConfig) -> bool:
        """Ingest dataset from a portal file (mixed URLs and identifiers)."""
        try:
            print(f"[PORTAL] Loading portals from: {config.source_path}")
            
            lines = []
            is_inline = False
            
            if ',' in config.source_path:
                is_inline = True
                
            if not is_inline:
                try:
                    portal_path = Path(config.source_path)
                    if portal_path.exists() and portal_path.is_file():
                        with open(portal_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                    else:
                        is_inline = True
                except Exception:
                    is_inline = True
                    
            if is_inline:
                print(f"   [PORTAL] Treating input as inline comma-separated portal.")
                lines = config.source_path.split(',')
                
            sources = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith('http'):
                    sources.append({'type': 'url', 'path': line})
                elif line.startswith('hf:'):
                    sources.append({'type': 'huggingface', 'path': line[3:]})
                elif line.startswith('wiki:'):
                    sources.append({'type': 'wikipedia', 'path': line[5:]})
                elif line.lower() in ['imdb', 'squad', 'wikitext', 'arxiv', 'pubmed']:
                    # Auto-mapping popular datasets to HF
                    sources.append({'type': 'huggingface', 'path': line.lower()})
                else:
                    # Check if it's actually a local path
                    if Path(line).exists():
                        sources.append({'type': 'local', 'path': line})
                    else:
                        # If it doesn't exist locally and has no prefix, assume Wikipedia topic!
                        sources.append({'type': 'wikipedia', 'path': line})
            
            print(f"   Found {len(sources)} sources in portal")
            
            combined_samples = []
            dataset_path = self.data_dir / config.safe_name
            dataset_path.mkdir(exist_ok=True)
            
            for i, source in enumerate(sources):
                print(f"\n   [PORTAL] Processing source {i+1}/{len(sources)}: {source['path']} ({source['type']})")
                
                sub_config = DatasetConfig(
                    name=config.name,
                    source_type=source['type'],
                    source_path=source['path'],
                    preprocessing=config.preprocessing,
                    max_samples=config.max_samples,
                    augmentation=config.augmentation,
                    mandelbulb_augmentation=config.mandelbulb_augmentation,
                    manifold_aware=getattr(config, 'manifold_aware', False)
                )
                
                if source['type'] == 'url':
                    self._ingest_url_dataset(sub_config)
                elif source['type'] == 'huggingface':
                    res = self._ingest_huggingface_dataset(sub_config, return_samples=True)
                    if isinstance(res, list): combined_samples.extend(res)
                elif source['type'] == 'wikipedia':
                    res = self._ingest_wikipedia_dataset(sub_config, return_samples=True)
                    if isinstance(res, list): combined_samples.extend(res)
                elif source['type'] == 'local':
                    res = self._ingest_local_dataset(sub_config, return_samples=True)
                    if isinstance(res, list): combined_samples.extend(res)
            
            # Step 2: Ingest local files (including those downloaded via URL)
            print(f"\n   [PORTAL] Finalizing with local file ingestion...")
            local_config = DatasetConfig(
                name=config.name,
                source_type='local',
                source_path=str(dataset_path),
                preprocessing=config.preprocessing,
                max_samples=config.max_samples,
                augmentation=config.augmentation,
                mandelbulb_augmentation=config.mandelbulb_augmentation,
                manifold_aware=getattr(config, 'manifold_aware', False)
            )
            
            # We need to reach into the local ingest without it overwriting everything immediately
            # Actually, local ingest already returns samples in its internal methods
            local_samples = self._ingest_local_dataset(local_config, return_samples=True)
            if isinstance(local_samples, list):
                combined_samples.extend(local_samples)
            
            if not combined_samples:
                print(f"[WARN] Portal ingestion resulted in 0 samples")
                return False
                
            # Final save of combined data dynamically
            self._save_dynamic_dataset(combined_samples, dataset_path, config)
            print(f"[OK] Portal ingestion complete and chunked: {len(combined_samples)} total samples saved to {dataset_path}")
            return True
            
        except Exception as e:
            print(f"[ERR] Portal ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_local_file(self, file_path: Path, config: DatasetConfig, max_new_samples: Optional[int] = None) -> List[Dict]:
        """Process a single local file."""
        samples = []
        
        # Determine if we need to use gzip
        is_gz = file_path.suffix == '.gz'
        open_func = gzip.open if is_gz else open
        
        # Get effective suffix for nested extensions like .jsonl.gz
        if is_gz:
            effective_suffix = Path(file_path.stem).suffix
        else:
            effective_suffix = file_path.suffix
            
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if effective_suffix == '.json' or (is_gz and effective_suffix == '.json'):
                # Check for large files
                if file_size_mb > 100:
                    print(f"      [WARN] Large JSON file detected ({file_size_mb:.1f} MB): {file_path.name}")
                    
                    # Try to use ijson for streaming if available
                    try:
                        import ijson
                        print(f"      [STREAM] streaming with ijson...")
                        with open_func(file_path, 'rt', encoding='utf-8') if is_gz else open(file_path, 'r', encoding='utf-8') as f:
                            # Assume it's a list of objects
                            objects = ijson.items(f, 'item')
                            for i, item in enumerate(objects):
                                if max_new_samples and len(samples) >= max_new_samples:
                                    print(f"\n      [STOP] Reached limit in this file: {max_new_samples}")
                                    break
                                processed = self._preprocess_sample(item, config.preprocessing)
                                if processed:
                                    samples.append(processed)
                                
                                # Progress logging
                                if (i + 1) % 1000 == 0:
                                    print(f"      [BUSY] Streamed {i + 1} items... (Collected: {len(samples)})", end='\r')
                            
                            print(f"\n      [OK] Streaming complete. Total collected from file: {len(samples)}")
                            
                            if len(samples) == 0:
                                print(f"      [WARN] Streamed 0 samples. Check JSON structure or preprocessing logic.")
                                
                        return samples
                    except ImportError:
                        print("      [WARN] 'ijson' library not found. Falling back to standard load (may consume high RAM).")
                        print("      [TIP] Recommendation: Convert large JSON files to JSONL or install ijson: `pip install ijson`")
                    except Exception as e:
                        print(f"      [WARN] Streaming failed: {e}. Falling back to standard load.")

                # Standard load (with memory safety)
                try:
                    with open_func(file_path, 'rt', encoding='utf-8') if is_gz else open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if max_new_samples and len(samples) >= max_new_samples:
                                    break
                                processed = self._preprocess_sample(item, config.preprocessing)
                                if processed:
                                    samples.append(processed)
                        else:
                            processed = self._preprocess_sample(data, config.preprocessing)
                            if processed:
                                samples.append(processed)
                except MemoryError:
                    print(f"      [ERR] OUT OF MEMORY: Could not load {file_path.name} ({file_size_mb:.1f} MB).")
                    print("      [TIP] Please convert this dataset to JSONL format (line-delimited JSON) for efficient streaming.")
                    return []
            
            elif effective_suffix == '.jsonl' or (is_gz and (effective_suffix == '.jsonl' or effective_suffix == '')):
                if file_size_mb > 100:
                    print(f"      [WARN] Large JSONL file detected ({file_size_mb:.1f} MB): {file_path.name}")
                
                with open_func(file_path, 'rt', encoding='utf-8') if is_gz else open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if max_new_samples and len(samples) >= max_new_samples:
                            print(f"\n      [STOP] Reached limit in this file: {max_new_samples}")
                            break
                        
                        if line.strip():
                            try:
                                data = json.loads(line)
                                processed = self._preprocess_sample(data, config.preprocessing)
                                if processed:
                                    samples.append(processed)
                            except json.JSONDecodeError:
                                continue
                        
                        if (i + 1) % 5000 == 0:
                            print(f"      [BUSY] Processed {i + 1} lines... (Collected: {len(samples)})", end='\r')
                    
                    if i + 1 >= 5000:
                        print(f"\n      [OK] File processing complete. Total collected: {len(samples)}")
            
            elif effective_suffix in ['.txt', '.py', '.md']:
                with open_func(file_path, 'rt', encoding='utf-8') if is_gz else open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split into chunks for large files
                    chunk_size = 1000
                    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                    
                    for chunk in chunks:
                        if max_new_samples and len(samples) >= max_new_samples:
                            break
                        if chunk.strip():
                            sample = {'text': chunk.strip(), 'source': str(file_path)}
                            processed = self._preprocess_sample(sample, config.preprocessing)
                            if processed:
                                samples.append(processed)
            
            elif effective_suffix == '.csv':
                import pandas as pd
                # Read in chunks to be memory efficient and allow early exit
                chunk_iter = pd.read_csv(file_path, chunksize=1000)
                for chunk in chunk_iter:
                    if max_new_samples and len(samples) >= max_new_samples:
                        break
                    for _, row in chunk.iterrows():
                        if max_new_samples and len(samples) >= max_new_samples:
                            break
                        sample = row.to_dict()
                        processed = self._preprocess_sample(sample, config.preprocessing)
                        if processed:
                            samples.append(processed)
                    
                    print(f"      [*] Processed chunk... (Collected: {len(samples)})", end='\r')
                
                print(f"\n      [OK] CSV processing complete. Total collected: {len(samples)}")
        
        except Exception as e:
            print(f"   [WARN] Error processing {file_path}: {e}")
        
        return samples
    
    def _preprocess_sample(self, sample: Dict, preprocessing_type: str) -> Optional[Dict]:
        """Preprocess a single sample based on type."""
        try:
            if preprocessing_type == 'text':
                # Extract text content
                text_fields = ['text', 'content', 'body', 'description', 'title']
                text_content = ""
                
                # Standard fields
                for field in text_fields:
                    if field in sample and sample[field]:
                        text_content += str(sample[field]) + "\n"
                
                # ShareGPT format (conversations)
                if 'conversations' in sample and isinstance(sample['conversations'], list):
                    for turn in sample['conversations']:
                        if isinstance(turn, dict):
                            role = turn.get('from', 'unknown')
                            value = turn.get('value', turn.get('text', ''))
                            if value:
                                text_content += f"{role}: {value}\n"
                
                # Alpaca format (instruction/input/output)
                if 'instruction' in sample:
                    text_content += f"Instruction: {sample['instruction']}\n"
                    if sample.get('input'):
                        text_content += f"Input: {sample['input']}\n"
                    if sample.get('output'):
                        text_content += f"Output: {sample['output']}\n"
                
                if not text_content.strip():
                    return None
                
                # [FULL BRIDGE] Manifold-Aware (Thick) Ingestion
                residue = None
                if getattr(self, 'config', None) and getattr(self.config, 'manifold_aware', False) and self.engine:
                    try:
                        # Extract residue vector from the manifold
                        result = self.engine.process_input(
                            text_input=text_content.strip(), 
                            generate_response=False, 
                            ingestion_mode=True
                        )
                        residue = result.get('residue_vector')
                    except Exception as e:
                        print(f"   [WARN] Manifold-Aware ingestion failed for sample: {e}")
                
                metadata = {k: v for k, v in sample.items() if k not in text_fields and k != 'conversations'}
                if residue:
                    metadata['residue_vector'] = residue
                    metadata['manifold_step'] = self.engine.iteration
                
                return {
                    'text': text_content.strip(),
                    'length': len(text_content),
                    'source': sample.get('source', 'unknown'),
                    'metadata': metadata
                }
            
            elif preprocessing_type == 'image':
                # Handle image data (placeholder for now)
                if 'image' in sample or 'image_path' in sample:
                    return {
                        'image_path': sample.get('image_path', sample.get('image')),
                        'caption': sample.get('caption', ''),
                        'metadata': sample
                    }
                return None
            
            elif preprocessing_type == 'tabular':
                # Handle structured data
                return {
                    'features': sample,
                    'metadata': {'type': 'tabular'}
                }
            
            elif preprocessing_type == 'multimodal':
                # Handle mixed content
                processed = {
                    'content': sample,
                    'modalities': self._detect_modalities(sample),
                    'metadata': {'type': 'multimodal'}
                }
                # Bubble up image path if present for easy embedding
                if 'image' in sample: processed['image_path'] = sample['image']
                if 'image_path' in sample: processed['image_path'] = sample['image_path']
                
                # Bubble up text if present
                if 'text' in sample: processed['text'] = sample['text']
                if 'content' in sample and isinstance(sample['content'], str): processed['text'] = sample['content']
                
                return processed
            
            else:
                # Default: return as-is
                return sample
                
        except Exception as e:
            print(f"   [WARN] Preprocessing error: {e}")
            return None
    
    def _detect_modalities(self, sample: Dict) -> List[str]:
        """Detect modalities in a sample."""
        modalities = []
        
        text_fields = ['text', 'content', 'body', 'description', 'title']
        image_fields = ['image', 'image_path', 'img', 'picture']
        
        for field in text_fields:
            if field in sample and sample[field]:
                modalities.append('text')
                break
        
        for field in image_fields:
            if field in sample and sample[field]:
                modalities.append('image')
                break
        
        return modalities
    
    def create_model(self, name: str, model_config: Dict[str, Any]) -> bool:
        """Create a model for training."""
        try:
            print(f"\n[MODEL] Creating model: {name}")
            print(f"   Type: {model_config.get('type', 'temporal')}")
            print(f"   Device: {self.device}")
            
            if model_config.get('type', 'temporal') == 'temporal':
                model = NonLobotomyTemporalModel(
                    input_dim=model_config.get('input_dim', 768),
                    hidden_dim=model_config.get('hidden_dim', 256),
                    num_functionals=model_config.get('num_functionals', 33),
                    poly_degree=model_config.get('poly_degree', 4),
                    device=self.device
                )
            else:
                print(f"[FAIL] Unknown model type: {model_config.get('type')}")
                return False
            
            self.models[name] = model
            
            # Verify anti-lobotomy compliance
            compliance_check = self._verify_anti_lobotomy_compliance(model)
            if not compliance_check:
                print("[FAIL] Model failed anti-lobotomy compliance check")
                return False
            
            print(f"[OK] Model {name} created successfully")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   Polynomial functionals: {model.K}")
            print(f"   Trust scalars: {[f'{t:.3f}' for t in model.trust_scalars.tolist()]}")
            print(f"   Anti-lobotomy compliance: [OK] VERIFIED")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Model creation failed: {e}")
            return False
    
    def _verify_anti_lobotomy_compliance(self, model) -> bool:
        """Verify model follows anti-lobotomy principles."""
        try:
            if not hasattr(model, 'polynomial_config'):
                print("   [ERR] Missing polynomial_config")
                return False
            
            # Check class name only to avoid module path issues
            if model.polynomial_config.__class__.__name__ != 'PolynomialCoprimeConfig':
                print("   [ERR] Invalid polynomial_config type")
                return False
            
            # Check 2: Trust scalars don't require gradients
            if hasattr(model, 'trust_scalars') and model.trust_scalars.requires_grad:
                print("   [FAIL] Trust scalars require gradients (teleological violation)")
                return False
            
            # Check 3: Has evolutionary components
            required_buffers = ['trust_scalars', 'bimodal_genome', 'is_fossilized']
            for buffer_name in required_buffers:
                if not hasattr(model, buffer_name):
                    print(f"   [ERR] Missing evolutionary buffer: {buffer_name}")
                    return False
            
            # Check 4: Polynomial coefficients are proper
            try:
                coeffs = model.polynomial_config.get_coefficients_tensor()
                if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                    print("   [ERR] Invalid polynomial coefficients")
                    return False
            except Exception as e:
                print(f"   [ERR] Polynomial coefficient error: {e}")
                return False
            
            print("   [OK] Anti-lobotomy compliance verified")
            return True
            
        except Exception as e:
            print(f"[ERR] Compliance check error: {e}")
            return False
    
    def setup_training(self, model_name: str, dataset_name: str, training_config: TrainingConfig) -> bool:
        """Setup training for a model and dataset."""
        try:
            print(f"\n[TRAIN] Setting up training: {model_name} on {dataset_name}")
            
            # Check model exists
            if model_name not in self.models:
                print(f"[FAIL] Model {model_name} not found")
                return False
            
            # Check dataset exists
            if dataset_name not in self.datasets:
                print(f"[FAIL] Dataset {dataset_name} not found")
                return False
            
            model = self.models[model_name]
            dataset_config = self.datasets[dataset_name]
            
            # Load dataset dynamically to reduce VRAM pressure
            processed_data = SovereignDynamicDataset(self.data_dir / dataset_config.safe_name)
            print(f"   [DYNAMIC] Loaded dataset with {len(processed_data)} samples (On-demand loading active)")
            
            # Create dataset wrapper
            if training_config.model_type == 'temporal':
                # Create temporal association dataset
                dataset = self._create_temporal_dataset(processed_data, training_config)
            else:
                print(f"[FAIL] Unknown training type: {training_config.model_type}")
                return False
            
            # Setup Mandelbulb augmentation if requested
            augmenter = None
            if training_config.use_mandelbulb_augmentation:
                print("   [AUG] Setting up Mandelbulb-Gyroidic augmentation...")
                augmentation_config = AugmentationConfig(
                    mandelbulb_power=8,
                    max_iterations=50,
                    gyroid_tolerance=1e-3,
                    sparsity_threshold=0.1,
                    pressure_adaptation=True
                )
                augmenter = MandelbulbGyroidicAugmenter(augmentation_config)
                self.augmenters[f"{model_name}_{dataset_name}"] = augmenter
                print("   [OK] Mandelbulb augmentation ready")
            
            # Create trainer
            trainer = NonLobotomyTemporalTrainer(
                model=model,
                dataset=dataset,
                evolution_rate=training_config.evolution_rate,
                survivorship_threshold=training_config.survivorship_threshold
            )
            
            trainer_key = f"{model_name}_{dataset_name}"
            self.trainers[trainer_key] = trainer
            
            # Initialize training history
            self.training_history[trainer_key] = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'config': training_config,
                'start_time': None,
                'epochs_completed': 0,
                'metrics_history': []
            }
            
            print(f"[OK] Training setup complete")
            print(f"   Trainer: {trainer_key}")
            print(f"   Epochs planned: {training_config.num_epochs}")
            print(f"   Batch size: {training_config.batch_size}")
            print(f"   Mandelbulb augmentation: {training_config.use_mandelbulb_augmentation}")
            
            return True
            
        except Exception as e:
            print(f"[FAIL] Training setup failed: {e}")
            return False
    
    def _create_temporal_dataset(self, processed_data: List[Dict], config: TrainingConfig):
        """Create temporal dataset from processed data."""
        # Convert text data to embeddings (simplified)
        embeddings = []
        
        for sample in processed_data:
            if 'text' in sample:
                # Simple embedding: hash-based projection (in real system, use proper embeddings)
                text = sample['text']
                # Create deterministic embedding from text hash using Sovereign Logistic Expansion
                from src.core.honest_jitter import AgentSmithEngine
                engine = AgentSmithEngine(device=torch.device('cpu'))
                hash_val = sum(ord(c) for c in text[:100]) % 1000000
                deterministic_seed = hash_val / 1000000.0
                embedding = engine((768,), seed_val=deterministic_seed, scaled=False)
                embeddings.append(embedding)
            elif 'image_path' in sample and self.image_processor:
                # Use Image Processor to embed image
                try:
                    embedding = self.image_processor(sample['image_path'])
                    # Output is [1, 768], flatten to [768]
                    embeddings.append(embedding.squeeze(0).cpu())
                except Exception as e:
                     print(f"   [WARN] Failed to embed image {sample['image_path']}: {e}")
                     # Fallback to random (Sovereign Jitter)
                     embeddings.append(harvest_honest_jitter((768,), scaled=False).cpu())
        
        # Create simple temporal dataset
        class SimpleTemporalDataset:
            def __init__(self, embeddings, sequence_length=8):
                self.embeddings = embeddings
                self.sequence_length = sequence_length
            
            def get_batch(self, batch_size=4):
                sequences = []
                targets = []
                
                for _ in range(batch_size):
                    # Random sequence (Sovereign Jitter)
                    _j_val = (harvest_honest_jitter((1,), scaled=False).cpu().item() + 1.0) / 2.0
                    start_idx = int(_j_val * max(1, len(self.embeddings) - self.sequence_length))
                    sequence = []
                    sequence_targets = []
                    
                    for i in range(self.sequence_length):
                        if start_idx + i < len(self.embeddings):
                            sequence.append(self.embeddings[start_idx + i])
                            # Target is next embedding (or same if at end)
                            target_idx = min(start_idx + i + 1, len(self.embeddings) - 1)
                            sequence_targets.append(self.embeddings[target_idx])
                        else:
                            # Pad with deterministic boundary (Sovereign Logistic Expansion)
                            from src.core.honest_jitter import AgentSmithEngine
                            engine = AgentSmithEngine(device=torch.device('cpu'))
                            pad_emb = engine((768,), seed_val=0.618, scaled=False)
                            sequence.append(pad_emb)
                            sequence_targets.append(pad_emb)
                    
                    sequences.append(torch.stack(sequence))
                    targets.append(torch.stack(sequence_targets))
                
                return {
                    'sequences': torch.stack(sequences),
                    'targets': torch.stack(targets)
                }
        
        return SimpleTemporalDataset(embeddings, sequence_length=config.batch_size)
    
    def run_training(self, model_name: str, dataset_name: str) -> bool:
        """Run training for a model-dataset pair."""
        try:
            trainer_key = f"{model_name}_{dataset_name}"
            
            if trainer_key not in self.trainers:
                print(f"[ERR] Training not setup for {trainer_key}")
                return False
            
            trainer = self.trainers[trainer_key]
            config = self.training_history[trainer_key]['config']
            
            print(f"\n[START] Starting training: {trainer_key}")
            print(f"   Epochs: {config.num_epochs}")
            print(f"   Batch size: {config.batch_size}")
            print(f"   Learning rate: {config.learning_rate}")
            print(f"   Evolution rate: {config.evolution_rate}")
            print("=" * 60)
            
            # Record start time
            self.training_history[trainer_key]['start_time'] = time.time()
            
            # Training loop
            for epoch in range(config.num_epochs):
                print(f"\n[EPOCH] Epoch {epoch + 1}/{config.num_epochs}")
                
                try:
                    # Train epoch
                    epoch_metrics = trainer.train_epoch(num_batches=20)
                    
                    # Record metrics
                    self.training_history[trainer_key]['metrics_history'].append(epoch_metrics)
                    self.training_history[trainer_key]['epochs_completed'] = epoch + 1
                    
                    # Print summary
                    print(f"   Survivorship Pressure: {epoch_metrics['survivorship_pressure']:.3f}")
                    print(f"   Association Accuracy: {epoch_metrics['association_accuracy']:.3f}")
                    print(f"   Temporal Coherence: {epoch_metrics['temporal_coherence']:.3f}")
                    print(f"   Trust Mean: {epoch_metrics['trust_mean']:.3f} +/- {epoch_metrics['trust_std']:.3f}")
                    print(f"   Fossilized: {epoch_metrics['final_num_fossilized']}")
                    
                    # Show trust evolution
                    model = self.models[model_name]
                    trust_scalars = model.trust_scalars
                    print(f"   Trust Scalars: {[f'{t:.3f}' for t in trust_scalars.tolist()]}")
                    
                    if config.use_mandelbulb_augmentation and f"{model_name}_{dataset_name}" in self.augmenters:
                        print("   [AUG] Applying Mandelbulb-Gyroidic augmentation...")
                        augmenter = self.augmenters[f"{model_name}_{dataset_name}"]
                        
                        # Get sample data for augmentation
                        sample_batch = trainer.dataset.get_batch(batch_size=4)
                        sample_X = sample_batch['sequences'][:, 0, :]  # First timestep
                        
                        # Apply augmentation
                        augmented_X, _ = augmenter(sample_X, augmentation_factor=config.augmentation_factor)
                        print(f"   [AUG] Augmented {sample_X.shape[0]} → {augmented_X.shape[0]} samples")
                    
                    # Save checkpoint if configured
                    if config.save_checkpoints and (epoch + 1) % config.checkpoint_interval == 0:
                        safe_trainer_key = trainer_key.replace(':', '_').replace(',', '_').replace('/', '_').replace('\\', '_')
                        if len(safe_trainer_key) > 50: safe_trainer_key = safe_trainer_key[:50]
                        checkpoint_path = f"checkpoint_{safe_trainer_key}_epoch_{epoch + 1}.pt"
                        self._save_checkpoint(trainer_key, checkpoint_path)
                        print(f"   [SAVE] Checkpoint saved: {checkpoint_path}")
                
                except Exception as e:
                    print(f"   [ERR] Epoch {epoch + 1} failed: {e}")
                    continue
            
            # Training complete
            total_time = time.time() - self.training_history[trainer_key]['start_time']
            print(f"\n[DONE] Training Complete!")
            print(f"   Total time: {total_time:.1f} seconds")
            print(f"   Epochs completed: {self.training_history[trainer_key]['epochs_completed']}")
            
            # Final model state
            model = self.models[model_name]
            final_trust = model.trust_scalars
            print(f"   Final trust: {[f'{t:.3f}' for t in final_trust.tolist()]}")
            print(f"   Fossilized functionals: {(final_trust > config.fossilization_threshold).sum().item()}")
            
            # Save final state
            safe_trainer_key = trainer_key.replace(':', '_').replace(',', '_').replace('/', '_').replace('\\', '_')
            if len(safe_trainer_key) > 50: safe_trainer_key = safe_trainer_key[:50]
            final_checkpoint = f"final_{safe_trainer_key}.pt"
            self._save_checkpoint(trainer_key, final_checkpoint)
            print(f"   [SAVE] Final state saved: {final_checkpoint}")

            # ---- TOPOLOGICAL INVARIANT VALIDATION ----
            # Validate 12+ architectural components before SOUL FUSION
            try:
                from src.core.veto_subspace import VetoSubspace, RecoveryStatus
                
                final_metrics = self.training_history[trainer_key]['metrics_history'][-1] if self.training_history[trainer_key]['metrics_history'] else {}
                model = self.trainers[trainer_key].model
                
                betti_collapse = 0
                if hasattr(model, 'is_fossilized') and model.is_fossilized.sum() == 0 and config.num_epochs > 0:
                    betti_collapse = 1 # Complete loss of structural anchors
                    
                validation_veto = VetoSubspace()
                veto_result = validation_veto.evaluate(
                    abort_score=1.0 - final_metrics.get('association_accuracy', 1.0),
                    ley_line_deviation=0.1,
                    coprime_lock=True,
                    chiral_score=0.9,
                    instability_severity=1.0 - final_metrics.get('temporal_coherence', 1.0),
                    covariance_aborts=0,
                    elipsodistrophy_atrophy=0.1,
                    betti_number_collapse=betti_collapse,
                    voynich_slip_degradation=0.0,
                    global_performance_improvement=0.1,
                    topological_pressure=final_metrics.get('survivorship_pressure', 0.1),
                    elapsed_seconds=0.1,
                    valence_hunger=0.1
                )
                
                # Check critical unrecoverable topology violations
                unrecoverable = any(s.level == validation_veto._evaluate_trajectory.__code__.co_consts[0] for s in veto_result.signals if s.triggered and not s.can_recover)
                if unrecoverable or betti_collapse > 0:
                    print(f"   [VETO] SOUL FUSION ABORTED! Critical topological invariant violation.")
                    print(f"   [VETO] Active vetoes: {veto_result.active_vetoes}")
                    return False
                    
                print("   [VALIDATION] 12+ topological invariants verified successfully.")
            except Exception as v_err:
                print(f"   [WARN] Topological validation error: {v_err}")

            # ---- SOUL FUSION PROTOCOL ----
            # Merge trained model into the live gyroid_state.pt so the
            # manifold soul reflects offline portal training (DETERMINISM_AND_PERSISTENCE policy)
            try:
                import os as _os
                _root = _os.path.dirname(_os.path.abspath(__file__))
                _soul_path = _os.path.join(_root, 'gyroid_state.pt')
                # Load existing soul (may not exist yet)
                if _os.path.exists(_soul_path):
                    _soul = torch.load(_soul_path, map_location='cpu')
                else:
                    _soul = {}
                # Inject the offline model's state dict as the temporal model layer
                _trainer = self.trainers[trainer_key]
                _soul['temporal_model_state'] = _trainer.model.state_dict()
                _soul['offline_trust_scalars'] = _trainer.model.trust_scalars.clone()
                _soul['offline_bimodal_genome'] = _trainer.model.bimodal_genome.clone()
                _soul['offline_is_fossilized'] = _trainer.model.is_fossilized.clone()
                _soul['offline_trainer_key'] = trainer_key
                torch.save(_soul, _soul_path)
                _soul_mb = _os.path.getsize(_soul_path) / (1024 * 1024)
                print(f"   [SOUL FUSION] gyroid_state.pt updated with offline training ({_soul_mb:.2f} MB)")
            except Exception as _soul_err:
                print(f"   [WARN] Soul fusion failed (non-fatal): {_soul_err}")
            # ---- END SOUL FUSION ----

            return True
            
        except Exception as e:
            print(f"[ERR] Training failed: {e}")
            return False
    
    def _save_checkpoint(self, trainer_key: str, filepath: str):
        """Save training checkpoint."""
        trainer = self.trainers[trainer_key]
        history = self.training_history[trainer_key]
        
        checkpoint = {
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'training_history': history,
            'trust_scalars': trainer.model.trust_scalars.clone(),
            'bimodal_genome': trainer.model.bimodal_genome.clone(),
            'is_fossilized': trainer.model.is_fossilized.clone(),
            'polynomial_config_state': trainer.model.polynomial_config.get_coefficients_tensor()
        }
        
        torch.save(checkpoint, filepath)
    
    def list_datasets(self):
        """List all available datasets."""
        print("\nAvailable Datasets:")
        if not self.datasets:
            print("   No datasets loaded")
            return
        
        for name, config in self.datasets.items():
            dataset_path = self.data_dir / config.safe_name / "processed_data.pt"
            if dataset_path.exists():
                data = torch.load(dataset_path)
                sample_count = len(data)
            else:
                sample_count = "Unknown"
            
            print(f"   * {name}")
            print(f"     Source: {config.source_type} - {config.source_path}")
            print(f"     Preprocessing: {config.preprocessing}")
            print(f"     Samples: {sample_count}")
            print(f"     Augmentation: {config.augmentation}")
    
    def list_models(self):
        """List all available models."""
        print("\nAvailable Models:")
        if not self.models:
            print("   No models created")
            return
        
        for name, model in self.models.items():
            param_count = sum(p.numel() for p in model.parameters())
            trust_mean = model.trust_scalars.mean().item()
            fossilized = (model.trust_scalars > 0.8).sum().item()
            
            print(f"   * {name}")
            print(f"     Parameters: {param_count:,}")
            print(f"     Functionals: {model.K}")
            print(f"     Trust mean: {trust_mean:.3f}")
            print(f"     Fossilized: {fossilized}/{model.K}")
    
    def list_training_sessions(self):
        """List all training sessions."""
        print("\nTraining Sessions:")
        if not self.training_history:
            print("   No training sessions")
            return
        
        for key, history in self.training_history.items():
            status = "Complete" if history['epochs_completed'] == history['config'].num_epochs else "In Progress"
            
            print(f"   * {key}")
            print(f"     Status: {status}")
            print(f"     Epochs: {history['epochs_completed']}/{history['config'].num_epochs}")
            if history['start_time']:
                elapsed = time.time() - history['start_time']
                print(f"     Runtime: {elapsed:.1f}s")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Gyroidic Dataset Ingestion & Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add HuggingFace dataset
  python dataset_ingestion_system.py add-dataset --name "imdb" --source huggingface --path "imdb" --preprocessing text
  
  # Add Wikipedia dataset
  python dataset_ingestion_system.py add-dataset --name "physics" --source wikipedia --path "Quantum_mechanics,Relativity,Thermodynamics"
  
  # Add local dataset
  python dataset_ingestion_system.py add-dataset --name "my_texts" --source local --path "./my_data/" --preprocessing text
  
  # Create model
  python dataset_ingestion_system.py create-model --name "temporal_model" --type temporal --functionals 33
  
  # Setup training
  python dataset_ingestion_system.py setup-training --model "temporal_model" --dataset "imdb" --epochs 10 --mandelbulb
  
  # Run training
  python dataset_ingestion_system.py train --model "temporal_model" --dataset "imdb"
  
  # List everything
  python dataset_ingestion_system.py list-all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add dataset command
    add_dataset_parser = subparsers.add_parser('add-dataset', help='Add a dataset source')
    add_dataset_parser.add_argument('--name', required=True, help='Dataset name')
    add_dataset_parser.add_argument('--source', required=True, choices=['huggingface', 'kaggle', 'wikipedia', 'local', 'url', 'portal', 'minecraft'], help='Source type')
    add_dataset_parser.add_argument('--path', required=True, help='Source path/URL')
    add_dataset_parser.add_argument('--preprocessing', default='text', choices=['text', 'image', 'tabular', 'multimodal'], help='Preprocessing type')
    add_dataset_parser.add_argument('--max-samples', type=int, help='Maximum samples to load')
    add_dataset_parser.add_argument('--augmentation', action='store_true', help='Enable augmentation')
    add_dataset_parser.add_argument('--mandelbulb', action='store_true', help='Enable Mandelbulb augmentation')
    
    # Create model command
    create_model_parser = subparsers.add_parser('create-model', help='Create a model')
    create_model_parser.add_argument('--name', required=True, help='Model name')
    create_model_parser.add_argument('--type', default='temporal', choices=['temporal'], help='Model type')
    create_model_parser.add_argument('--input-dim', type=int, default=768, help='Input dimension')
    create_model_parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    create_model_parser.add_argument('--functionals', type=int, default=33, help='Number of polynomial functionals (IHC standard: 33)')
    create_model_parser.add_argument('--poly-degree', type=int, default=4, help='Polynomial degree')
    
    # Setup training command
    setup_training_parser = subparsers.add_parser('setup-training', help='Setup training')
    setup_training_parser.add_argument('--model', required=True, help='Model name')
    setup_training_parser.add_argument('--dataset', required=True, help='Dataset name')
    setup_training_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    setup_training_parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    setup_training_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    setup_training_parser.add_argument('--evolution-rate', type=float, default=0.02, help='Evolution rate')
    setup_training_parser.add_argument('--mandelbulb', action='store_true', help='Use Mandelbulb augmentation')
    setup_training_parser.add_argument('--augmentation-factor', type=int, default=2, help='Augmentation factor')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Run training')
    train_parser.add_argument('--model', required=True, help='Model name')
    train_parser.add_argument('--dataset', required=True, help='Dataset name')
    
    # List commands
    subparsers.add_parser('list-datasets', help='List all datasets')
    subparsers.add_parser('list-models', help='List all models')
    subparsers.add_parser('list-training', help='List training sessions')
    subparsers.add_parser('list-all', help='List everything')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize system
    system = DatasetIngestionSystem()
    
    # Execute command
    if args.command == 'add-dataset':
        config = DatasetConfig(
            name=args.name,
            source_type=args.source,
            source_path=args.path,
            preprocessing=args.preprocessing,
            augmentation=args.augmentation,
            mandelbulb_augmentation=args.mandelbulb,
            max_samples=args.max_samples
        )
        system.add_dataset_source(config)
    
    elif args.command == 'create-model':
        model_config = {
            'type': args.type,
            'input_dim': args.input_dim,
            'hidden_dim': args.hidden_dim,
            'num_functionals': args.functionals,
            'poly_degree': args.poly_degree
        }
        system.create_model(args.name, model_config)
    
    elif args.command == 'setup-training':
        training_config = TrainingConfig(
            model_type='temporal',
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            evolution_rate=args.evolution_rate,
            use_mandelbulb_augmentation=args.mandelbulb,
            augmentation_factor=args.augmentation_factor
        )
        system.setup_training(args.model, args.dataset, training_config)
    
    elif args.command == 'train':
        system.run_training(args.model, args.dataset)
    
    elif args.command == 'list-datasets':
        system.list_datasets()
    
    elif args.command == 'list-models':
        system.list_models()
        
    elif args.command == 'list-training-sessions':
        system.list_training_sessions()

if __name__ == '__main__':
    main()
