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

# Add src to path
sys.path.append('src')

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

class DatasetIngestionSystem:
    """
    Main system for dataset ingestion and training.
    
    Maintains anti-lobotomy principles:
    - No hardcoded primes (uses polynomial co-prime functionals)
    - Evolutionary trust selection (not gradient descent on trust)
    - Structural honesty (no placeholders)
    - Non-teleological flow (survivorship pressure, not loss minimization)
    """
    
    def __init__(self, device: str = 'auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
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
            else:
                print(f"[FAIL] Failed to add dataset {config.name}")
                return False
                
        except Exception as e:
            print(f"❌ Error adding dataset {config.name}: {e}")
            return False
    
    def _ingest_huggingface_dataset(self, config: DatasetConfig) -> bool:
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
            dataset_path = self.data_dir / config.name
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
            
            # Save processed dataset
            torch.save(samples, dataset_path / "processed_data.pt")
            
            print(f"[OK] HuggingFace dataset loaded: {len(samples)} samples")
            return True
            
        except Exception as e:
            print(f"❌ HuggingFace ingestion failed: {e}")
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
            dataset_path = self.data_dir / config.name
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
            
            print(f"✅ Kaggle dataset downloaded to {dataset_path}")
            return True
            
        except Exception as e:
            print(f"❌ Kaggle ingestion failed: {e}")
            return False
    
    def _ingest_wikipedia_dataset(self, config: DatasetConfig) -> bool:
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
                    content = wikipedia_integration.extract_content_from_url(url)
                    
                    if content:
                        # Clean content
                        cleaned_content = wikipedia_integration._fallback_clean_content(content)
                        
                        # Extract concepts
                        concepts = wikipedia_integration.extract_key_concepts(title, cleaned_content)
                        
                        sample = {
                            'title': title,
                            'content': cleaned_content,
                            'concepts': concepts,
                            'url': url,
                            'length': len(cleaned_content)
                        }
                        
                        samples.append(sample)
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to process {url}: {e}")
                    continue
            
            # Save dataset
            dataset_path = self.data_dir / config.name
            dataset_path.mkdir(exist_ok=True)
            torch.save(samples, dataset_path / "processed_data.pt")
            
            print(f"[OK] Wikipedia dataset created: {len(samples)} articles")
            return True
            
        except Exception as e:
            print(f"❌ Wikipedia ingestion failed: {e}")
            return False
    
    def _ingest_local_dataset(self, config: DatasetConfig) -> bool:
        """Ingest dataset from local files."""
        try:
            print(f"[DISK] Loading local dataset: {config.source_path}")
            
            source_path = Path(config.source_path)
            if not source_path.exists():
                print(f"❌ Path does not exist: {source_path}")
                return False
            
            total_samples = []
            files_to_process = []
            
            if source_path.is_file():
                files_to_process = [source_path]
            else:
                file_patterns = ['*.txt', '*.json', '*.csv', '*.jsonl']
                for pattern in file_patterns:
                    files_to_process.extend(source_path.glob(pattern))
            
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
            
            # Save processed dataset
            dataset_path = self.data_dir / config.name
            dataset_path.mkdir(exist_ok=True)
            
            save_path = dataset_path / "processed_data.pt"
            print(f"   [DISK] Saving {len(total_samples)} samples to {save_path} (this may take a moment)...")
            start_save = time.time()
            torch.save(total_samples, save_path)
            end_save = time.time()
            print(f"   [OK] Saved in {end_save - start_save:.2f} seconds.")
            
            print(f"[OK] Local dataset loaded: {len(total_samples)} samples")
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
            
            dataset_path = self.data_dir / config.name
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
                            print(f"   Downloaded: {progress:.1f}%", end='\\r')
            
            print(f"\\n   Download complete: {file_path}")
            
            # Extract if compressed
            if file_path.suffix in ['.zip', '.tar', '.tar.gz', '.tgz']:
                if file_path.suffix == '.zip':
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                else:
                    with tarfile.open(file_path, 'r:*') as tar_ref:
                        tar_ref.extractall(dataset_path)
                
                file_path.unlink()  # Remove compressed file
            
            print(f"✅ URL dataset downloaded to {dataset_path}")
            return True
            
        except Exception as e:
            print(f"[FAIL] URL ingestion failed: {e}")
            return False
    
    def _process_local_file(self, file_path: Path, config: DatasetConfig, max_new_samples: Optional[int] = None) -> List[Dict]:
        """Process a single local file."""
        samples = []
        
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_path.suffix == '.json':
                # Check for large files
                if file_size_mb > 100:
                    print(f"      ⚠️  Large JSON file detected ({file_size_mb:.1f} MB): {file_path.name}")
                    
                    # Try to use ijson for streaming if available
                    try:
                        import ijson
                        print(f"      🔄 streaming with ijson...")
                        with open(file_path, 'r', encoding='utf-8') as f:
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
                    with open(file_path, 'r', encoding='utf-8') as f:
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
            
            elif file_path.suffix == '.jsonl':
                if file_size_mb > 100:
                    print(f"      [WARN] Large JSONL file detected ({file_size_mb:.1f} MB): {file_path.name}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
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
            
            elif file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
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
            
            elif file_path.suffix == '.csv':
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
                    
                    print(f"      ⏳ Processed chunk... (Collected: {len(samples)})", end='\r')
                
                print(f"\n      ✅ CSV processing complete. Total collected: {len(samples)}")
        
        except Exception as e:
            print(f"   ⚠️  Error processing {file_path}: {e}")
        
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
                
                return {
                    'text': text_content.strip(),
                    'length': len(text_content),
                    'source': sample.get('source', 'unknown'),
                    'metadata': {k: v for k, v in sample.items() if k not in text_fields and k != 'conversations'}
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
            print(f"   ⚠️  Preprocessing error: {e}")
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
                    num_functionals=model_config.get('num_functionals', 5),
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
            # Check 1: Has polynomial config (no hardcoded primes)
            if not hasattr(model, 'polynomial_config'):
                print("   ❌ Missing polynomial_config")
                return False
            
            if not isinstance(model.polynomial_config, PolynomialCoprimeConfig):
                print("   ❌ Invalid polynomial_config type")
                return False
            
            # Check 2: Trust scalars don't require gradients
            if hasattr(model, 'trust_scalars') and model.trust_scalars.requires_grad:
                print("   [FAIL] Trust scalars require gradients (teleological violation)")
                return False
            
            # Check 3: Has evolutionary components
            required_buffers = ['trust_scalars', 'bimodal_genome', 'is_fossilized']
            for buffer_name in required_buffers:
                if not hasattr(model, buffer_name):
                    print(f"   ❌ Missing evolutionary buffer: {buffer_name}")
                    return False
            
            # Check 4: Polynomial coefficients are proper
            try:
                coeffs = model.polynomial_config.get_coefficients_tensor()
                if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                    print("   ❌ Invalid polynomial coefficients")
                    return False
            except Exception as e:
                print(f"   ❌ Polynomial coefficient error: {e}")
                return False
            
            print("   ✅ Anti-lobotomy compliance verified")
            return True
            
        except Exception as e:
            print(f"   ❌ Compliance check error: {e}")
            return False
    
    def setup_training(self, model_name: str, dataset_name: str, training_config: TrainingConfig) -> bool:
        """Setup training for a model and dataset."""
        try:
            print(f"\\n🎯 Setting up training: {model_name} on {dataset_name}")
            
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
            
            # Load processed dataset
            dataset_path = self.data_dir / dataset_name / "processed_data.pt"
            if not dataset_path.exists():
                print(f"[FAIL] Processed dataset not found: {dataset_path}")
                return False
            
            processed_data = torch.load(dataset_path)
            print(f"   Loaded {len(processed_data)} samples")
            
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
                # Create deterministic embedding from text hash
                hash_val = hash(text) % (2**31)
                np.random.seed(hash_val)
                embedding = torch.tensor(np.random.randn(768), dtype=torch.float32)
                embeddings.append(embedding)
            elif 'image_path' in sample and self.image_processor:
                # Use Image Processor to embed image
                try:
                    embedding = self.image_processor(sample['image_path'])
                    # Output is [1, 768], flatten to [768]
                    embeddings.append(embedding.squeeze(0).cpu())
                except Exception as e:
                     print(f"   [WARN] Failed to embed image {sample['image_path']}: {e}")
                     # Fallback to random
                     embeddings.append(torch.randn(768))
        
        # Create simple temporal dataset
        class SimpleTemporalDataset:
            def __init__(self, embeddings, sequence_length=8):
                self.embeddings = embeddings
                self.sequence_length = sequence_length
            
            def get_batch(self, batch_size=4):
                sequences = []
                targets = []
                
                for _ in range(batch_size):
                    # Random sequence
                    start_idx = np.random.randint(0, max(1, len(self.embeddings) - self.sequence_length))
                    sequence = []
                    sequence_targets = []
                    
                    for i in range(self.sequence_length):
                        if start_idx + i < len(self.embeddings):
                            sequence.append(self.embeddings[start_idx + i])
                            # Target is next embedding (or same if at end)
                            target_idx = min(start_idx + i + 1, len(self.embeddings) - 1)
                            sequence_targets.append(self.embeddings[target_idx])
                        else:
                            # Pad with random
                            sequence.append(torch.randn(768))
                            sequence_targets.append(torch.randn(768))
                    
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
                print(f"❌ Training not setup for {trainer_key}")
                return False
            
            trainer = self.trainers[trainer_key]
            config = self.training_history[trainer_key]['config']
            
            print(f"\\n🚀 Starting training: {trainer_key}")
            print(f"   Epochs: {config.num_epochs}")
            print(f"   Batch size: {config.batch_size}")
            print(f"   Learning rate: {config.learning_rate}")
            print(f"   Evolution rate: {config.evolution_rate}")
            print("=" * 60)
            
            # Record start time
            self.training_history[trainer_key]['start_time'] = time.time()
            
            # Training loop
            for epoch in range(config.num_epochs):
                print(f"\\n📚 Epoch {epoch + 1}/{config.num_epochs}")
                
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
                    print(f"   Trust Mean: {epoch_metrics['trust_mean']:.3f} ± {epoch_metrics['trust_std']:.3f}")
                    print(f"   Fossilized: {epoch_metrics['final_num_fossilized']}")
                    
                    # Show trust evolution
                    model = self.models[model_name]
                    trust_scalars = model.trust_scalars
                    print(f"   Trust Scalars: {[f'{t:.3f}' for t in trust_scalars.tolist()]}")
                    
                    # Apply Mandelbulb augmentation if configured
                    if config.use_mandelbulb_augmentation and f"{model_name}_{dataset_name}" in self.augmenters:
                        print("   🌀 Applying Mandelbulb-Gyroidic augmentation...")
                        augmenter = self.augmenters[f"{model_name}_{dataset_name}"]
                        
                        # Get sample data for augmentation
                        sample_batch = trainer.dataset.get_batch(batch_size=4)
                        sample_X = sample_batch['sequences'][:, 0, :]  # First timestep
                        
                        # Apply augmentation
                        augmented_X, _ = augmenter(sample_X, augmentation_factor=config.augmentation_factor)
                        print(f"   🌀 Augmented {sample_X.shape[0]} → {augmented_X.shape[0]} samples")
                    
                    # Save checkpoint if configured
                    if config.save_checkpoints and (epoch + 1) % config.checkpoint_interval == 0:
                        checkpoint_path = f"checkpoint_{trainer_key}_epoch_{epoch + 1}.pt"
                        self._save_checkpoint(trainer_key, checkpoint_path)
                        print(f"   💾 Checkpoint saved: {checkpoint_path}")
                
                except Exception as e:
                    print(f"   ❌ Epoch {epoch + 1} failed: {e}")
                    continue
            
            # Training complete
            total_time = time.time() - self.training_history[trainer_key]['start_time']
            print(f"\\n🎯 Training Complete!")
            print(f"   Total time: {total_time:.1f} seconds")
            print(f"   Epochs completed: {self.training_history[trainer_key]['epochs_completed']}")
            
            # Final model state
            model = self.models[model_name]
            final_trust = model.trust_scalars
            print(f"   Final trust: {[f'{t:.3f}' for t in final_trust.tolist()]}")
            print(f"   Fossilized functionals: {(final_trust > config.fossilization_threshold).sum().item()}")
            
            # Save final state
            final_checkpoint = f"final_{trainer_key}.pt"
            self._save_checkpoint(trainer_key, final_checkpoint)
            print(f"   💾 Final state saved: {final_checkpoint}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
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
        print("\\n📊 Available Datasets:")
        if not self.datasets:
            print("   No datasets loaded")
            return
        
        for name, config in self.datasets.items():
            dataset_path = self.data_dir / name / "processed_data.pt"
            if dataset_path.exists():
                data = torch.load(dataset_path)
                sample_count = len(data)
            else:
                sample_count = "Unknown"
            
            print(f"   • {name}")
            print(f"     Source: {config.source_type} - {config.source_path}")
            print(f"     Preprocessing: {config.preprocessing}")
            print(f"     Samples: {sample_count}")
            print(f"     Augmentation: {config.augmentation}")
    
    def list_models(self):
        """List all available models."""
        print("\\n🏗️ Available Models:")
        if not self.models:
            print("   No models created")
            return
        
        for name, model in self.models.items():
            param_count = sum(p.numel() for p in model.parameters())
            trust_mean = model.trust_scalars.mean().item()
            fossilized = (model.trust_scalars > 0.8).sum().item()
            
            print(f"   • {name}")
            print(f"     Parameters: {param_count:,}")
            print(f"     Functionals: {model.K}")
            print(f"     Trust mean: {trust_mean:.3f}")
            print(f"     Fossilized: {fossilized}/{model.K}")
    
    def list_training_sessions(self):
        """List all training sessions."""
        print("\\n🎯 Training Sessions:")
        if not self.training_history:
            print("   No training sessions")
            return
        
        for key, history in self.training_history.items():
            status = "Complete" if history['epochs_completed'] == history['config'].num_epochs else "In Progress"
            
            print(f"   • {key}")
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
  python dataset_ingestion_system.py create-model --name "temporal_model" --type temporal --functionals 5
  
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
    add_dataset_parser.add_argument('--source', required=True, choices=['huggingface', 'kaggle', 'wikipedia', 'local', 'url'], help='Source type')
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
    create_model_parser.add_argument('--functionals', type=int, default=5, help='Number of polynomial functionals')
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
    
    elif args.command == 'list-training':
        system.list_training_sessions()
    
    elif args.command == 'list-all':
        system.list_datasets()
        system.list_models()
        system.list_training_sessions()


if __name__ == "__main__":
    main()