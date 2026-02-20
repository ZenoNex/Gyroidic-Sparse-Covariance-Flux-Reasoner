#!/usr/bin/env python3
"""
Gyroidic Dataset Command Interface

A comprehensive command-line interface for dataset ingestion, training, and management.
Provides easy access to multiple dataset sources and training configurations.

Usage Examples:
    # Quick start with HuggingFace dataset
    python dataset_command_interface.py quick-start --dataset imdb --samples 1000
    
    # Add Wikipedia knowledge
    python dataset_command_interface.py add-wikipedia --topics "quantum_mechanics,relativity" --samples 500
    
    # Train on local files
    python dataset_command_interface.py train-local --path "./documents/" --epochs 10
    
    # Full pipeline with Mandelbulb augmentation
    python dataset_command_interface.py full-pipeline --source huggingface --dataset "squad" --augment

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
import subprocess
import time
from urllib.parse import urlparse

# Add src to path and set up package structure
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also add examples directory
examples_dir = os.path.join(current_dir, 'examples')
if examples_dir not in sys.path:
    sys.path.insert(0, examples_dir)

# Import our systems
from dataset_ingestion_system import DatasetIngestionSystem, DatasetConfig, TrainingConfig

class DatasetCommandInterface:
    """
    Command-line interface for the Gyroidic Dataset Ingestion System.
    
    Provides easy access to:
    - Popular dataset sources (HuggingFace, Kaggle, Wikipedia)
    - Pre-configured training pipelines
    - Storage-optimized workflows for 100GB constraint
    - Anti-lobotomy compliant training
    """
    
    def __init__(self):
        self.system = DatasetIngestionSystem()
        self.popular_datasets = {
            # Text datasets
            'imdb': {'source': 'huggingface', 'path': 'imdb', 'type': 'text'},
            'squad': {'source': 'huggingface', 'path': 'squad', 'type': 'text'},
            'wikitext': {'source': 'huggingface', 'path': 'wikitext-2-raw-v1', 'type': 'text'},
            'openwebtext': {'source': 'huggingface', 'path': 'openwebtext', 'type': 'text'},
            
            # Code datasets
            'codeparrot': {'source': 'huggingface', 'path': 'codeparrot/codeparrot-clean', 'type': 'text'},
            'github_code': {'source': 'huggingface', 'path': 'github-code', 'type': 'text'},
            
            # Scientific datasets
            'arxiv': {'source': 'huggingface', 'path': 'arxiv_dataset', 'type': 'text'},
            'pubmed': {'source': 'huggingface', 'path': 'pubmed_qa', 'type': 'text'},
            
            # Multimodal datasets
            'coco': {'source': 'huggingface', 'path': 'coco', 'type': 'multimodal'},
            'flickr30k': {'source': 'huggingface', 'path': 'flickr30k', 'type': 'multimodal'},
        }
        
        self.wikipedia_topics = {
            'physics': ['Quantum_mechanics', 'General_relativity', 'Thermodynamics', 'Statistical_mechanics', 'Particle_physics'],
            'mathematics': ['Linear_algebra', 'Calculus', 'Topology', 'Abstract_algebra', 'Number_theory'],
            'computer_science': ['Machine_learning', 'Algorithms', 'Data_structures', 'Computer_graphics', 'Cryptography'],
            'philosophy': ['Philosophy_of_mind', 'Epistemology', 'Logic', 'Ethics', 'Metaphysics'],
            'biology': ['Molecular_biology', 'Evolution', 'Genetics', 'Neuroscience', 'Ecology'],
        }
    
    def quick_start(self, args):
        """Quick start with a popular dataset."""
        print(f"🚀 Quick Start: {args.dataset}")
        print("=" * 50)
        
        if args.dataset not in self.popular_datasets:
            print(f"❌ Unknown dataset: {args.dataset}")
            print(f"Available datasets: {list(self.popular_datasets.keys())}")
            return False
        
        dataset_info = self.popular_datasets[args.dataset]
        
        # Create dataset config
        config = DatasetConfig(
            name=args.dataset,
            source_type=dataset_info['source'],
            source_path=dataset_info['path'],
            preprocessing=dataset_info['type'],
            max_samples=args.samples,
            augmentation=True,
            mandelbulb_augmentation=args.augment
        )
        
        # Add dataset
        print(f"📥 Adding dataset: {args.dataset}")
        success = self.system.add_dataset(config)
        
        if success:
            # Create model
            print(f"🧠 Creating temporal model...")
            model_config = {
                'name': f"{args.dataset}_model",
                'type': 'temporal',
                'functionals': 5,
                'hidden_dim': 256
            }
            self.system.create_model(model_config)
            
            # Setup training
            print(f"⚙️ Setting up training...")
            training_config = TrainingConfig(
                num_epochs=args.epochs,
                batch_size=4,
                use_mandelbulb_augmentation=args.augment,
                augmentation_factor=2
            )
            self.system.setup_training(f"{args.dataset}_model", args.dataset, training_config)
            
            # Start training
            print(f"🏋️ Starting training...")
            self.system.train(f"{args.dataset}_model", args.dataset)
            
            print(f"✅ Quick start completed for {args.dataset}")
            return True
        else:
            print(f"❌ Failed to add dataset {args.dataset}")
            return False
    
    def add_wikipedia(self, args):
        """Add Wikipedia articles on specific topics."""
        print(f"📚 Adding Wikipedia Knowledge")
        print("=" * 40)
        
        # Parse topics
        if args.topics in self.wikipedia_topics:
            # Use predefined topic collection
            topics = self.wikipedia_topics[args.topics]
            dataset_name = f"wikipedia_{args.topics}"
        else:
            # Use custom topics
            topics = args.topics.split(',')
            dataset_name = f"wikipedia_custom"
        
        print(f"Topics: {topics}")
        
        # Create dataset config
        config = DatasetConfig(
            name=dataset_name,
            source_type='wikipedia',
            source_path=','.join(topics),
            preprocessing='text',
            max_samples=args.samples,
            augmentation=True,
            temporal_associations=True
        )
        
        # Add dataset
        success = self.system.add_dataset(config)
        
        if success:
            print(f"✅ Added Wikipedia dataset: {dataset_name}")
            print(f"   Topics: {len(topics)}")
            print(f"   Max samples: {args.samples}")
            
            if args.train:
                # Auto-create model and train
                model_name = f"{dataset_name}_model"
                print(f"🧠 Creating model: {model_name}")
                
                model_config = {
                    'name': model_name,
                    'type': 'temporal',
                    'functionals': 7,  # More functionals for knowledge
                    'hidden_dim': 512
                }
                self.system.create_model(model_config)
                
                # Setup and run training
                training_config = TrainingConfig(
                    num_epochs=15,
                    batch_size=2,  # Smaller batch for memory efficiency
                    learning_rate=5e-5,
                    use_mandelbulb_augmentation=True
                )
                
                self.system.setup_training(model_name, dataset_name, training_config)
                self.system.train(model_name, dataset_name)
            
            return True
        else:
            print(f"❌ Failed to add Wikipedia dataset")
            return False
    
    def train_local(self, args):
        """Train on local files."""
        print(f"📁 Training on Local Files")
        print("=" * 40)
        
        local_path = Path(args.path)
        if not local_path.exists():
            print(f"❌ Path not found: {local_path}")
            return False
        
        # Create dataset config
        dataset_name = f"local_{local_path.name}"
        config = DatasetConfig(
            name=dataset_name,
            source_type='local',
            source_path=str(local_path),
            preprocessing='text',
            max_samples=args.samples,
            augmentation=True,
            mandelbulb_augmentation=args.augment
        )
        
        # Add dataset
        success = self.system.add_dataset(config)
        
        if success:
            # Create model
            model_name = f"{dataset_name}_model"
            model_config = {
                'name': model_name,
                'type': 'temporal',
                'functionals': 6,
                'hidden_dim': 384
            }
            self.system.create_model(model_config)
            
            # Setup training
            training_config = TrainingConfig(
                num_epochs=args.epochs,
                batch_size=4,
                use_mandelbulb_augmentation=args.augment,
                augmentation_factor=2
            )
            
            self.system.setup_training(model_name, dataset_name, training_config)
            
            # Train
            print(f"🏋️ Training on {local_path}")
            self.system.train(model_name, dataset_name)
            
            print(f"✅ Local training completed")
            return True
        else:
            print(f"❌ Failed to process local files")
            return False
    
    def full_pipeline(self, args):
        """Run full pipeline with all features."""
        print(f"🌟 Full Gyroidic Pipeline")
        print("=" * 50)
        
        # Determine dataset source
        if args.source == 'huggingface' and args.dataset in self.popular_datasets:
            dataset_info = self.popular_datasets[args.dataset]
            config = DatasetConfig(
                name=args.dataset,
                source_type='huggingface',
                source_path=dataset_info['path'],
                preprocessing=dataset_info['type'],
                max_samples=args.samples,
                augmentation=True,
                mandelbulb_augmentation=args.augment,
                temporal_associations=True
            )
        elif args.source == 'wikipedia':
            topics = args.dataset.split(',')
            config = DatasetConfig(
                name=f"wikipedia_{args.dataset.replace(',', '_')}",
                source_type='wikipedia',
                source_path=args.dataset,
                preprocessing='text',
                max_samples=args.samples,
                augmentation=True,
                mandelbulb_augmentation=args.augment,
                temporal_associations=True
            )
        elif args.source == 'local':
            config = DatasetConfig(
                name=f"local_{Path(args.dataset).name}",
                source_type='local',
                source_path=args.dataset,
                preprocessing='text',
                max_samples=args.samples,
                augmentation=True,
                mandelbulb_augmentation=args.augment,
                temporal_associations=True
            )
        else:
            print(f"❌ Invalid source/dataset combination")
            return False
        
        # Execute full pipeline
        print(f"📥 Step 1: Adding dataset...")
        success = self.system.add_dataset(config)
        
        if not success:
            print(f"❌ Failed to add dataset")
            return False
        
        print(f"🧠 Step 2: Creating advanced model...")
        model_name = f"{config.name}_advanced"
        model_config = {
            'name': model_name,
            'type': 'temporal',
            'functionals': 8,  # Maximum functionals
            'poly_degree': 5,
            'hidden_dim': 768,  # Large model
            'num_heads': 12
        }
        self.system.create_model(model_config)
        
        print(f"⚙️ Step 3: Advanced training setup...")
        training_config = TrainingConfig(
            num_epochs=args.epochs,
            batch_size=2,  # Small batch for large model
            learning_rate=1e-5,  # Conservative learning rate
            evolution_rate=0.01,
            use_mandelbulb_augmentation=args.augment,
            augmentation_factor=3,  # Aggressive augmentation
            save_checkpoints=True,
            checkpoint_interval=2
        )
        
        self.system.setup_training(model_name, config.name, training_config)
        
        print(f"🏋️ Step 4: Full training with all features...")
        self.system.train(model_name, config.name)
        
        print(f"🎯 Step 5: Evaluation and analysis...")
        # TODO: Add evaluation metrics
        
        print(f"✅ Full pipeline completed!")
        print(f"   Model: {model_name}")
        print(f"   Dataset: {config.name}")
        print(f"   Augmentation: {'✅' if args.augment else '❌'}")
        print(f"   Temporal Associations: ✅")
        
        return True
    
    def list_datasets(self, args):
        """List available datasets and sources."""
        print(f"📊 Available Datasets and Sources")
        print("=" * 50)
        
        print(f"\n🤗 Popular HuggingFace Datasets:")
        for name, info in self.popular_datasets.items():
            print(f"   {name:15} - {info['type']:10} - {info['path']}")
        
        print(f"\n📚 Wikipedia Topic Collections:")
        for topic, articles in self.wikipedia_topics.items():
            print(f"   {topic:15} - {len(articles)} articles")
        
        print(f"\n💾 Storage Estimates (per 1000 samples):")
        print(f"   Text dataset:     ~50-100 MB")
        print(f"   + Fingerprints:   ~5 MB")
        print(f"   + Embeddings:     ~15 MB")
        print(f"   + Augmentation:   ~20-40 MB")
        print(f"   Total per 1k:     ~90-160 MB")
        
        print(f"\n🎯 Recommended for 100GB constraint:")
        print(f"   Small datasets:   1,000-5,000 samples")
        print(f"   Medium datasets:  500-2,000 samples")
        print(f"   Large datasets:   100-1,000 samples")
        
        return True
    
    def status(self, args):
        """Show system status and storage usage."""
        print(f"📊 Gyroidic System Status")
        print("=" * 40)
        
        # Check storage usage
        data_dir = Path("datasets")
        if data_dir.exists():
            total_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
            print(f"💾 Storage Usage:")
            print(f"   Data directory: {total_size / 1024 / 1024:.1f} MB")
            print(f"   Available: ~{100 * 1024 - total_size / 1024 / 1024:.1f} MB (assuming 100GB limit)")
        
        # Check for existing datasets
        if hasattr(self.system, 'datasets') and self.system.datasets:
            print(f"\n📚 Loaded Datasets:")
            for name, dataset in self.system.datasets.items():
                print(f"   {name}: {len(dataset)} samples")
        else:
            print(f"\n📚 No datasets loaded")
        
        # Check for models
        if hasattr(self.system, 'models') and self.system.models:
            print(f"\n🧠 Available Models:")
            for name, model in self.system.models.items():
                param_count = sum(p.numel() for p in model.parameters())
                print(f"   {name}: {param_count:,} parameters")
        else:
            print(f"\n🧠 No models created")
        
        # System health
        print(f"\n🔧 System Health:")
        print(f"   Device: {self.system.device}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        return True

def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Gyroidic Dataset Command Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with IMDB dataset
  python dataset_command_interface.py quick-start --dataset imdb --samples 1000 --epochs 5

  # Add Wikipedia physics knowledge
  python dataset_command_interface.py add-wikipedia --topics physics --samples 500 --train

  # Train on local documents
  python dataset_command_interface.py train-local --path ./documents/ --epochs 10 --augment

  # Full pipeline with HuggingFace dataset
  python dataset_command_interface.py full-pipeline --source huggingface --dataset squad --augment --epochs 15

  # List available datasets
  python dataset_command_interface.py list-datasets

  # Check system status
  python dataset_command_interface.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Quick start command
    quick_parser = subparsers.add_parser('quick-start', help='Quick start with popular dataset')
    quick_parser.add_argument('--dataset', required=True, help='Dataset name (imdb, squad, wikitext, etc.)')
    quick_parser.add_argument('--samples', type=int, default=1000, help='Max samples to use')
    quick_parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    quick_parser.add_argument('--augment', action='store_true', help='Use Mandelbulb augmentation')
    
    # Wikipedia command
    wiki_parser = subparsers.add_parser('add-wikipedia', help='Add Wikipedia knowledge')
    wiki_parser.add_argument('--topics', required=True, help='Topics (physics, math, cs) or custom list')
    wiki_parser.add_argument('--samples', type=int, default=500, help='Max samples per topic')
    wiki_parser.add_argument('--train', action='store_true', help='Auto-train after adding')
    
    # Local training command
    local_parser = subparsers.add_parser('train-local', help='Train on local files')
    local_parser.add_argument('--path', required=True, help='Path to local files/directory')
    local_parser.add_argument('--samples', type=int, default=None, help='Max samples to use')
    local_parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    local_parser.add_argument('--augment', action='store_true', help='Use Mandelbulb augmentation')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full-pipeline', help='Run full pipeline with all features')
    full_parser.add_argument('--source', required=True, choices=['huggingface', 'wikipedia', 'local'], help='Data source')
    full_parser.add_argument('--dataset', required=True, help='Dataset name/path/topics')
    full_parser.add_argument('--samples', type=int, default=1000, help='Max samples to use')
    full_parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    full_parser.add_argument('--augment', action='store_true', help='Use Mandelbulb augmentation')
    
    # List datasets command
    subparsers.add_parser('list-datasets', help='List available datasets and sources')
    
    # Status command
    subparsers.add_parser('status', help='Show system status and storage usage')
    
    return parser

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize interface
    interface = DatasetCommandInterface()
    
    # Execute command
    try:
        if args.command == 'quick-start':
            success = interface.quick_start(args)
        elif args.command == 'add-wikipedia':
            success = interface.add_wikipedia(args)
        elif args.command == 'train-local':
            success = interface.train_local(args)
        elif args.command == 'full-pipeline':
            success = interface.full_pipeline(args)
        elif args.command == 'list-datasets':
            success = interface.list_datasets(args)
        elif args.command == 'status':
            success = interface.status(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            success = False
        
        if success:
            print(f"\n✅ Command completed successfully!")
        else:
            print(f"\n❌ Command failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()