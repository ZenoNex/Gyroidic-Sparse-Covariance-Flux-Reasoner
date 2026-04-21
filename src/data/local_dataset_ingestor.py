"""
Local Dataset Ingestor

Specialized nutrient extractor for massive local image datasets 
(CIFAR-10, MNIST, etc.). Implements drip-feeding to sustain 
manifold growth without saturation.

Author: William Matthew Bryant
Created: April 2026
"""

import os
import csv
import torch
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional
from src.data.conversational_types import Conversation, ConversationTurn, _stable_id
from src.data.canonical_projection import CanonicalProjector

class LocalDatasetIngestor:
    """
    Ingestor for local assets found in DeepLearningStudio paths.
    """
    def __init__(self, datasets_root: str, device: str = 'cpu'):
        self.root = Path(datasets_root)
        self.device = device
        self.projector = CanonicalProjector(device=device)
        
        if not self.root.exists():
            print(f" Local Dataset Root not found: {self.root}")

    def cifar10_generator(self, limit: int = 2000) -> Generator[Conversation, None, None]:
        """
        Drip-feeds CIFAR-10 images as conversational dyads (Label -> Image).
        """
        cifar_path = self.root / 'cifar-10'
        image_dir = cifar_path / 'images'
        csv_path = cifar_path / 'train.csv'
        
        if not csv_path.exists():
            print(f" CIFAR-10 train.csv not found at {csv_path}")
            return

        print(f" Initializing CIFAR-10 Drip-Feed from {image_dir}...")
        
        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if count >= limit:
                    break
                    
                img_path = image_dir / row['filename']
                if not img_path.exists():
                    continue
                
                try:
                    # Project image path to manifold state
                    # We use project_image_path_to_state to get residue and entropy
                    proj = self.projector.project_image_path_to_state(str(img_path))
                    
                    # Create a dyad representation: Turn 1 (Label), Turn 2 (Image Residue)
                    turns = [
                        ConversationTurn(
                            speaker_id="label_oracle",
                            text=f"Class: {row['label']}",
                            metadata={'label': row['label']}
                        ),
                        ConversationTurn(
                            speaker_id="visual_sensor",
                            text="[Image Projection Embedded]",
                            embedding=proj['state'],
                            metadata={'entropy': proj['entropy'], 'source': str(img_path)}
                        )
                    ]
                    
                    yield Conversation(
                        conversation_id=_stable_id("cifar10", row['filename']),
                        turns=turns,
                        context={'source': 'local_deeplearning_studio', 'dataset': 'cifar-10'},
                        source='local_dataset'
                    )
                    count += 1
                except Exception as e:
                    print(f" Skip CIFAR frame {row['filename']}: {e}")
                    continue

    def mnist_generator(self, limit: int = 2000) -> Generator[Conversation, None, None]:
        """
        Placeholder for MNIST ingestion logic.
        """
        # Logic similar to CIFAR if images are stored individually
        return
        yield from [] 
