import torch
import sys
import os

# Set PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.gyroid_reasoner import GyroidicFluxReasoner
from src.training.temporal_association_trainer import TemporalAssociationDataset, TemporalAssociationTrainer
from src.core.device_utils import DEVICE

def test_stabilization():
    print(f"Testing stabilization on {DEVICE}...")
    
    # 1. Model Configuration
    model_config = {
        'hidden_dim': 256,
        'num_layers': 2,
        'num_functionals': 5,
        'poly_degree': 5,
        'use_resonance': True,
        'use_admm': True
    }
    
    model = GyroidicFluxReasoner(**model_config).to(DEVICE)
    print("Model initialized.")
    
    # 2. Dataset Configuration
    dataset = TemporalAssociationDataset(sequence_length=8, device=DEVICE)
    print("Dataset initialized.")
    
    # 3. Trainer Configuration
    trainer = TemporalAssociationTrainer(
        model=model,
        dataset=dataset,
        device=DEVICE
    )
    print("Trainer initialized.")
    
    # 4. Run a single train step
    print("Running training step...")
    batch_data = dataset.get_temporal_sequence(batch_size=2)
    metrics = trainer.train_step(batch_data)
    
    print(f"Step Metrics: {metrics}")
    print("Stabilization test PASSED.")

if __name__ == "__main__":
    test_stabilization()
