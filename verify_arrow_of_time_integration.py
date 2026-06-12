#!/usr/bin/env python3
import torch
import sys
import os

# Add src to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.gyroid_reasoner import GyroidicFluxReasoner
from src.training.temporal_association_trainer import TemporalAssociationTrainer, TemporalAssociationDataset

def main():
    print("[TEST] Verifying Arrow of Time Integration in Trainer")
    print("=" * 60)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        
    print(f"Using device: {device}")
    
    # Instantiate Reasoner Model with use_admm=False for fast CPU verification
    print("Initializing GyroidicFluxReasoner (use_admm=False)...")
    model = GyroidicFluxReasoner(use_admm=False).to(device)
    
    # Instantiate Dataset
    print("Initializing TemporalAssociationDataset...")
    dataset = TemporalAssociationDataset(
        sequence_length=8,
        association_window=2,
        num_concepts=100,
        device=device
    )
    
    # Instantiate Trainer
    print("Initializing TemporalAssociationTrainer...")
    trainer = TemporalAssociationTrainer(
        model=model,
        dataset=dataset,
        learning_rate=0.01,
        fossilization_threshold=0.8,
        device=device
    )
    
    # Generate a sample temporal sequence batch
    print("Generating batch sequence...")
    batch_data = dataset.get_temporal_sequence(batch_size=2)
    
    print(f"Batch sequence shape: {batch_data['sequences'].shape}")
    print(f"Batch associations shape: {batch_data['associations'].shape}")
    print(f"Batch contexts: {batch_data['contexts']}")
    
    # Perform a single training step
    print("Executing train_step...")
    metrics = trainer.train_step(batch_data)
    
    print("\n[METRICS] Step Metrics:")
    for key, val in metrics.items():
        if key == 'repair_diagnostics':
            print(f"  {key}:")
            for sub_key, sub_val in val.items():
                print(f"    {sub_key}: {sub_val}")
        else:
            print(f"  {key}: {val}")
            
    # Check that the expected keys are present and have valid values
    assert 'arrow_of_time_asymmetry' in metrics, "Missing 'arrow_of_time_asymmetry' in step metrics"
    assert isinstance(metrics['arrow_of_time_asymmetry'], float), "arrow_of_time_asymmetry should be a float"
    
    print("\n[OK] Arrow of Time integration verified successfully!")
    
if __name__ == "__main__":
    main()
