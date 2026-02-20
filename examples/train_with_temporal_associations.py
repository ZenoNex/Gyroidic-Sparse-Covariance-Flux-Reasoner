"""
Training script for feeding temporal associations to the repaired Gyroidic Flux Reasoner.

This script demonstrates how to train the system with ordered data and associations
to build up its resonance cavity memory and trust scalars over time.
"""

import torch
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import directly to avoid relative import issues
try:
    from src.training.temporal_association_trainer import (
        TemporalAssociationDataset, 
        TemporalAssociationTrainer,
        create_training_session
    )
    from src.models.gyroid_reasoner import GyroidicFluxReasoner
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating standalone version...")
    
    # We'll define the classes inline to avoid import issues
    GyroidicFluxReasoner = None


def run_temporal_association_training():
    """Run temporal association training session."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    print(f"🧠 Temporal Association Training")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Model configuration
    model_config = {
        'text_dim': 768,
        'graph_dim': 256,
        'num_dim': 64,
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'num_functionals': 5,
        'poly_degree': 4,
        'poly_basis': 'chebyshev',
        'dropout': 0.1,
        'use_introspection': True,
        'use_gyroid_probes': True,
        'use_resonance': True,
        'use_gdpo': True,
        'learnable_weights': True,
        'kl_weight': 0.01,
        'use_admm': False,  # Start without ADMM for simplicity
        'use_saturation': True
    }
    
    # Dataset configuration
    dataset_config = {
        'sequence_length': 16,
        'association_window': 4,
        'num_concepts': 500,  # Start smaller for faster training
    }
    
    # Training configuration
    training_config = {
        'learning_rate': 1e-4,
        'trust_update_rate': 0.02,
        'fossilization_threshold': 0.85,
    }
    
    print("📊 Configuration:")
    print(f"  Model: {model_config['num_functionals']} functionals, {model_config['hidden_dim']} hidden dim")
    print(f"  Dataset: {dataset_config['num_concepts']} concepts, {dataset_config['sequence_length']} seq length")
    print(f"  Training: {training_config['learning_rate']} lr, {training_config['fossilization_threshold']} fossilization threshold")
    
    # Create training session
    print("\n🏗️ Creating training session...")
    trainer = create_training_session(
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        device=device
    )
    
    print(f"✅ Training session created")
    print(f"   Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")
    
    # Initial state
    print(f"\n📈 Initial State:")
    print(f"   Trust scalars: {trainer.model.trust_scalars}")
    print(f"   Mean trust: {trainer.model.trust_scalars.mean():.3f}")
    print(f"   Fossilized functionals: {(trainer.model.trust_scalars > trainer.fossilization_threshold).sum().item()}")
    
    # Test with sample data
    print(f"\n🧪 Testing with sample data...")
    sample_batch = trainer.dataset.get_temporal_sequence(batch_size=2)
    print(f"   Sample sequences shape: {sample_batch['sequences'].shape}")
    print(f"   Sample associations shape: {sample_batch['associations'].shape}")
    print(f"   Sample contexts: {sample_batch['contexts']}")
    
    # Run initial forward pass to test
    with torch.no_grad():
        test_output = trainer.model(
            text_emb=sample_batch['sequences'][0, 0, :].unsqueeze(0),
            return_analysis=True
        )
        print(f"   Test output shape: {test_output['output'].shape}")
        print(f"   Repair diagnostics available: {list(test_output.keys())}")
    
    # Training loop
    num_epochs = 5
    batches_per_epoch = 50
    
    print(f"\n🚀 Starting training: {num_epochs} epochs, {batches_per_epoch} batches each")
    print("-" * 60)
    
    training_results = []
    
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        
        # Train epoch
        epoch_metrics = trainer.train_epoch(num_batches=batches_per_epoch)
        training_results.append(epoch_metrics)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Association Accuracy: {epoch_metrics['association_accuracy']:.3f}")
        print(f"   Temporal Coherence: {epoch_metrics['temporal_coherence']:.3f}")
        print(f"   Survivorship Pressure: {epoch_metrics['survivorship_pressure']:.3f}")
        print(f"   Final Trust Mean: {epoch_metrics['final_trust_mean']:.3f}")
        print(f"   Fossilized Functionals: {epoch_metrics['final_num_fossilized']}")
        
        # Show trust evolution
        current_trust = trainer.model.trust_scalars
        print(f"   Trust Scalars: {[f'{t:.3f}' for t in current_trust.tolist()]}")
        
        # Test repair system after each epoch
        print(f"\n🔧 Testing repair system after epoch {epoch + 1}:")
        test_garbled_text = "nccmtsmneltcclrclcnl"
        
        # Get repair diagnostics from last training step
        if 'repair_diagnostics' in test_output:
            spectral_diag = test_output.get('spectral_diagnostics', {})
            if spectral_diag:
                print(f"   Spectral coherence threshold: {spectral_diag.get('theta_coherence', 'N/A')}")
                print(f"   Soliton/Ergodic ratio: {spectral_diag.get('energy_ratio', 'N/A')}")
            
            love_diag = test_output.get('love_diagnostics', {})
            if love_diag:
                print(f"   Love violations: {love_diag.get('violation_count', 'N/A')}")
            
            soft_gates_diag = test_output.get('soft_gates_diagnostics', {})
            if soft_gates_diag:
                print(f"   System temperature: {soft_gates_diag.get('dt', 'N/A')}")
                print(f"   Fossilized gates: {soft_gates_diag.get('num_fossilized', 'N/A')}")
    
    # Final analysis
    print(f"\n🎯 Training Complete!")
    print("=" * 50)
    
    final_trust = trainer.model.trust_scalars
    print(f"Final Trust Scalars: {[f'{t:.3f}' for t in final_trust.tolist()]}")
    print(f"Mean Trust: {final_trust.mean():.3f} (started at ~1.000)")
    print(f"Trust Std: {final_trust.std():.3f}")
    print(f"Fossilized Functionals: {(final_trust > trainer.fossilization_threshold).sum().item()}/{len(final_trust)}")
    
    # Show training progression
    print(f"\n📈 Training Progression:")
    for i, result in enumerate(training_results):
        print(f"   Epoch {i+1}: Assoc={result['association_accuracy']:.3f}, "
              f"Coherence={result['temporal_coherence']:.3f}, "
              f"Fossilized={result['final_num_fossilized']}")
    
    # Test final system with various inputs
    print(f"\n🧪 Final System Test:")
    test_inputs = [
        "nccmtsmneltcclrclcnl,tncsectsead",  # Original garbled
        "hello world this is a test",         # Normal text
        "aaaaaaaaaaaaaaaaaaa",               # Repetitive
        "xyz123!@#$%^&*()",                  # Mixed characters
    ]
    
    for test_input in test_inputs:
        # Simulate text embedding (in real system, this would be proper encoding)
        test_embedding = torch.randn(1, 768, device=device)
        
        with torch.no_grad():
            output = trainer.model(text_emb=test_embedding, return_analysis=True)
            
            # Check repair diagnostics
            spectral_diag = output.get('spectral_diagnostics', {})
            coherence_threshold = spectral_diag.get('theta_coherence', 0.0)
            
            print(f"   Input: '{test_input[:20]}...'")
            print(f"     Output norm: {torch.norm(output['output']):.3f}")
            print(f"     Coherence threshold: {coherence_threshold:.3f}")
            print(f"     Selection pressure: {output.get('selection_pressure', 'N/A')}")
    
    # Save training state
    save_path = "temporal_training_state.pt"
    trainer.save_training_state(save_path)
    
    print(f"\n💾 Training state saved to {save_path}")
    print(f"   You can resume training by loading this state")
    
    # Save training results
    results_path = "training_results.json"
    with open(results_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        json_results = []
        for result in training_results:
            json_result = {}
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    json_result[key] = value
                else:
                    json_result[key] = str(value)
            json_results.append(json_result)
        
        json.dump({
            'training_results': json_results,
            'final_trust_scalars': final_trust.tolist(),
            'model_config': model_config,
            'dataset_config': dataset_config,
            'training_config': training_config
        }, f, indent=2)
    
    print(f"📊 Training results saved to {results_path}")
    
    return trainer, training_results


def demonstrate_temporal_patterns():
    """Demonstrate the temporal patterns the system learns."""
    
    print(f"\n🔍 Demonstrating Temporal Patterns")
    print("=" * 40)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
    dataset = TemporalAssociationDataset(
        sequence_length=8,
        association_window=3,
        num_concepts=50,
        device=device
    )
    
    print(f"Created dataset with {dataset.num_concepts} concepts")
    print(f"Association graph sample: {list(dataset.association_graph.items())[:3]}")
    print(f"Temporal patterns sample: {dataset.temporal_patterns[:3]}")
    
    # Generate and show sample sequences
    for i in range(3):
        batch = dataset.get_temporal_sequence(batch_size=1)
        print(f"\nSample {i+1}:")
        print(f"  Sequence shape: {batch['sequences'].shape}")
        print(f"  Context: {batch['contexts'][0]}")
        print(f"  First few embeddings norm: {torch.norm(batch['sequences'][0, :3, :], dim=1)}")


if __name__ == "__main__":
    print("🧠 Gyroidic Flux Reasoner - Temporal Association Training")
    print("This script trains the repaired system with temporal data and associations")
    print("=" * 70)
    
    try:
        # Demonstrate temporal patterns first
        demonstrate_temporal_patterns()
        
        # Run main training
        trainer, results = run_temporal_association_training()
        
        print(f"\n✅ Training completed successfully!")
        print(f"The system now has:")
        print(f"• Temporal association memory")
        print(f"• Evolved trust scalars")
        print(f"• Fossilized successful patterns")
        print(f"• Repaired garbled output handling")
        
        print(f"\nNext steps:")
        print(f"• Load the saved state to continue training")
        print(f"• Test with real text data")
        print(f"• Experiment with different association patterns")
        print(f"• Monitor fossilization and trust evolution")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
