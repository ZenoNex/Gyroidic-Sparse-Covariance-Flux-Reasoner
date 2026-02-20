"""
Test CALM + KAGH + HarmonicWaveDecomposition integration in DiegeticPhysicsEngine.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui.diegetic_backend import DiegeticPhysicsEngine

print("Initializing DiegeticPhysicsEngine with CALM + KAGH...")
engine = DiegeticPhysicsEngine()
print("SUCCESS: Engine initialized")

print("\nTesting process_input...")
metrics = engine.process_input("Hello world, this is a test of the integrated systems.")

print(f"\nResponse: {metrics['response'][:100]}...")
print(f"Response length: {metrics['output_length']} / {metrics['max_output_length']} max")
print(f"CALM abort score: {metrics['calm_abort_score']:.4f}")
print(f"CALM step factor: {metrics['calm_step_factor']:.4f}")
print(f"Gyroid entropy: {metrics['spectral_entropy']:.4f}")
print(f"Mischief: {metrics['mischief']:.4f}")
print(f"Iteration: {metrics['iteration']}")

print("\n=== CALM + KAGH Integration Test PASSED ===")
