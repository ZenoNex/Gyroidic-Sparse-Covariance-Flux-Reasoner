import torch
import os
import json
from src.core.knowledge_dyad_fossilizer import DyadFossilizer, KnowledgeDyad
from src.core.quantum_tda import QuantumBettiApproximator

def execute_v2():
    fossil_path = "data/encodings/fossil_1777063956306_healed.pt"
    if not os.path.exists(fossil_path):
        print(f"Error: Fossil not found at {fossil_path}")
        return

    print(f"Loading healed fossil: {fossil_path}")
    data = torch.load(fossil_path)
    
    # Reconstruct Dyad
    dyad = KnowledgeDyad(
        linguistic_description=data.get('description', 'Recovered Lazarus State'),
        image_fingerprint=data.get('image_fingerprint'),
        gyroid_residue=data.get('gyroid_residue'),
        meta_state=data.get('meta_state'),
        metadata=data.get('dyad_metadata', {})
    )
    
    # Derive Prime Frequencies (from meta_state)
    if dyad.meta_state is not None:
        # Use first 5 dimensions as prime frequency proxies (M=5)
        prime_frequencies = torch.abs(dyad.meta_state.flatten()[:5])
    else:
        prime_frequencies = torch.tensor([2, 3, 5, 7, 11], dtype=torch.float32)
        
    # Estimate Betti Numbers
    betti_numbers = {0: float(data.get('betti_0', 1)), 1: float(data.get('betti_1', 0))}
    
    # Initialize Fossilizer
    fossilizer = DyadFossilizer()
    
    # Execute Agent Smith Export
    print("Executing export_agent_smith protocol...")
    export_path = fossilizer.export_agent_smith(
        dyad=dyad,
        prime_frequencies=prime_frequencies,
        betti_numbers=betti_numbers,
        filename="soliton_smith_lazarus"
    )
    
    print(f"Agent Smith Protocol Complete. Identity exported to: {export_path}")
    
    # Verify JSON content
    with open(export_path, 'r') as f:
        payload = json.load(f)
        print(f"Verification: blake2s_digest = {payload.get('blake2s_digest')}")
        print(f"Verification: pestov_ionin_growth = {payload.get('pestov_ionin_growth_h_gamma')}")

if __name__ == "__main__":
    execute_v2()
