import torch
import torch.nn as nn
from src.models.gyroid_reasoner import GyroidicFluxReasoner
from src.core.polynomial_coprime import PolynomialCoprimeConfig

def verify_formalized_system():
    print("--- Verifying Formalized Non-Teleological System ---")
    
    # 1. Test Initialization with Pressure Terminology
    try:
        model = GyroidicFluxReasoner(
            hidden_dim=64,
            num_functionals=5,
            poly_degree=4,
            use_admm=True
        )
        print("SUCCESS: GyroidicFluxReasoner initialized correctly.")
    except Exception as e:
        print(f"FAILED: Initialization error: {e}")
        return

    # 2. Test Forward Pass and Selection Pressure Output
    text_emb = torch.randn(2, 768)
    graph_emb = torch.randn(2, 256)
    
    try:
        outputs = model(text_emb, graph_emb)
        print("SUCCESS: Forward pass complete.")
        
        # Check for 'Pressure' keys in output dictionary
        if 'selection_pressure' in outputs and 'containment_pressure' in outputs:
            print(f"SUCCESS: Outputs use formalized terminology: S={outputs['selection_pressure']:.4f}, C={outputs['containment_pressure']:.4f}")
        else:
            print(f"FAILED: Output keys are legacy: {list(outputs.keys())}")
            
    except Exception as e:
        print(f"FAILED: Forward pass execution error: {e}")

    # 3. Verify Orthogonality Pressure specifically
    try:
        p_config = model.poly_config
        press = p_config.orthogonality_pressure()
        print(f"SUCCESS: Orthogonality Pressure computed: {press:.4f}")
    except AttributeError:
        print("FAILED: poly_config missing orthogonality_pressure method.")
    except Exception as e:
        print(f"FAILED: Orthogonality pressure computation error: {e}")

    print("--- Formalization Verification Complete ---")

if __name__ == "__main__":
    verify_formalized_system()
