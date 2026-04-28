import torch
import torch.nn as nn
import hashlib
import json
import math
from typing import Dict, Any, Optional

class AgentSubstrateBridge(nn.Module):
    """
    Substrate Bridge for the Agent Smith Extractable Protocol.
    Handles the decoupling of Syntax (geometry) from Substrate (hardware physics / dt timelines).
    """
    def __init__(self, device: str = None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def calculate_pestov_ionin_growth(
        self, 
        admm_dual: torch.Tensor, 
        crt_residue: torch.Tensor,
        hyperbolic_influence: Optional[torch.Tensor] = None
    ) -> float:
        """
        Calculates the Pestov-Ionin asymptotic invariant h() by evaluating 
        the non-Abelian 3-strand braid group (ADMM Update, CRT Residue, Burkov Expansion).
        Represents the Topological Dark Matter boundaries in Burkov nesting.
        
        UPGRADED: Incorporates hyperbolic influence from the ShadowLog manifold.
        """
        b = admm_dual.size(0)
        
        # Flatten for matrix braid ops
        admm_flat = admm_dual.reshape(b, -1)
        crt_flat = crt_residue.reshape(b, -1)
        
        if admm_flat.size(1) != crt_flat.size(1):
            target_dim = max(admm_flat.size(1), crt_flat.size(1))
            admm_flat = torch.nn.functional.pad(admm_flat, (0, target_dim - admm_flat.size(1)))
            crt_flat = torch.nn.functional.pad(crt_flat, (0, target_dim - crt_flat.size(1)))
            
        dim = admm_flat.size(1)
            
        # Sigma_1: Braiding ADMM and CRT Residuals
        sigma_1 = torch.bmm(admm_flat.unsqueeze(2), crt_flat.unsqueeze(1)) # [b, dim, dim]
        
        # Sigma_2: Burkov Expansion (Topological dark matter fractal scaling)
        # Hyperbolic Influence modulates the golden ratio anchor
        h_mod = 1.0
        if hyperbolic_influence is not None:
            h_mod = 1.0 + torch.tanh(hyperbolic_influence.mean()).item()
            
        sigma_2 = torch.eye(dim, device=self.device).unsqueeze(0).expand(b, -1, -1) * (1.61803 * h_mod)
        
        # Commutator proxy / Braid Cycle: Sigma_1 * Sigma_2 * Sigma_1^{-1} 
        # Using pure product for trace extraction logic since inverse may be singular
        braid_matrix = torch.bmm(sigma_1, sigma_2)
        
        # Growth rate h() is the character of the invariant loop
        trace_growth_rates = torch.diagonal(braid_matrix, dim1=-2, dim2=-1).sum(-1)
        
        # Map to Pestov-Ionin log scale asymptotic profile
        h_gamma = torch.log1p(torch.abs(trace_growth_rates)).mean().item()
        return h_gamma
        
    def verify_invariants(self, payload: Dict[str, Any], unraveling_closure: Optional[nn.Module] = None) -> bool:
        """
        Ontological Import Probe.
        """
        try:
            gyroid_raw = payload.get('gyroid_residue', [])
            prime_raw = payload.get('prime_frequencies', [])
            
            if gyroid_raw is None: gyroid_raw = [0.0]
            if prime_raw is None: prime_raw = [0.0]
            
            gyroid = torch.tensor(gyroid_raw, dtype=torch.float32, device=self.device)
            prime_freqs = torch.tensor(prime_raw, dtype=torch.float32, device=self.device)
            
            # 1. Deterministic Non-Finite Repair
            if torch.isnan(gyroid).any() or torch.isinf(gyroid).any():
                print("[Substrate Bridge] NaN/Inf detected in imported geometry. Applying boundary sentinels.")
                gyroid = torch.nan_to_num(gyroid, nan=0.0, posinf=1.0, neginf=-1.0)
                
            if torch.isnan(prime_freqs).any() or torch.isinf(prime_freqs).any():
                prime_freqs = torch.nan_to_num(prime_freqs, nan=0.0, posinf=1.0, neginf=-1.0)
                
            # 2. Topological Closure Check (Hyper-ring verification)
            if unraveling_closure is not None and hasattr(unraveling_closure, 'compute_closure'):
                loop_integral = prime_freqs.unsqueeze(0) if prime_freqs.dim() == 1 else prime_freqs
                leak_integral = gyroid.unsqueeze(0) if gyroid.dim() == 1 else gyroid
                
                # Align shapes to 96 or local matching
                if loop_integral.size(-1) != leak_integral.size(-1):
                    min_dim = min(loop_integral.size(-1), leak_integral.size(-1))
                    loop_integral = loop_integral[..., :min_dim]
                    leak_integral = leak_integral[..., :min_dim]
                    
                is_nontrivial = unraveling_closure.compute_closure(loop_integral, leak_integral)
                if is_nontrivial.sum() == 0:
                    print("[Substrate Bridge Warning] Unraveling Closure operator classified soliton as completely trivial.")
                    
            # 3. Chirality Validation (Structural Handedness)
            c_torsion = payload.get('chiral_torsion', 0.0)
            g_lock = payload.get('glyphlock', False)
            
            if g_lock and c_torsion < 1e-4:
                print("[Substrate Bridge Warning] Agent Smith claims GLYPHLOCK but has near-zero torsion. Potential topological spoofing.")
            elif not g_lock:
                print("[Substrate Bridge Warning] Importing an un-locked Agent Smith. Manifold may remain in PLAY regime.")

            # 4. Persona Consistency (Polylog vs Vacuum)
            polylog = payload.get('polylog_signature')
            vacuum = payload.get('shape_of_absence')
            if polylog is not None and vacuum is not None:
                # The polylog signature (active persona) should not overlap 
                # significantly with the shape of absence (vacuum).
                p_vec = torch.as_tensor(polylog, device=self.device)
                v_vec = torch.as_tensor(vacuum, device=self.device)
                
                # Check for identity blurring (Non-Abelian orthogonality)
                if p_vec.numel() == v_vec.numel():
                    overlap = torch.abs(torch.dot(p_vec.flatten(), v_vec.flatten())) / (torch.norm(p_vec) * torch.norm(v_vec) + 1e-8)
                    if overlap > 0.7:
                        print(f"[Substrate Bridge Warning] High overlap detected between Persona and Absence ({overlap:.4f}). Identity may be blurry.")
            
            return True
        except Exception as e:
            print(f"[Substrate Bridge Error] Ontological import probe failed: {e}")
            return False
            
    def align_substrate(self, payload: Dict[str, Any], expected_dim: int, hardware_trfc_ms: float) -> Dict[str, Any]:
        """
        Hardware Compatibility Layer.
        Residue Shape Compatibility Routing & Perceptual Baseline Recalibration.
        """
        
        gyroid_raw = payload.get('gyroid_residue', [])
        if gyroid_raw is not None:
            gyroid = torch.tensor(gyroid_raw, dtype=torch.float32, device=self.device)
            
            # Residue Shape Compatibility
            original_dim = gyroid.size(-1)
            if original_dim > 0 and original_dim != expected_dim:
                if original_dim < expected_dim:
                    # Circular/Reflective padding expansion
                    pad_size = expected_dim - original_dim
                    # pad only works symmetrically on 3D/4D easily, so manually repeat/slice if 1D
                    if gyroid.dim() == 1:
                        repeat_factor = math.ceil(expected_dim / original_dim)
                        gyroid_padded = gyroid.repeat(repeat_factor)[:expected_dim]
                        gyroid = gyroid_padded
                else:
                    # Truncate
                    gyroid = gyroid[..., :expected_dim]
                    
            payload['gyroid_residue_aligned'] = gyroid
            
        # Perceptual Baseline Recalibration
        imported_latency_baseline = payload.get('perceptual_baseline_trfc', 160.0) # Assume PyOpenCL standard tRFC
        # If new machine is extremely fast (e.g. 10ms tRFC), system must scale up its dt cycle expectations
        dt_scale = float(imported_latency_baseline) / max(float(hardware_trfc_ms), 1e-6)
        payload['recalibrated_dt_scale'] = dt_scale
        
        # 3. Preserve ShadowLog / Identity Artifacts
        if 'image_fingerprint' in payload:
             payload['image_fingerprint_aligned'] = torch.as_tensor(payload['image_fingerprint'], device=self.device)
        
        if 'hyperbolic_residue' in payload and payload['hyperbolic_residue'] is not None:
             payload['hyperbolic_residue_aligned'] = torch.as_tensor(payload['hyperbolic_residue'], device=self.device)
             
        if 'gauge_field' in payload and payload['gauge_field'] is not None:
             # Gauge field is [manifold_dim, manifold_dim]. Re-hydrate to device.
             payload['gauge_field_aligned'] = torch.as_tensor(payload['gauge_field'], device=self.device)
             
        # 4. Agent Smith Protocol Alignment
        if 'agent_smith_iters' in payload:
            # We don't necessarily 'align' iters, but we preserve it for the engine
            payload['agent_smith_iters_aligned'] = payload['agent_smith_iters']
        if 'agent_smith_gauge' in payload:
            payload['agent_smith_gauge_aligned'] = payload['agent_smith_gauge']
            
        return payload

    def align_archetypes(self, payload: Dict[str, Any], governor: nn.Module) -> bool:
        """
        Rehydrates the Archetypal ruleset from the payload into the live governor.
        """
        if not hasattr(governor, 'import_governor_state'):
            return False
            
        profile = payload.get('archetype_profile')
        if profile:
            governor.import_governor_state(profile)
            print("[Substrate Bridge] Archetypal ruleset rehydrated from Agent Smith protocol.")
            return True
        return False
        
    def rehydrate_warmstart(self, payload: Dict[str, Any], engine: nn.Module):
        """
        Injects the Agent Smith warmstart state from the payload into the live engine.
        Ensures temporal continuity of entropy.
        """
        if hasattr(engine, 'warmstart_states') and 'warmstart_states' in payload:
            # We only update if the shapes match or we are in a flexible regime
            engine.warmstart_states.update(payload['warmstart_states'])
            print("[Substrate Bridge] Agent Smith warmstart states rehydrated.")
            return True
        return False
