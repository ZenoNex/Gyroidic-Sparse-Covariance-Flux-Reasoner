"""
Diegetic Responder Heads and Data Association Layers.

Provides the system with "Autoeclectic" output heads that warp latent states
into human-readable resonance, and "Data Association" input layers for
ingesting knowledge dyads (text/image pairs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from src.core.chern_simons_gasket import ChernSimonsGasket
from src.core.honest_jitter import harvest_honest_jitter

# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class DataAssociationLayer(nn.Module):
    """
    Ingestion layer for Knowledge Dyads.
    
    Fuses multi-modal inputs (e.g., Image semantics + Textual descriptions)
    into a unified polynomial residue representation.
    """
    def __init__(self, input_dim: int, hidden_dim: int, k: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        
        # Cross-modal projection
        self.text_prj = nn.Linear(input_dim, hidden_dim)
        self.img_prj = nn.Linear(input_dim, hidden_dim)
        
        # Fusion gate
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Residue projection
        self.residue_map = nn.Linear(hidden_dim, k)
        
    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        """
        Produce a co-prime residue vector from a knowledge dyad.
        """
        t = F.silu(self.text_prj(text_emb))
        i = F.silu(self.img_prj(image_emb))
        
        fused = torch.cat([t, i], dim=-1)
        latent = F.silu(self.fusion_gate(fused))
        
        # Map to co-prime field residues [batch, k]
        residues = torch.tanh(self.residue_map(latent))
        
        return residues

class AutoeclecticResponderHead(nn.Module):
    """
    Autoeclectic Diegetic Responder Head.
    
    Warps latent states through the topological "roughness" of the manifold
    to produce responses that reflect the system's current entropy/coherence.
    """
    def __init__(self, hidden_dim: int, output_dim: int, num_modes: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Diegetic modulation layers
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        self.modulation_basis = nn.Parameter(harvest_honest_jitter((num_modes, hidden_dim, output_dim), scaled=True))
        self.entropy_gate = nn.Linear(1, num_modes)
        
        # Final output projection
        self.out_prj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self, 
        state: torch.Tensor, 
        spectral_entropy: torch.Tensor,
        curvature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
           state: [batch, hidden_dim]
           spectral_entropy: [batch, 1] System logic health
           curvature: [batch, 1] Topological roughness
        """
        # 1. Compute diegetic modulation weights based on entropy
        # High entropy -> activate "chaotic" output modes
        # Low entropy -> converge to "stable" output modes
        mod_weights = torch.softmax(self.entropy_gate(spectral_entropy), dim=-1) # [batch, num_modes]
        
        # 2. Blend modulation basis
        # B = [batch, hidden_dim, output_dim]
        blended_basis = torch.einsum('bm,mho->bho', mod_weights, self.modulation_basis)
        
        # 3. Apply state transformation
        mischief_output = torch.einsum('bh,bho->bo', state, blended_basis)
        
        # 4. Standard residua
        base_output = self.out_prj(state)
        
        # 5. Non-linear fusion
        # If curvature is high (roughness), amplify the mischief
        mix = torch.sigmoid(curvature if curvature is not None else torch.zeros_like(spectral_entropy))
        
        final_output = (1 - mix) * base_output + mix * mischief_output
        
        return final_output

class ResonanceLarynx(nn.Module):
    """
    Project topological states to symbolic sequences (Characters/Tokens).
    Uses Hebbian learning to reinforce valid communication pathways.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable projection: Topology -> Next Topological State (JEPA)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.1)
        
        # Confidence gate (how loud to speak)
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Chern-Simons Gasket to repair logic leaks at linguistic boundary
        self.chern_simons = ChernSimonsGasket(manifold_dim=3)
        
    def forward(self, state: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
           state: [batch, hidden_dim]
           temperature: scalar
           
        Returns:
           predicted_state: [batch, hidden_dim]
           confidence: [batch, 1]
        """
        # Apply logic leak protection via Chern-Simons gasket
        # Ensure state is 3D [batch, K, D] for the gasket
        if state.dim() == 2:
            safe_state = state.unsqueeze(1)
        else:
            safe_state = state
            
        # Polynomial coeffs placeholder or derived from state
        # Use only the last dimension for D, and treat K as 1 if not explicit
        poly_placeholder = torch.ones(state.shape[0], state.shape[-1], device=state.device)
        safe_state = self.chern_simons.plug_logic_leak(safe_state, poly_placeholder)
        
        if state.dim() == 2:
            safe_state = safe_state.squeeze(1)
        else:
            safe_state = safe_state
        
        # Hazard Protection: Ensure temperature is never zero
        safe_temp = max(temperature, 1e-6)
        predicted_state = self.proj(safe_state) / safe_temp
        conf = self.confidence(safe_state)
        return predicted_state, conf
        
    def hebbian_update(self, state_trace: torch.Tensor, symbol_trace: torch.Tensor, rate: float = 0.01):
        """
        Reinforce the path from state -> symbol.
        delta_W = rate * (symbol * state^T)
        
        Args:
            state_trace: [batch, hidden_dim] (The topological context)
            target_state: [batch, hidden_dim] (The topological target)
            rate: Learning rate (Reward)
        """
        with torch.no_grad():
            # target_state: [B, H], state_trace: [B, H]
            # update: [H, H]
            update = torch.einsum('bv,bh->vh', target_state, state_trace)
            
            # Normalize update by batch size
            update = update / (state_trace.shape[0] + 1e-8)
            
            # Apply to weights
            self.proj.weight += rate * update
            
            # Normalization to prevent blowup (Oja's rule style damping)
            # W = W / norm(W)
            self.proj.weight.div_(torch.norm(self.proj.weight, dim=1, keepdim=True) + 1e-8)

    def generate_response(self, 
                          text_input: str, 
                          context: List[torch.Tensor], 
                          affordance_gradients: Dict[str, float],
                          quantum_state: bool = False,
                          matrioshka_level: int = 0) -> Tuple[torch.Tensor, Dict]:
        """
        Autoregressive generation of diegetic response (now topological).
        Modulated by advanced physics states (System 2).
        """
        temperature = 1.0
        if quantum_state:
            temperature = 1.5 
        
        constraint_mode = False
        if matrioshka_level >= 3:
            temperature = 0.5
            constraint_mode = True
            
        if context:
            seed = torch.stack(context[-min(len(context), 5):]).mean(dim=0).unsqueeze(0) 
        else:
            seed = harvest_honest_jitter((1, self.hidden_dim), device=self.proj.weight.device, scaled=True)
            
        current_state = seed
        generated_states = []
        confidence_sum = 0.0
        
        max_len = 60
        
        for i in range(max_len):
            predicted_state, conf = self.forward(current_state, temperature=temperature)
            confidence_sum += conf.item()
            
            generated_states.append(predicted_state)
            
            # Non-linear feedback
            feedback = torch.tanh(predicted_state)
            current_state = 0.9 * current_state + 0.1 * feedback + 0.05 * harvest_honest_jitter(current_state.shape, device=current_state.device, scaled=True)
            
        final_trajectory = torch.cat(generated_states, dim=0)
        
        metrics = {
            "avg_confidence": confidence_sum / max(len(generated_states), 1),
            "length": len(generated_states),
            "temperature_used": temperature,
            "mode": "CONSTRAINT" if constraint_mode else ("QUANTUM" if quantum_state else "STANDARD")
        }
        
        return final_trajectory, metrics
