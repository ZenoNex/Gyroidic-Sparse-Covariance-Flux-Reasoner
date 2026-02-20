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
        self.modulation_basis = nn.Parameter(torch.randn(num_modes, hidden_dim, output_dim))
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
    def __init__(self, hidden_dim: int, vocab_size: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Learnable projection: Topology -> Symbols
        # initialized with high variance to promote "babbling"
        self.proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.1)
        
        # Confidence gate (how loud to speak)
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
           state: [batch, hidden_dim]
           temperature: scalar
           
        Returns:
           logits: [batch, vocab_size]
           confidence: [batch, 1]
        """
        logits = self.proj(state) / temperature
        conf = self.confidence(state)
        return logits, conf
        
    def hebbian_update(self, state_trace: torch.Tensor, symbol_trace: torch.Tensor, rate: float = 0.01):
        """
        Reinforce the path from state -> symbol.
        delta_W = rate * (symbol * state^T)
        
        Args:
            state_trace: [batch, hidden_dim] (The topological context)
            symbol_trace: [batch, vocab_size] (The one-hot symbol produced)
            rate: Learning rate (Reward)
        """
        with torch.no_grad():
            # Batch Hebbian update
            # sum over batch of O * I^T
            # symbol_trace: [B, V], state_trace: [B, H]
            # update: [V, H]
            update = torch.einsum('bv,bh->vh', symbol_trace, state_trace)
            
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
                          matrioshka_level: int = 0) -> Tuple[str, Dict]:
        """
        Autoregressive generation of diegetic response.
        Modulated by advanced physics states (System 2).
        """
        # 1. Modulate Dynamics based on Physics
        temperature = 1.0
        if quantum_state:
            # High temp for superposition (creative/chaotic)
            temperature = 1.5 
        
        constraint_mode = False
        if matrioshka_level >= 3:
            # Deep quantization -> Logic/Rigid mode
            temperature = 0.5
            constraint_mode = True
            
        # 2. Seed State from Context
        if context:
            # Average recent context for seed
            # context is list of [dim] tensors
            seed = torch.stack(context[-min(len(context), 5):]).mean(dim=0).unsqueeze(0) # [1, dim]
        else:
            seed = torch.randn(1, self.hidden_dim, device=self.proj.weight.device)
            
        # 3. Generation Loop
        # We simulate a "Singer" - the state evolves via self-resonance
        current_state = seed
        generated_chars = []
        confidence_sum = 0.0
        
        max_len = 150
        min_len = 20
        
        for i in range(max_len):
            logits, conf = self.forward(current_state, temperature=temperature)
            confidence_sum += conf.item()
            
            # Sampling
            probs = torch.softmax(logits, dim=-1)
            
            if constraint_mode:
                # Greedy decoding for rigid logic
                char_idx = torch.argmax(probs, dim=-1).item()
            else:
                # Stochastic sampling
                try:
                    char_idx = torch.multinomial(probs, 1).item()
                except:
                    char_idx = torch.argmax(probs, dim=-1).item() # Fallback
            
            # ASCII decoding (safe range)
            char_code = max(32, min(126, char_idx))
            char = chr(char_code)
            generated_chars.append(char)
            
            # Stop token logic (heuristic)
            if len(generated_chars) > min_len and char in ['.', '!', '?']:
                # Higher prob of stopping if confident
                if conf.item() > 0.8:
                    break
                    
            # 4. Diegetic State Evolution (Singing)
            # State rotates slightly based on emitted symbol (Reaction)
            # Non-linear feedback
            feedback = torch.tanh(self.proj.weight[char_idx].unsqueeze(0)) # [1, dim]
            current_state = 0.9 * current_state + 0.1 * feedback + 0.05 * torch.randn_like(current_state)
            
        final_text = "".join(generated_chars)
        
        metrics = {
            "avg_confidence": confidence_sum / len(generated_chars) if generated_chars else 0.0,
            "length": len(generated_chars),
            "temperature_used": temperature,
            "mode": "CONSTRAINT" if constraint_mode else ("QUANTUM" if quantum_state else "STANDARD")
        }
        
        return final_text, metrics
