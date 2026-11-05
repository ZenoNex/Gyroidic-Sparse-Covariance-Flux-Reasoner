"""
Diegetic Responder Heads and Data Association Layers.

Provides the system with "Autoeclectic" output heads that warp latent states
into human-readable resonance, and "Data Association" input layers for
ingesting knowledge dyads (text/image pairs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from src.core.chern_simons_gasket import ChernSimonsGasket
from src.core.honest_jitter import harvest_honest_jitter
from src.core.invariants import PHI, get_ihc_ladder_tensor

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
        
        # Inverted Hypersphere Cosmology (IHC): 33-shell golden ratio ladder scaling
        ihc_ladder = get_ihc_ladder_tensor(self.k, device=residues.device, dtype=residues.dtype)
        residues = residues * ihc_ladder
        
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
            
        # Polynomial coeffs derived dynamically from the input state to track geometry
        # Normalize state to use as basis coefficients
        poly_coeffs = torch.nn.functional.normalize(state, p=2, dim=-1)
        safe_state = self.chern_simons.plug_logic_leak(safe_state, poly_coeffs)
        
        if state.dim() == 2:
            safe_state = safe_state.squeeze(1)
        else:
            safe_state = safe_state
        
        # Hazard Protection: Ensure temperature is never zero
        safe_temp = max(temperature, 1e-6)
        
        # Chern-Simons Breather Mode Excitation: dresses phonemic projection with chiral soliton depth
        breather_packet = self.chern_simons.excite_chern_simons_breathers(safe_state)
        if breather_packet.shape != safe_state.shape:
            breather_packet = breather_packet.view_as(safe_state)
            
        predicted_state = (self.proj(safe_state) + 0.08 * breather_packet) / safe_temp
        conf = self.confidence(safe_state)
        return predicted_state, conf
        
    def hebbian_update(self, state_trace: torch.Tensor, symbol_trace: Optional[torch.Tensor] = None, target_state: Optional[torch.Tensor] = None, rate: float = 0.01):
        """
        Reinforce the path from state -> symbol.
        delta_W = rate * (symbol * state^T)
        
        Args:
            state_trace: [batch, hidden_dim] (The topological context)
            symbol_trace: [batch, hidden_dim] (The topological symbol trace / target)
            target_state: [batch, hidden_dim] (The topological target, alias for symbol_trace)
            rate: Learning rate (Reward)
        """
        if target_state is None:
            target_state = symbol_trace
        if target_state is None:
            return

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

    def compute_jepa_transition_energy(
        self,
        current_state: torch.Tensor,
        target_state: torch.Tensor,
        moment_transport: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Computes JEPA Monge-Ampere predictive energy for topological transitions under
        compute affordance gating. Falls back to normalized cosine distance in honeybee mode.
        """
        from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType
        hw = HardwareMonitor()
        if not hw.has_compute_affordance(ComputeAffordanceType.ALGORITHMIC):
            norm_curr = torch.nn.functional.normalize(current_state, p=2, dim=-1)
            norm_tgt = torch.nn.functional.normalize(target_state, p=2, dim=-1)
            return (1.0 - (norm_curr * norm_tgt).sum(dim=-1)).mean()
            
        if moment_transport is None:
            from src.core.conjugate_moment_transport import ConjugateMomentTransport
            moment_transport = ConjugateMomentTransport(dim=current_state.shape[-1]).to(current_state.device)
            
        # Hard-project context state to Cayley Cubic surface
        if hasattr(moment_transport, 'project_to_cayley_surface'):
            current_state = moment_transport.project_to_cayley_surface(current_state, is_honeybee_mode=False)
            
        pas_anisotropy = torch.ones(current_state.shape[0], 1, device=current_state.device)
        return moment_transport.compute_context_adaptive_monge_ampere_loss(
            source=current_state,
            target=target_state,
            pas_anisotropy=pas_anisotropy,
            shell_depth=0
        )

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

    def prove_self_play_drgrpo_step(
        self,
        seed_state: torch.Tensor,
        target_state: Optional[torch.Tensor] = None,
        love_protector: Optional[Any] = None,
        group_size: int = 4,
        lr: float = 0.005,
    ) -> Dict[str, Any]:
        """
        Live PROVE Perturb-Repair-Verify Self-Play DrGRPO Weight Mutation
        (Huang, Du, Snoek, docs/2611.05158v1.pdf):
          1. Mutator role: generates subtle topological perturbations z_tilde of clean seed_state
             with reward r_M = 0.4 * I[y_1 = 1] + 0.6 * I[y_2 = 0] - 0.1 * C_M.
          2. Dual-Check Verifier role: evaluates (y_1, y_2) on (clean, perturbed/repaired) pairs.
          3. Repairer role: repairs z_tilde back toward the valid manifold with reward
             r_R = 0.7 * I[y_2_rep = 1] + 0.3 * I[y_1 != y_2] - 0.1 * C_R.
          4. Applies mean-centered DrGRPO group relative advantage A_i = r_i - mean(r_G)
             to self.proj.weight, shielded by LoveInvariantProtector null-space projection.
        """
        from src.core.invariants import compute_prove_self_play_drgrpo_advantage

        x = seed_state.detach().float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.hidden_dim:
            if x.shape[-1] > self.hidden_dim:
                x = x[..., :self.hidden_dim]
            else:
                pad = torch.zeros((x.shape[0], self.hidden_dim - x.shape[-1]), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=-1)

        if target_state is None:
            with torch.no_grad():
                tgt, _ = self.forward(x, temperature=0.8)
                target = tgt.detach().float()
        else:
            target = target_state.detach().float()
            if target.dim() == 1:
                target = target.unsqueeze(0)
            if target.shape[-1] != self.hidden_dim:
                if target.shape[-1] > self.hidden_dim:
                    target = target[..., :self.hidden_dim]
                else:
                    pad = torch.zeros((target.shape[0], self.hidden_dim - target.shape[-1]), dtype=target.dtype, device=target.device)
                    target = torch.cat([target, pad], dim=-1)

        g_size = max(2, int(group_size))
        group_rewards = []
        group_updates = []
        y1_last, y2_last = 1, 0
        edit_cost_last = 0.1

        with torch.no_grad():
            pred_clean, conf_clean = self.forward(x, temperature=1.0)
            cos_clean = float(torch.nn.functional.cosine_similarity(pred_clean, target, dim=-1).mean().item())
            y1_verdict = 1 if (cos_clean >= 0.15 or float(conf_clean.mean().item()) >= 0.45) else 0
            y1_last = y1_verdict

            for g_idx in range(g_size):
                # 1. Mutator: subtle perturbation scaled by group index
                scale = 0.08 * float(g_idx + 1)
                jitter = harvest_honest_jitter(x.shape, device=x.device, scaled=True)
                x_mutated = x + scale * torch.tanh(jitter)
                edit_cost_m = float(torch.norm(x_mutated - x).item() / (torch.norm(x).item() + 1e-6))

                # 2. Verifier on mutated state
                pred_mut, _ = self.forward(x_mutated, temperature=1.0)
                cos_mut = float(torch.nn.functional.cosine_similarity(pred_mut, target, dim=-1).mean().item())
                # Subtle flaw: surface similarity remains moderate, but exact verification drops below clean
                y2_mut_verdict = 0 if cos_mut < (cos_clean - 0.01) else 1
                y2_last = y2_mut_verdict
                edit_cost_last = edit_cost_m

                # 3. Repairer: minimal targeted repair step
                repair_dir = torch.nn.functional.normalize(target - pred_mut, p=2, dim=-1)
                x_repaired = x_mutated + 0.15 * repair_dir
                pred_rep, _ = self.forward(x_repaired, temperature=0.9)
                cos_rep = float(torch.nn.functional.cosine_similarity(pred_rep, target, dim=-1).mean().item())
                y2_rep_verdict = 1 if cos_rep >= cos_mut else 0

                r_m = 0.4 * float(y1_verdict == 1) + 0.6 * float(y2_mut_verdict == 0) - 0.1 * edit_cost_m
                r_r = 0.7 * float(y2_rep_verdict == 1) + 0.3 * float(y1_verdict != y2_mut_verdict) - 0.1 * edit_cost_m
                combined_r = 0.5 * (r_m + r_r) + 0.2 * cos_rep
                group_rewards.append(float(combined_r))

                # Hebbian trace from repaired state -> target
                upd = torch.einsum("bv,bh->vh", target, x_repaired) / float(max(1, x_repaired.shape[0]))
                group_updates.append(upd)

            prove_stats = compute_prove_self_play_drgrpo_advantage(
                rewards=group_rewards,
                y1_verdict=y1_last,
                y2_verdict=y2_last,
                edit_cost=edit_cost_last,
            )
            advantages = prove_stats["drgrpo_advantages"]

            # Compute advantage-weighted policy gradient delta_W = sum_i A_i * upd_i
            delta_w = torch.zeros_like(self.proj.weight)
            for adv_i, upd_i in zip(advantages, group_updates):
                delta_w = delta_w + float(adv_i) * upd_i.to(delta_w.device, delta_w.dtype)
            delta_w = delta_w / float(g_size)

            # Optional LoveInvariantProtector null-space shielding
            if love_protector is not None and hasattr(love_protector, "Love") and love_protector.Love is not None:
                love_vec = love_protector.Love.detach().float().to(delta_w.device).flatten()
                if love_vec.numel() >= delta_w.shape[-1]:
                    lv = torch.nn.functional.normalize(love_vec[:delta_w.shape[-1]], p=2, dim=0)
                    proj_love = torch.mv(delta_w, lv).unsqueeze(-1) * lv.unsqueeze(0)
                    delta_w = delta_w - proj_love

            self.proj.weight.add_(float(lr) * delta_w)
            self.proj.weight.div_(torch.norm(self.proj.weight, dim=1, keepdim=True) + 1e-8)

        self.last_prove_stats = prove_stats
        return prove_stats
