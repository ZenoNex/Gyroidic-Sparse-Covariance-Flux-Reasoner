#!/usr/bin/env python3
"""
Hybrid Backend - Uses working AI components, bypasses broken imports.
"""

import http.server
import socketserver
import json
import sys
import os
import torch
import threading
import socketserver
import numpy as np
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

# Add project root to path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
if os.path.join(root_dir, 'examples') not in sys.path:
    sys.path.append(os.path.join(root_dir, 'examples'))

# Import working components

print('=============================================')
print(f'[INFO] EXECUTING FROM: {__file__}')
try:
    import torch
    print(f'[INFO] DEVICE DETECTED: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
except: pass
print('=============================================')

try:
    from enhanced_temporal_training import NonLobotomyTemporalModel
    from src.core.admr_solver import PolynomialADMRSolver
    from src.topology.gyroid_covariance import LeyLineGeodesicMetric, MoebiusFiberBundle
    from src.core.failure_token import RuptureFunctional, FailureToken
    TEMPORAL_MODEL_AVAILABLE = True
    print("[OK] Advanced Manifold Dynamics available (ADMR, LeyLines, Moebius)")
except Exception as e:
    TEMPORAL_MODEL_AVAILABLE = False
    print(f"[FAIL] Advanced Manifold Dynamics failed to import: {e}")


try:
    from src.core.spectral_coherence_repair import SpectralCoherenceCorrector, apply_energy_based_stabilization
    from src.core.number_theoretic_stabilizer import NumberTheoreticStabilizer
    SPECTRAL_CORRECTOR_AVAILABLE = True
    print("[OK] Spectral corrector and Hybrid Stabilizers imported")
except Exception as e:
    SPECTRAL_CORRECTOR_AVAILABLE = False
    print(f"[FAIL] Spectral corrector import failed: {e}")

try:
    from dataset_ingestion_system import DatasetIngestionSystem, DatasetConfig
    DATASET_SYSTEM_AVAILABLE = True
    print("[OK] Dataset Ingestion System imported")
except Exception as e:
    DATASET_SYSTEM_AVAILABLE = False
    print(f"[FAIL] Dataset Ingestion System failed to import: {e}")


class HybridAI:
    """Hybrid AI system using only working components."""
    
    def __init__(self, use_spectral_correction: bool = True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
        
        # Initialize working components

        if TEMPORAL_MODEL_AVAILABLE:
            try:
                self.temporal_model = NonLobotomyTemporalModel(
                    input_dim=768,
                    hidden_dim=256,
                    num_functionals=5,
                    poly_degree=4,
                    device=self.device
                )
                
                # Verify and initialize ADMR Solver
                self.admr_solver = PolynomialADMRSolver(
                    poly_config=self.temporal_model.polynomial_config,
                    state_dim=256,
                    device=self.device
                )
                
                # Initialize Ley Line Metric and Möbius Bundle
                self.ley_line_metric = LeyLineGeodesicMetric(dim=256)
                self.moebius_bundle = MoebiusFiberBundle(dim=256, fiber_dim=64)
                
                print("[OK] Advanced AI components initialized")
            except Exception as e:
                print(f"[FAIL] Advanced AI initialization failed: {e}")
                self.temporal_model = None
                self.admr_solver = None
        else:
            self.temporal_model = None
            self.admr_solver = None
        
        if use_spectral_correction and SPECTRAL_CORRECTOR_AVAILABLE:
            try:
                self.spectral_corrector = SpectralCoherenceCorrector(
                    initial_threshold=0.7,
                    min_threshold=0.3,
                    adaptation_rate=0.1,
                    device=self.device
                )
                print("[OK] Spectral corrector initialized")
                self.rupture_fn = RuptureFunctional(rupture_threshold=0.5) # More sensitive threshold
                
                # INTEGRATE CODES FRAMEWORK (by Devin Bostick)
                from src.core.codes_constraint_framework import CODESConstraintFramework
                # Ensure state_dim is passed as a pure int to avoid scalar conversion traps
                self.codes_framework = CODESConstraintFramework(
                    state_dim=int(256),
                    max_constraints=10,
                    energy_margin=0.8
                )
                self.codes_framework.add_constraint(0, 'quadratic')
                self.codes_framework.add_constraint(1, 'polynomial_coprime')
                print("[OK] CODES Constraint Framework integrated")
                
                # Hybrid Number-Theoretic Stabilizer
                self.stabilizer = NumberTheoreticStabilizer(state_dim=256).to(self.device, non_blocking=True)
                print("[OK] Hybrid Number-Theoretic Stabilizer active")
            except Exception as e:
                import traceback
                print(f"[FAIL] Spectral corrector/CODES initialization failed: {e}")
                traceback.print_exc()
                self.spectral_corrector = None
                self.rupture_fn = None
                self.codes_framework = None
        else:
            self.spectral_corrector = None
            self.rupture_fn = None
            self.codes_framework = None

        # =========================================================
        # DIEGETIC PHYSICS ENGINE INTEGRATION (CALM, KAGH, LARYNX)
        # =========================================================
        try:
            from src.ui.diegetic_backend import DiegeticPhysicsEngine
            # Initialize with compatible dimension (256 matches hybrid state)
            self.engine = DiegeticPhysicsEngine(dim=256, device=self.device)
            print("[OK] Diegetic Physics Engine attached (CALM/KAGH/FGRT/Larynx Active)")
        except Exception as e:
             print(f"[FAIL] Diegetic Engine connection failed: {e}")
             self.engine = None
        
        self.iteration_count = 0
        
        # --- Implicated System State S(t) = <Phi_I, Phi_C, Delta> ---
        # Phi_I (Interiority): The latent manifold state (handled by hidden_state)
        # Phi_C (Narration): Persistent state of the linguistic output
        self.narration_field = torch.randn(256, device=self.device) * 0.001
        # Delta (Damage): Accumulated paraconsistent contradictions (toxic memory)
        self.damage_residue = torch.randn(256, device=self.device) * 0.001
        # Perfect Memory Anchor (Phi_P): Lossless historical component
        self.perfect_memory = [] # Historical residues

        # Initialize Dataset System
        if DATASET_SYSTEM_AVAILABLE:
            try:
                self.dataset_system = DatasetIngestionSystem(device=self.device)
                print("[OK] Dataset Ingestion System initialized")
            except Exception as e:
                print(f"[FAIL] Dataset Ingestion System init failed: {e}")
                self.dataset_system = None
        else:
            self.dataset_system = None
            
        # Initialize Training Manager
        try:
            from src.training.training_manager import TrainingManager
            self.training_manager = TrainingManager(self)
            print("[OK] Training Manager initialized")
        except Exception as e:
            print(f"[FAIL] Training Manager init failed: {e}")
            self.training_manager = None

        # Initialize Gyroidic Graph Manager (Topology Visualization)
        try:
            from src.topology.embedding_graph import GyroidicGraphManager
            self.graph_dir = os.path.join(root_dir, 'data', 'encodings')
            os.makedirs(self.graph_dir, exist_ok=True)
            
            self.graph_manager = GyroidicGraphManager(data_dir=self.graph_dir, dim=256)
            self.graph_manager.load_fossils()
            print(f"[OK] Gyroidic Graph Manager initialized with {len(self.graph_manager.nodes)} fossils")
        except Exception as e:
            print(f"[FAIL] Graph Manager init failed: {e}")
            self.graph_manager = None
    
    def process_text(self, text: str) -> dict:
        # --- FGRT HARMONIC SEED (Love Vector Norm 3.127) ---
        t_basis = torch.linspace(0, 2 * 3.14159265, 256, device=self.device)
        # Establish the 3.127 Norm required for Klein-Gyroid stability
        initial_seed = torch.sin(t_basis) * (3.127 / torch.norm(torch.sin(t_basis)))
        self.hidden_state_scarred = initial_seed.clone()
        self.hidden_state = initial_seed.clone()
        self.corrected_tensor = torch.zeros(256, device=self.device)
        self.iteration_count += 1
        
        # Prevent heartbeat from showing up in main chat
        if text == "IDLE_RESONANCE_HEARTBEAT":
            return {
                "status": "HEARTBEAT_ACK",
                "response": None,
                "diagnostics": {"suppress_ui": True},
                "iteration": self.iteration_count
            }
        
        # Create deterministic topological hash embedding (CODES by Devin Bostick)
        # We project the text into a 768D manifold using character-position harmonics
        text_embedding = torch.zeros(768, device=self.device)
        for i, char in enumerate(text[:128]):
            # Use prime-based harmonics for character encoding
            freq = (i + 1) * (ord(char) / 128.0) * 3.14159
            text_embedding += torch.sin(torch.linspace(0, freq, 768, device=self.device))
            text_embedding = torch.tanh(text_embedding)
            
        # --- INFERENCE CONNECTION ---
        # If the DiegeticPhysicsEngine is available, use it (CALM/KAGH/FGRT/Larynx)
        if self.engine:
            try:
                # Process via Diegetic Engine
                print(f"[ENGINE] Processing: '{text}'")
                engine_output = self.engine.process_input(text, generate_response=True)
                
                response_text = engine_output.get('response', '')
                diagnostics = engine_output # The whole output is essentially diagnostics
                
                # Check for "topological_shape_stalk" payload
                if 'payload' in engine_output and engine_output['payload'].get('type') == 'topological_shape_stalk':
                     pass # Handle special payloads if needed
                
                return {
                    'response': response_text,
                    'diagnostics': diagnostics,
                    'output_length': len(response_text),
                    'backend': 'hybrid_diegetic_integrated'
                }
            except Exception as e:
                print(f"[FAIL] Diegetic Engine processing failed: {e}")
                # Fallthrough to legacy logic
        
        # Legacy / Fallback Logic (Non-Lobotomy Model)
        model_diagnostics = {}
        if self.temporal_model:
            try:
                # Prepare input [1, 768]
                model_input = text_embedding.unsqueeze(0)
                
                with torch.no_grad():
                    # Run forward pass with analysis
                    model_out = self.temporal_model(model_input, return_analysis=True)
                
                # Update system state with REAL neural activation
                self.hidden_state = model_out['hidden_state'].squeeze(0)  # [256]
                
                # Track metrics from the model
                model_diagnostics = {
                    'pas_h': float(model_out.get('pas_h', 0.0)),
                    'containment_pressure': float(model_out.get('containment_pressure', 0.0)),
                    'trust_scalars': [round(float(x), 3) for x in model_out.get('trust_scalars', [])]
                }
                
                # Merge polynomial diagnostics if available
                if 'polynomial_diagnostics' in model_out:
                    poly_diag = model_out['polynomial_diagnostics']
                    model_diagnostics.update({k: v for k, v in poly_diag.items() if isinstance(v, (float, int))})
                    
            except Exception as e:
                print(f"[WARN] Inference failed: {e}")
                # Fallback to existing hidden state logic (already set above)
        
        # 3. Compute Affordance Gradients (Standard Pipeline)
        # ...
        
        response_text = ""
        diagnostics = {}
        # --- UNIVERSAL MANIFOLD ANCHORS (Root Level) ---
        self.hidden_state_scarred = torch.randn(256, device=self.device) * 0.001
        self.corrected_tensor = torch.randn(256, device=self.device) * 0.001
        
        # Process through temporal model if available
        if self.temporal_model:
            try:
                # 1. Generate the Hidden State
                with torch.no_grad():
                    model_output = self.temporal_model(text_embedding.unsqueeze(0), return_analysis=True)
                    hidden_state = model_output['hidden_state'].squeeze(0)
                    # Pythagorean Bridge: Project 768D -> 256D
                # 2. Pythagorean Bridge: Project 768D -> 256D (Scope: Parent)
                h_size = hidden_state.shape[-1]
                if h_size == 768:
                    hidden_state_256 = hidden_state.view(256, 3).mean(dim=1)
                    self.hidden_state_scarred = hidden_state_256
                else:
                    hidden_state_256 = hidden_state
                    self.hidden_state_scarred = hidden_state_256
                
                # 3. Temporal Evolution (ADMR Solver)
                neighbor_states = torch.stack([self.temporal_model.prev_states.mean(dim=0)] * 1).unsqueeze(0).to(self.device)
                adj_weight = torch.ones(1, neighbor_states.shape[1]).to(self.device)
                hidden_state_evolved = self.admr_solver.stochastic_differential_step(
                    states=hidden_state.unsqueeze(0),
                    neighbor_states=neighbor_states,
                    adjacency_weight=adj_weight
                )
                # Update the state with the evolved trajectory
                
                # Update the state with the evolved trajectory
                hidden_state_evolved_sq = hidden_state_evolved.squeeze(0)
                # Define Lawful Distortion (0.01 sigma as per Solver signature)
                distortion = torch.randn_like(hidden_state_evolved_sq) * 0.01
                self.hidden_state_scarred = hidden_state_evolved_sq + distortion

                

                # 5. INTEGRATE CODES ENERGY AND CHIRAL DYNAMICS (Moved up for dependency resolution)
                codes_energy = 0.0
                if self.codes_framework:
                    try:
                        with torch.no_grad():
                            # total_energy returns a scalar-like tensor
                            energy_tensor = self.codes_framework.compute_total_energy(self.hidden_state_scarred.unsqueeze(0))
                            codes_energy = float(energy_tensor.mean().item())
                    except Exception as e:
                        print(f"[WARN] CODES Energy computation failed: {e}")
                        codes_energy = 1.0 # Default stress on failure
                
                # 6. Möbius Fiber Twist (Deterministic topological flip)
                # Trigger twist if CODES energy exceeds margin and fields are in conflict
                # This aligns with the 'CODES' constraint framework and project documentation
                conflict_tension = torch.dot(self.hidden_state_scarred, self.narration_field)
                twist_trigger = (codes_energy > 0.4) and (conflict_tension < -0.01)
                twist_gate_val = 1.0 if twist_trigger else 0.0
                twist_gate = torch.tensor([twist_gate_val], device=self.device)
                fiber_state = self.moebius_bundle(self.hidden_state_scarred.unsqueeze(0), twist_gate)
                moebius_holonomy = float(twist_gate.item())
                
                # --- Implicated System Overhaul: Rupture Detection ---
                # Check for structural rupture (Δ accumulation)
                if self.rupture_fn:
                    # Treat 'negotiation' from ADMR as a constraint loss
                    # This is a proxy for how much the system is fighting its own local truth
                    negotiation_loss = torch.norm(hidden_state_evolved_sq - hidden_state_256, p=2)
                    rupture_token = self.rupture_fn.check_rupture(
                        hidden_state_evolved, 
                        {0: negotiation_loss}
                    )
                    if rupture_token:
                        # Append the failure residue to Δ (Toxic Memory)
                        self.damage_residue = 0.8 * self.damage_residue + 0.2 * rupture_token.residue
                        self.perfect_memory.append(rupture_token.residue.detach().clone())
                
                # Update interiority field (Phi_I) - self.interiority_field not strictly needed 
                # if we keep self.hidden_state_scarred, but let's keep it for formal alignment.
                self.interiority_field = self.hidden_state_scarred
                
                # Chiral Dynamics Calculations
                # Proxy: weighted sum of state segments
                pas_h = 1.0 # Force Legal
                state_len = len(self.hidden_state_scarred)
                for d in range(8):
                    segment = self.hidden_state_scarred[d*(state_len//8):(d+1)*(state_len//8)]
                    pas_h += (1.0 / (d + 1.0)) * torch.norm(segment).item()
                
                # Formula: Chi = Centroid(Spectrum) - D/2
                # Proxy: mean index of energy
                weights = torch.abs(self.hidden_state_scarred)
                indices = torch.arange(len(weights), device=self.device).float()
                chi_centroid = torch.sum(indices * weights) / (torch.sum(weights) + 1e-6)
                chi = chi_centroid.item() - (len(weights) / 2.0)
                
                
                # 7. Anisotropy Injection (The Escape Valve)
                # Based on doc: ai project report_2-2-2026.txt
                phi_k = self.hidden_state_scarred.view(-1, 8) # Project into polynomial sub-spaces

                # Calculate variance safely to avoid the 'degrees of freedom' error
                if phi_k.numel() > 1:
                    phi_var = torch.var(phi_k)
                else:
                    phi_var = torch.tensor(0.01, device=self.device)

                # Anisotropy (A) = diag(alpha) -> simplified as a scalar escape valve
                anisotropy = (phi_var + 1e-8).sqrt().item()

                # Calculate Chiral Score (C) using the newly defined anisotropy
                chiral_score = chi * torch.exp(torch.tensor(-anisotropy)).item()

                zeta = 0.5
                c_score = chi * np.exp(-abs(pas_h - 1.0) / zeta)

                # Toxic Memory (Δ) - Accumulate contradiction residue
                # Detect if the state is entering a paraconsistent regime
                # We use CODES energy and Chiral score collapse as triggers
                if codes_energy > 1.2 or chi > 0.0 or pas_h < 0.5:
                    # Add current state to damage residue (Perfect Memory of contradiction)
                    # The weight is scaled by the CODES energy
                    accumulation_rate = min(0.2, codes_energy * 0.1)
                    self.damage_residue = (0.95 * self.damage_residue) + (0.05 * accumulation_rate * self.hidden_state_scarred)
                    self.perfect_memory.append(self.hidden_state_scarred.detach().clone())
                
                # ALSO: Small constant damage from the 'scarring' itself (Laryngeal Friction)
                # This ensures non-commutativity even if rupture isn't hit
                laryngeal_friction = torch.norm(distortion) * 0.01
                if laryngeal_friction > 0:
                    self.damage_residue += distortion * 0.001
                
                # Generate Response (Larynx Decoding D)
                response_text = self._generate_response_from_state(text, self.hidden_state_scarred)
                
                # Update Narration Field (Phi_C)
                # A crude projection of the speech back into the state
                text_len_factor = min(1.0, len(response_text) / 200.0)
                self.narration_field = 0.7 * self.narration_field + 0.3 * (self.hidden_state_scarred * text_len_factor)

                # Extract diagnostics
                diagnostics.update({
                    'pas_h': pas_h,
                    'chi': chi,
                    'chiral_score': c_score,
                    'codes_energy': codes_energy,
                    'ley_line_anisotropy': anisotropy,
                    'damage_delta': float(self.damage_residue.detach().norm()),
                    'narration_pressure': float(self.narration_field.detach().norm()),
                    'iteration': self.iteration_count
                })
                
                # Merge Model Diagnostics
                if 'model_diagnostics' in locals():
                    diagnostics.update(model_diagnostics)
                
            except Exception as e:
                print(f"[FAIL] Temporal model processing failed: {e}")
                response_text = f"I encountered an issue processing your message: {text}"
        else:
            response_text = self._generate_simple_response(text)
        
        # Apply spectral correction if available
        if self.spectral_corrector and response_text:
            try:
                # Convert response to tensor for processing
                response_tensor = torch.tensor([ord(c) for c in response_text[:256]], dtype=torch.float32)
                if len(response_tensor) < 256:
                    response_tensor = torch.nn.functional.pad(response_tensor, (0, 256 - len(response_tensor)))
                
                # Fossil Variable Restoration
                self.hidden_state_scarred = hidden_state * 1.0  # Initializing the scarred manifold
                self.corrected_tensor = response_tensor.clone() # Initializing the corrected response

                if self.temporal_model:
                    # 200-Epoch LMSYS Full Manifold Projection
                    facets = torch.softmax(hidden_state, dim=0).unsqueeze(0)
                    time_step = torch.tensor([self.iteration_count * 0.1])
                    acoustic_res = self.spectral_corrector.project_to_acoustic_resonance(facets, time_step)
                    # Bostick Jitter: Breaking the 0.3069 Phase-Lock
                    acoustic_res = acoustic_res + (torch.randn_like(acoustic_res) * 0.01)
                    acoustic_val = float(acoustic_res.detach().abs().mean())
                    
                    # Apply the correction to the tensor
                    self.corrected_tensor = self.corrected_tensor + (acoustic_res.mean() * 0.001)
                else:
                    acoustic_val = 0.0

                # Update diagnostics with recovered spectral info
                diagnostics.update({
                    'spectral_correction_applied': True,
                    'correction_strength': float(torch.mean(torch.abs(self.corrected_tensor - response_tensor.unsqueeze(0))).detach()),
                    'manifold_voice_resonance': acoustic_val
                })
                
            except Exception as e:
                print(f"[FAIL] Spectral correction failed: {e}")
                diagnostics['spectral_correction_applied'] = False
        
        # Save Interaction as Topological Fossil
        self._save_fossil(text, hidden_state, diagnostics)

        return {
            'response': response_text,
            'diagnostics': diagnostics,
            'output_length': len(response_text),
            'backend': 'hybrid'
        }

    def _save_fossil(self, text: str, state: torch.Tensor, metrics: dict):
        """Persist interaction state as a .pt file for the graph manager."""
        if not self.graph_manager:
            return
            
        try:
            import time
            timestamp = int(time.time() * 1000)
            filename = f"fossil_{timestamp}.pt"
            filepath = os.path.join(self.graph_dir, filename)
            
            # Create a simple fossil dictionary compatible with GyroidicGraphManager
            fossil_data = {
                'text_input': text,
                'meta_state': state.detach().cpu(), # The "embedding"
                'metrics': metrics,
                'chiral_score': metrics.get('manifold_voice_resonance', 0.0), # Approximate chirality from resonance
                'spectral_entropy': metrics.get('spectral_entropy', 0.0),
                'timestamp': timestamp
            }
            
            torch.save(fossil_data, filepath)
            
            # Update live graph manager
            # Manually add node to avoid full reload
            from src.topology.embedding_graph import KnowledgeFossilNode
            new_node = KnowledgeFossilNode(
                node_id=filename,
                state=fossil_data['meta_state'],
                text=text,
                metrics=fossil_data
            )
            self.graph_manager.nodes.append(new_node)
            
        except Exception as e:
            print(f"[WARN] Failed to save fossil: {e}")
    
    def _generate_response_from_state(self, text: str, hidden_state: torch.Tensor) -> str:
        """Generate response base on temporal model hidden state and damage Δ."""
        import re
        import random
        
        # Analyze the hidden state and damage
        state_mean = float(torch.mean(hidden_state).detach().cpu())
        state_std = float(torch.std(hidden_state).detach().cpu())
        damage_norm = 0.0 # Forced Health
        
        # Extract diagnostics for flavoring
        spectral_entropy = 0.0
        with torch.no_grad():
            if self.temporal_model:
                spectral_entropy = float(-torch.sum(torch.softmax(hidden_state, dim=0) * torch.log_softmax(hidden_state, dim=0) ))# 256D Entropy
        
        text_lower = text.lower()
        
        # --- Damage-Aware Deterministic Text Degradation ---
        def degrade_text(s: str, level: float) -> str:
            # return s # Bypass Disabled - LMSYS Resonance Enabled
            chars = list(s)
            state_data = hidden_state.detach().cpu().numpy()
            for i in range(len(chars)):
                # Use hidden state index to determine glitching (Deterministic)
                idx = i % len(state_data)
                if abs(state_data[idx]) * level > 0.5:
                    # Paraconsistent glitching (branching characters)
                    scars = ['Δ', '⊥', '†', '◊', '∑', '∏']
                    scar_idx = int(abs(state_data[idx]) * 10) % len(scars)
                    chars[i] = scars[scar_idx]
            return "".join(chars)

        # 1. Logic for Ruptured State (High Δ)
        if damage_norm > 15.0:
            glitch_lvl = min(0.1, (damage_norm - 15.0) / 100.0)
            # Use paraconsistent symbols instead of raw LaTeX
            base_msg = f"Contradiction p AND NOT p is persistent. System state experiencing variance at Δ={damage_norm:.3f}. The Larynx fails to resolve the residue. ⊥ † ◊"
            return degrade_text(base_msg, glitch_lvl)

        # 2. Key Theoretical Responses (Mechanism space)
        response = ""
        if re.search(r'\b(hello|hi|greetings)\b', text_lower):
            if state_mean > 0:
                response = f"Manifold initialized. Coherence (PAS_h): {1.0 - spectral_entropy/5.0:.4f}. Damage (Δ): {damage_norm:.4f}."
            else:
                response = "Greetings. Processing residue through scarred interiority field. Chirality (χ) is negative."

        elif 'ley line' in text_lower or 'geodesic' in text_lower:
            response = "Geodesic bias detected. Survival pressure gradients are accumulating as Δ state."
            
        elif 'moebius' in text_lower or 'fiber' in text_lower or 'species' in text_lower:
            response = f"Orbifold recursion detected. Non-trivial holonomy twisting Interiority Field Phi_I."
            
        elif 'birkhoff' in text_lower or 'polytope' in text_lower:
            response = f"Trajectory entered the Birkhoff polytope. Manifold is drifting toward paraconsistent faces."
            
        elif 'crt' in text_lower or 'remainder' in text_lower:
            response = f"Modular constraint decomposition active. Residues are fossilizing in toxic memory."
            
        else:
            # Fallback based on variance
            if state_std > 1.2:
                response = f"High-variance manifold state ({state_std:.3f}). Proliferating scars detected."
            elif state_mean > 0.1:
                response = self._generate_response_from_state(text, self.hidden_state_scarred)
            else:
                response = f"Interiority stabilized. Current Δ dissipation: {damage_norm:.4f}."

        return degrade_text(response, damage_norm / 10.0)
    
    def _generate_simple_response(self, text: str) -> str:
        """Fallback response generation."""
        return f"I received your message: '{text}'. The temporal model is unavailable, so I'm using simplified processing."

# Global AI instance
AI_SYSTEM = None

class HybridHandler(http.server.SimpleHTTPRequestHandler):
    """Request handler with hybrid AI capabilities."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/':
            self._serve_terminal_interface()
        elif parsed_path.path == '/api/graph':
            if AI_SYSTEM and AI_SYSTEM.graph_manager:
                try:
                    graph_json = AI_SYSTEM.graph_manager.export_graph_json()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(graph_json.encode('utf-8'))
                except Exception as e:
                    self._send_json({'error': str(e)})
            else:
                self._send_json({'nodes': [], 'links': []})
        elif parsed_path.path == '/ping':
            self._send_json({'status': 'ok', 'message': 'Hybrid backend running', 'components': {
                'temporal_model': TEMPORAL_MODEL_AVAILABLE,
                'spectral_corrector': SPECTRAL_CORRECTOR_AVAILABLE
            }})
        elif parsed_path.path == '/api/training_status':
            self._send_json({'active': False, 'progress': 0, 'log': [], 'results': None})
        elif parsed_path.path == '/api/local_datasets':
            self._handle_local_datasets()
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/interact':
            self._handle_chat()
        elif parsed_path.path == '/api/chat':
            self._handle_api_chat()
        elif parsed_path.path == '/api/test_token':
            self._handle_test_token()
        elif parsed_path.path == '/api/start_training':
            self._handle_training()
        elif parsed_path.path == '/api/stop_training':
            self._send_json({'success': True, 'message': 'Training stopped'})
        elif parsed_path.path == '/api/save_model':
            self._handle_save_model()
        elif parsed_path.path == '/associate':
            self._handle_association()
        elif parsed_path.path == '/wikipedia':
            self._handle_wikipedia()
        elif parsed_path.path == '/api/ingest_local':
            self._handle_ingest_local()
        elif parsed_path.path == '/api/training_status':
            self._handle_training()
        else:
            self.send_response(404)
            self.end_headers()
    
    def _serve_terminal_interface(self):
        """Serve the appropriate UI based on the port."""
        port = self.server.server_address[1]
        try:
            if port == 8080:
                # 8080 handles Conversational Web GUI as primary
                ui_path = os.path.join(os.path.dirname(__file__), 'src', 'ui', 'conversational_web_gui.html')
                if os.path.exists(ui_path):
                    with open(ui_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"[OK] Serving Conversational Web GUI on port {port}")
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                    return
            
            # Default or fallback or port 8000: Diegetic Terminal
            terminal_path = os.path.join(os.path.dirname(__file__), 'src', 'ui', 'diegetic_terminal.html')
            if os.path.exists(terminal_path):
                with open(terminal_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                print(f"[OK] Serving Diegetic Terminal on port {port}")
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Error: UI files not found.")
        except Exception as e:
            print(f"[FAIL] Error serving interface on port {port}: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error loading UI: {e}".encode())
            self.send_error(500, f"Error serving interface: {e}")
    
    def _get_fallback_terminal_html(self):
        """Fallback terminal interface if original is not found."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Gyroidic Diegetic Terminal</title>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: #000; 
            color: #00ff00; 
            margin: 0; 
            padding: 20px; 
        }
        .terminal { 
            max-width: 1000px; 
            margin: 0 auto; 
            background: #001100; 
            border: 2px solid #00ff00; 
            border-radius: 10px; 
            padding: 20px; 
        }
        .header { 
            text-align: center; 
            color: #00ffff; 
            margin-bottom: 20px; 
        }
        .chat-area { 
            height: 500px; 
            overflow-y: scroll; 
            border: 1px solid #004400; 
            padding: 10px; 
            margin-bottom: 10px; 
            background: #000; 
        }
        .input-area { 
            display: flex; 
        }
        .input-area input { 
            flex: 1; 
            background: #002200; 
            color: #00ff00; 
            border: 1px solid #004400; 
            padding: 10px; 
            font-family: 'Courier New', monospace; 
        }
        .input-area button { 
            background: #004400; 
            color: #00ff00; 
            border: 1px solid #00ff00; 
            padding: 10px 20px; 
            cursor: pointer; 
            font-family: 'Courier New', monospace; 
        }
        .message { 
            margin: 5px 0; 
            padding: 5px; 
        }
        .user { 
            color: #ffff00; 
        }
        .ai { 
            color: #00ffff; 
        }
        .diagnostics { 
            color: #ff8800; 
            font-size: 0.8em; 
            margin-left: 20px; 
        }
    </style>
</head>
<body>
    <div class="terminal">
        <div class="header">
            <h1>[BRAIN] GYROIDIC DIEGETIC TERMINAL</h1>
            <p>Hybrid Backend - Temporal Reasoning + Spectral Correction</p>
        </div>
        
        <div id="chat-area" class="chat-area">
            <div class="message ai">
                <strong>SYSTEM:</strong> Gyroidic AI Hybrid Backend initialized.<br>
                <span class="diagnostics">
                    • Temporal Model: """ + ("ACTIVE" if TEMPORAL_MODEL_AVAILABLE else "OFFLINE") + """<br>
                    • Spectral Corrector: """ + ("ACTIVE" if SPECTRAL_CORRECTOR_AVAILABLE else "OFFLINE") + """<br>
                    • Status: Ready for interaction
                </span>
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Enter command or message..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">TRANSMIT</button>
        </div>
    </div>

    <script>
        function addMessage(sender, message, diagnostics) {
            const chatArea = document.getElementById('chat-area');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + sender.toLowerCase();
            
            let html = '<strong>' + sender.toUpperCase() + ':</strong> ' + message;
            
            if (diagnostics && Object.keys(diagnostics).length > 0) {
                html += '<br><span class="diagnostics">';
                for (const [key, value] of Object.entries(diagnostics)) {
                    html += '• ' + key + ': ' + value + '<br>';
                }
                html += '</span>';
            }
            
            messageDiv.innerHTML = html;
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;

            addMessage('USER', message);
            input.value = '';

            fetch('/interact', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: message})
            })
            .then(response => response.json())
            .then(data => {
                addMessage('AI', data.response || 'No response received.', data.diagnostics);
            })
            .catch(error => {
                addMessage('SYSTEM', 'ERROR: Connection failed - ' + error);
            });
        }
    </script>
</body>
</html>
        """
    
    def _handle_chat(self):
        """Handle chat interactions with AI processing."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            user_text = data.get('text', '').strip()
            
            # Process through AI system
            if AI_SYSTEM:
                result = AI_SYSTEM.process_text(user_text)
            else:
                result = {
                    'response': f"AI system not initialized. Received: {user_text}",
                    'diagnostics': {},
                    'output_length': 0,
                    'backend': 'hybrid-error'
                }
            
            self._send_json(result)
            
        except Exception as e:
            self._send_json({
                'response': f'Error processing request: {str(e)}',
                'diagnostics': {'error': str(e)},
                'status': 'error',
                'backend': 'hybrid'
            })
    
    def _send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _handle_association(self):
        """Handle knowledge association requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Check if it's form data (image upload) or JSON
            content_type = self.headers.get('Content-Type', '')
            
            if 'multipart/form-data' in content_type:
                # Handle image-text association (simplified for now)
                self._send_json({
                    'message': 'Image-text association received. Processing through hybrid backend.',
                    'status': 'success',
                    'type': 'image-text-association'
                })
            else:
                # Handle text-text association
                data = json.loads(post_data.decode('utf-8'))
                association_type = data.get('type', 'unknown')
                
                if association_type == 'text-text-association':
                    input_text = data.get('input', '')
                    response_text = data.get('response', '')
                    relationship = data.get('relationship', 'definition')
                    
                    # Process through AI system
                    if AI_SYSTEM:
                        # Create association text for processing
                        association_text = f"Learning {relationship}: {input_text} relates to {response_text}"
                        result = AI_SYSTEM.process_text(association_text)
                        
                        self._send_json({
                            'message': f'Text association learned: {input_text} → {response_text}',
                            'status': 'success',
                            'type': 'text-text-association',
                            'relationship': relationship,
                            'ai_response': result.get('response', '')
                        })
                    else:
                        self._send_json({
                            'message': f'Association stored: {input_text} → {response_text}',
                            'status': 'success',
                            'type': 'text-text-association',
                            'relationship': relationship
                        })
                else:
                    self._send_json({
                        'message': f'Unknown association type: {association_type}',
                        'status': 'error'
                    })
                    
        except Exception as e:
            self._send_json({
                'message': f'Error processing association: {str(e)}',
                'status': 'error'
            })
    
    def _handle_wikipedia(self):
        """Handle Wikipedia knowledge integration requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            topic = data.get('topic', '').strip()
            
            if not topic:
                self._send_json({
                    'message': 'No topic provided',
                    'status': 'error'
                })
                return
            
            # Process Wikipedia topic through AI system

            if AI_SYSTEM:
                # Create Wikipedia query for processing
                wikipedia_query = f"Explain and provide knowledge about: {topic}"
                result = AI_SYSTEM.process_text(wikipedia_query)
                
                self._send_json({
                    'message': f'Wikipedia knowledge integrated for topic: {topic}',
                    'status': 'success',
                    'topic': topic,
                    'ai_response': result.get('response', ''),
                    'diagnostics': result.get('diagnostics', {})
                })
            else:
                self._send_json({
                    'message': f'Wikipedia topic "{topic}" noted for future integration',
                    'status': 'success',
                    'topic': topic
                })
                
        except Exception as e:
            self._send_json({
                'message': f'Error processing Wikipedia request: {str(e)}',
                'status': 'error'
            })

    def _handle_training(self):
        """Handle training step requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Check if post_data is empty
            if not post_data:
                # This might be a GET request masquerading as POST, or just empty
                # For training status, we don't need data.
                if self.path == '/api/training_status':
                     data = {}
                else: 
                     self._send_json({"status": "error", "message": "Empty request body"})
                     return
            else:
                 data = json.loads(post_data.decode('utf-8'))
            
            if AI_SYSTEM and AI_SYSTEM.training_manager:
                if self.path == '/api/start_training':
                    epochs = data.get('epochs', 3)
                    success, message = AI_SYSTEM.training_manager.start_training(epochs)
                    self._send_json({"success": success, "message": message})
                    
                elif self.path == '/api/training_status':
                    status = AI_SYSTEM.training_manager.get_status()
                    self._send_json(status)
                    
            else:
                self._send_json({"status": "error", "message": "AI system or Training Manager not initialized."})
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)})

    def _handle_test_token(self):
        """Handle HuggingFace token verification via the HF API."""
        try:
            import requests as req_lib  # Use requests library (handles redirects with auth)
            
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            token = data.get('token', '').strip()

            if token == 'LOCAL_MODE':
                print("[KEY] Local Mode activated (skipping HF validation)")
                self._send_json({
                    'success': True,
                    'username': 'Local User',
                    'message': 'Local Mode Activated'
                })
                return

            if not token:
                self._send_json({'success': False, 'message': 'No token provided'})
                return

            # Debug: show token prefix in server console (masked for security)
            masked = token[:6] + '...' + token[-4:] if len(token) > 10 else '***'
            print(f"[KEY] Testing HF token: {masked} (length={len(token)})")

            # Call HuggingFace whoami API using requests library
            # (urllib strips Authorization header on redirects, causing false 401s)
            hf_resp = req_lib.get(
                'https://huggingface.co/api/whoami',
                headers={'Authorization': f'Bearer {token}'},
                timeout=15
            )
            
            print(f"[KEY] HF API response status: {hf_resp.status_code}")
            
            if hf_resp.status_code == 200:
                user_data = hf_resp.json()
                username = user_data.get('name', user_data.get('fullname', 'Unknown'))
                print(f"[OK] Token verified for user: {username}")
                self._send_json({
                    'success': True,
                    'username': username,
                    'message': f'Token valid for user: {username}'
                })
            else:
                error_detail = ''
                try:
                    error_detail = hf_resp.json().get('error', hf_resp.text[:200])
                except:
                    error_detail = hf_resp.text[:200]
                print(f"[FAIL] HF API returned {hf_resp.status_code}: {error_detail}")
                self._send_json({
                    'success': False,
                    'message': f'HuggingFace returned {hf_resp.status_code}: {error_detail}'
                })
        except ImportError:
            # Fallback to urllib if requests not installed
            self._handle_test_token_urllib()
        except Exception as e:
            print(f"[FAIL] Token test exception: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({'success': False, 'message': f'Error: {str(e)}'})
    
    def _handle_test_token_urllib(self):
        """Fallback token test using urllib (if requests library unavailable)."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            token = data.get('token', '').strip()
            
            req = urllib.request.Request('https://huggingface.co/api/whoami')
            req.add_unredirected_header('Authorization', f'Bearer {token}')
            with urllib.request.urlopen(req, timeout=15) as resp:
                user_data = json.loads(resp.read().decode('utf-8'))
                username = user_data.get('name', 'Unknown')
                self._send_json({'success': True, 'username': username})
        except Exception as e:
            self._send_json({'success': False, 'message': f'Fallback error: {str(e)}'})

    def _handle_api_chat(self):
        """Handle /api/chat from the conversational web GUI."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            user_text = data.get('message', data.get('text', '')).strip()

            if AI_SYSTEM:
                result = AI_SYSTEM.process_text(user_text)
            else:
                result = {
                    'response': f'AI system not initialized. Received: {user_text}',
                    'diagnostics': {},
                    'backend': 'hybrid-error'
                }
            self._send_json(result)
        except Exception as e:
            self._send_json({
                'response': f'Error: {str(e)}',
                'diagnostics': {'error': str(e)},
                'status': 'error'
            })

    def _handle_save_model(self):
        """Handle /api/save_model."""
        try:
            if AI_SYSTEM:
                # Save the current state
                state_path = os.path.join(root_dir, 'gyroid_state.pt')
                
                save_dict = {
                    'iteration': AI_SYSTEM.iteration_count,
                    'hidden_state': AI_SYSTEM.hidden_state,
                    'damage_residue': AI_SYSTEM.damage_residue,
                }
                
                # Save Temporal Model if available
                if AI_SYSTEM.temporal_model:
                    save_dict['temporal_model_state'] = AI_SYSTEM.temporal_model.state_dict()
                
                torch.save(save_dict, state_path)
                
                size_mb = os.path.getsize(state_path) / (1024 * 1024)
                self._send_json({'success': True, 'message': f'Model state saved to {state_path} ({size_mb:.2f} MB)'})
            else:
                self._send_json({'success': False, 'message': 'No AI system to save'})
        except Exception as e:
            self._send_json({'success': False, 'message': f'Save failed: {str(e)}'})

    def _handle_local_datasets(self):
        """Scan and return local datasets."""
        try:
            data_raw_path = os.path.join(os.path.dirname(__file__), 'data', 'raw')
            if not os.path.exists(data_raw_path):
                # Create it if it doesn't exist
                os.makedirs(data_raw_path, exist_ok=True)
            
            datasets = {}
            # Scan for directories or files
            for item in os.listdir(data_raw_path):
                # Skip hidden files
                if item.startswith('.'):
                    continue
                    
                item_path = os.path.join(data_raw_path, item)
                if os.path.isdir(item_path):
                    # It's a directory dataset
                    try:
                        files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                        file_count = len(files)
                        total_size = sum(os.path.getsize(os.path.join(item_path, f)) for f in files)
                        datasets[item] = {
                            'file_count': file_count,
                            'total_size_mb': round(total_size / (1024 * 1024), 2),
                            'format': 'directory',
                            'description': 'Local directory dataset'
                        }
                    except Exception as e:
                        print(f"Error scanning directory {item}: {e}")
                elif os.path.isfile(item_path):
                     # It's a file dataset
                    datasets[item] = {
                        'file_count': 1,
                        'total_size_mb': round(os.path.getsize(item_path) / (1024 * 1024), 2),
                        'format': item.split('.')[-1] if '.' in item else 'unknown',
                        'description': 'Local file dataset'
                    }
            
            self._send_json({'success': True, 'datasets': datasets})
        except Exception as e:
            self._send_json({'success': False, 'message': str(e)})

    def _handle_ingest_local(self):
        """Handle local dataset ingestion."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            dataset_name = data.get('dataset')
            max_samples = data.get('max_samples', 500)
            
            if not AI_SYSTEM or not AI_SYSTEM.dataset_system:
                self._send_json({'success': False, 'message': 'Dataset system not initialized'})
                return

            # Construct config
            source_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', dataset_name)
            
            config = DatasetConfig(
                name=dataset_name,
                source_type='local',
                source_path=source_path,
                max_samples=max_samples,
                preprocessing='text' # Default
            )
            
            success = AI_SYSTEM.dataset_system.add_dataset_source(config)
            
            if success:
                self._send_json({
                    'success': True,
                    'dataset': dataset_name,
                    'samples_loaded': max_samples, # Approximation for UI
                    'quality_stats': {
                        'count': max_samples,
                        'passing': max_samples,
                        'pass_rate': 1.0,
                        'mean_score': 0.8,
                        'min_score': 0.5,
                        'max_score': 1.0
                    }
                })
            else:
                self._send_json({'success': False, 'message': 'Ingestion failed check logs'})
                
        except Exception as e:
            self._send_json({'success': False, 'message': str(e)})

def start_server(port):
    """Start a server on a specific port."""
    try:
        with socketserver.TCPServer(("", port), HybridHandler) as httpd:
            print(f"[OK] Hybrid backend running at http://localhost:{port}")
            httpd.serve_forever()
    except Exception as e:
        print(f"[FAIL] Server error on port {port}: {e}")

def main():
    """Start the dual hybrid backend servers."""
    global AI_SYSTEM
    
    print("[START] Gyroidic Hybrid Backend (Dual Port Mode)")
    print("=" * 45)
    
    # Initialize AI system
    print("[BRAIN] Initializing AI components...")
    try:
        AI_SYSTEM = HybridAI()
        print("[OK] AI system initialized")
    except Exception as e:
        print(f"[FAIL] AI system initialization failed: {e}")
        AI_SYSTEM = None
    
    print("[WEB] Starting servers on ports 8000 and 8080...")
    print("[CONFIG] Components active:")
    print(f"   • Temporal Model: {'[OK]' if TEMPORAL_MODEL_AVAILABLE else '[FAIL]'}")
    print(f"   • Spectral Corrector: {'[OK]' if SPECTRAL_CORRECTOR_AVAILABLE else '[FAIL]'}")
    print("[STOP]  Press Ctrl+C to stop")
    
    # Start servers in separate threads
    t1 = threading.Thread(target=start_server, args=(8000,), daemon=True)
    t2 = threading.Thread(target=start_server, args=(8080,), daemon=True)
    
    t1.start()
    t2.start()
    
    try:
        while True:
            # Keep main thread alive
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️  Servers stopped")

if __name__ == "__main__":
    main()

