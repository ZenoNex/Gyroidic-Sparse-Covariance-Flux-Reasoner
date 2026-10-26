import torch
import torch.nn as nn
import math
from typing import Dict, List, Any, Optional

from src.optimization.operational_admm import OperationalAdmmPrimitive
from src.core.admr_solver import PolynomialADMRSolver
from src.core.conjugate_moment_transport import ConjugateMomentTransport
from src.core.honest_jitter import harvest_honest_jitter
from src.topology.hyper_ring_closure import HyperRingClosureChecker
from src.core.zeitgeist_router import ZeitgeistRouter
from src.core.invariants import PhaseAlignmentInvariant
from src.models.modular_attention import ModularAttention
from src.models.diegetic_heads import AutoeclecticResponderHead, DataAssociationLayer, ResonanceLarynx
from src.surrogates.calm_predictor import CALM
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.valence_drive import ValenceFunctional

class DiegeticPhysicsEngine(nn.Module):
    """
    Master Orchestration Pipeline in Voxelboxter.
    Executes the 9-Stage pipeline on every game tick with full attention-system integration.
    """
    def __init__(self, device: str = "cpu", state_dim: int = 32):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        
        # Core mathematical subsystems
        self.router = ZeitgeistRouter(dim=state_dim).to(device)
        self.admr = PolynomialADMRSolver(poly_config=None, state_dim=state_dim).to(device)
        self.optimal_transport = ConjugateMomentTransport(dim=state_dim).to(device)
        self.hyper_ring = HyperRingClosureChecker()
        self.pas = PhaseAlignmentInvariant(poly_degree=3).to(device)
        from src.core.zeitgeist_router import ZeitgeistState
        self.zeitgeist_state = ZeitgeistState.initial(self.router.moduli)

        # Attention, Diegetic Heads & Trajectory Veto Subsystems
        self.poly_config = PolynomialCoprimeConfig(k=4, degree=3, basis_type='chebyshev', learnable=False)
        attn_heads = 4 if state_dim % 4 == 0 else max([h for h in [4, 2, 1] if state_dim % h == 0], default=1)
        self.modular_attention = ModularAttention(
            hidden_dim=state_dim,
            num_heads=attn_heads,
            poly_config=self.poly_config,
            num_functionals=4,
            use_birkhoff=True
        ).to(device)
        self.calm = CALM(dim=state_dim).to(device)
        self.autoeclectic_head = AutoeclecticResponderHead(hidden_dim=state_dim, output_dim=state_dim).to(device)
        self.data_association = DataAssociationLayer(input_dim=state_dim, hidden_dim=state_dim, k=4).to(device)
        self.larynx = ResonanceLarynx(state_dim).to(device)

        # Geotechnical yield criteria & PointerlessOctree substrate
        from src.ui.voxelboxter_simulation import PointerlessOctree
        from src.core.yield_criteria import MohrCoulombProjection, DruckerPragerProjection
        self.terrain_octree = PointerlessOctree(max_depth=8)
        self.mohr_coulomb = MohrCoulombProjection(friction_angle=30.0, cohesion=0.8).to(device)
        self.drucker_prager = DruckerPragerProjection(alpha=0.2, k=1.0).to(device)
        self.valence_drive = ValenceFunctional(device=device)

        # Quadratic & Convex Local Feasibility Subsystems
        from src.core.codes_constraint_framework import CODESConstraintFramework
        from src.optimization.local_qp_operator import LocalActuationQPOperator
        self.codes_constraint = CODESConstraintFramework(state_dim=state_dim).to(device)
        self.qp_probe = LocalActuationQPOperator(constraint_index=0, actuation_dim=state_dim, device=device)

        self.initialize_spatial_metric()
        
        # Strict low-pass damping filter for spin precession / PSH telemetry
        # Read-only telemetry: ensures raw spin precession never directly modulates state update rates
        # without low-pass filtering, preventing phase velocity jitter from destabilizing frame rate or triggering false CALM aborts
        self.register_buffer("damped_spin_precession", torch.tensor(0.0, device=device))
        self.spin_damp_alpha = 0.05

    def initialize_spatial_metric(self, spatial_metric_data: Optional[Dict[str, Any]] = None):
        """
        Maps spatial metric data directly into the PointerlessOctree before the 9-stage physics loop begins.
        """
        if spatial_metric_data is None:
            for x in range(3):
                for y in range(3):
                    for z in range(2):
                        self.terrain_octree.add_adaptive_fractal_node(float(x), float(y), float(z), material=1, complexity=0.2)
        else:
            rent = float(spatial_metric_data.get("land_rent_density", 1.0))
            for x in range(3):
                for y in range(3):
                    complexity = rent * (1.0 + 0.1 * (x + y))
                    self.terrain_octree.add_adaptive_fractal_node(float(x), float(y), 0.0, material=2 if rent > 1.5 else 1, complexity=complexity)

    def process_input(self, 
                      vehicle_state: Dict[str, Any], 
                      track_state: Dict[str, Any],
                      controller_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        The 9-Stage Sovereign Physics Loop
        """
        # --- Stage 1: Control & Affordance Ingestion ---
        # Map physical controller inputs (throttle, yaw) to the semantic state vector
        # Compute PAS_h (Phase Alignment Score) to check baseline coherence.
        diegetic_affordance_ok = True
        try:
            from src.core.hardware_monitor import has_compute_affordance, ComputeAffordanceType
            diegetic_affordance_ok = bool(has_compute_affordance(ComputeAffordanceType.DIEGETIC))
        except Exception:
            pass

        input_tensor = torch.tensor([
            controller_input.get('throttle', 0.0),
            controller_input.get('yaw', 0.0),
            controller_input.get('pitch', 0.0),
            controller_input.get('roll', 0.0)
        ], device=self.device, dtype=torch.float32)
        
        # Pad or project input to state_dim
        if input_tensor.size(0) < self.state_dim:
            c_in = torch.nn.functional.pad(input_tensor, (0, self.state_dim - input_tensor.size(0)))
        else:
            c_in = input_tensor[:self.state_dim]
            
        pas_h = self.pas(c_in)

        # Draw from ModularAttention to reality-check active resonance slots and validate Birkhoff stochasticity
        c_seq = c_in.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        path_topo = controller_input.get('path_topology_vectors', None)
        attn_out = self.modular_attention(c_seq, path_topology_vectors=path_topo)
        attn_integrity = self.modular_attention.validate_structural_integrity()
        attended_c = attn_out.squeeze(0).squeeze(0)

        # Strictly Damped Spin Precession & PSH Proximity (Read-Only Telemetry)
        # Low-pass filter (tau/EMA) prevents phase velocity jitter from modulating state update rates
        # and guards against false CALM aborts
        alpha_soc = 0.15
        beta_soc = 0.15
        delta_psh = abs(alpha_soc**2 - beta_soc**2)
        ang_vel = vehicle_state.get('angular_velocity', (0.0, 0.0, 0.0))
        yaw_rate = float(ang_vel[1]) if len(ang_vel) > 1 else float(controller_input.get('yaw', 0.0))
        raw_precession = abs(yaw_rate) * math.sqrt(alpha_soc**2 + beta_soc**2)
        with torch.no_grad():
            self.damped_spin_precession = (1.0 - self.spin_damp_alpha) * self.damped_spin_precession + self.spin_damp_alpha * raw_precession

        # --- Stage 2: Non-Commutative Braid Routing ---
        # The router treats steering before braking differently from braking before steering.
        mode, self.zeitgeist_state, router_metric, routed_c = self.router(c_in, self.zeitgeist_state)
        if 'pas_h' not in router_metric:
            router_metric['pas_h'] = pas_h

        # --- Stage 3: System 1 Symbolic Trajectory Draft ---
        # Generate c_sym using coprime polynomials. In Voxelboxter, this is the speculative "ghost" trajectory
        # before checking terrain collisions.
        # [REHYBRIDIZATION]: Use CODES driver for proper topological chordlock projection
        # rather than the shallow simplified sin(x * pi) draft.
        if not hasattr(self, 'codes_driver'):
            from src.optimization.codes_driver import CODES
            self.codes_driver = CODES(state_dim=self.state_dim, constraint_depth=3).to(self.device)
            
        c_sym = self.codes_driver.project_chordlock(routed_c)

        # --- Stage 4: Gyroid Violation Probes ---
        # Evaluate local violation V to dictate sparsification vs dense compute.
        # If the speculative trajectory intersects a voxel, V spikes.
        # Track and terrain Betti topology (islands beta_0, tunnels/chasm loops beta_1) scales obstacle complexity.
        b0 = track_state.get('betti_0', 1)
        b1 = track_state.get('betti_1', 0)
        topo_complexity = 1.0 + 0.1 * max(0, b0 - 1) + 0.25 * b1
        violation_v = torch.norm(c_sym) * pas_h.mean() * topo_complexity

        # --- Stage 4.1: Valence Drive (Structural Dissonance & Metaphysical Entropy) ---
        mischief_val = track_state.get('mischief', None)
        if mischief_val is not None and not isinstance(mischief_val, torch.Tensor):
            mischief_val = torch.tensor(mischief_val, dtype=torch.float32, device=self.device)
        entropy_val = track_state.get('entropy', None)
        if entropy_val is not None and not isinstance(entropy_val, torch.Tensor):
            entropy_val = torch.tensor(entropy_val, dtype=torch.float32, device=self.device)
        categories_val = track_state.get('categories', None)
        if categories_val is not None and not isinstance(categories_val, torch.Tensor):
            categories_val = torch.tensor(categories_val, dtype=torch.float32, device=self.device)
            
        valence_hunger = self.valence_drive(
            current_pressure=violation_v.view(1),
            mischief=mischief_val,
            entropy=entropy_val,
            categories=categories_val
        )

        # --- Stage 5: System 2 ADMM & ADMR Constraint Probes ---
        # Check constraints (tire slip, structural breakage) modulated by valence hunger.
        admr_output = self.admr(c_sym, valence=valence_hunger)

        # CALM Trajectory Veto Gating (TOPOLOGICAL_EXTENSIONS.md Part VI §7)
        history_c = c_in.unsqueeze(0).repeat(1, 8, 1)  # [1, 8, state_dim]
        calm_abort, calm_rho, calm_step, calm_forcing, calm_gauge, calm_constraints = self.calm(history_c)
        geom_veto = float(violation_v.item() > 2.0)
        calm_veto = float(calm_abort.mean().item())
        gauge_val = float(torch.clamp(calm_gauge.mean(), 0.0, 1.0).item())
        total_veto = (1.0 - gauge_val) * geom_veto + gauge_val * calm_veto
        
        # --- Stage 5.1: Carnot-Möbius Thermodynamic Ledger ---
        # Compute thermal runaway based on ADMR topological friction.
        if not hasattr(self, 'carnot_ledger'):
            from src.core.carnot_mobius_ledger import CarnotMobiusLedger
            self.carnot_ledger = CarnotMobiusLedger().to(self.device)
            
        # Mocking Lambda_vac and Lambda_pump from the input constraints for now
        lambda_vac = torch.tensor([0.1], device=self.device)
        lambda_pump = torch.norm(c_in) + 0.5 
        
        ledger = self.carnot_ledger(
            lambda_vac=lambda_vac, 
            lambda_pump=lambda_pump, 
            admr_residues=admr_output, 
            depth_N=10
        )
        
        # Stage 5: Local Feasibility Probing & Quadratic Well Relaxation
        def _feasibility_forward_op(state_tensor, *args, **kwargs):
            if hasattr(self, 'codes_constraint') and self.codes_constraint is not None:
                relaxed_state, _ = self.codes_constraint.evolve_state(state_tensor, num_steps=2)
                return relaxed_state
            return state_tensor
        
        probes = [self.qp_probe] if hasattr(self, 'qp_probe') and self.qp_probe is not None else None
        admm_state, admm_status = OperationalAdmmPrimitive.apply(
            c_sym, _feasibility_forward_op, 
            0.1, 0.01, 10, 0.5, 3, 
            None, None, True, probes, len(probes) if probes else 1, None
        )

        # --- Stage 6: SCCCG Recovery & Fossilization ---
        # Inject Mohr-Coulomb / Drucker-Prager yield criterion evaluation on stress tensor
        stress_tensor = admm_state.view(-1, 1) @ admm_state.view(1, -1) if admm_state.dim() == 1 else admm_state
        normal_load = torch.norm(c_in) + 1.0
        
        mc_yield = self.mohr_coulomb(stress_tensor[:2, :2] if stress_tensor.shape[0] >= 2 else stress_tensor, normal_load)
        dp_envelope = self.drucker_prager(stress_tensor[:3, :3] if stress_tensor.shape[0] >= 3 else stress_tensor)
        yield_exceeded = bool((mc_yield > 0.8).any().item()) or bool((dp_envelope > 1.2).any().item())

        strain_limit = 5.0
        thermal_runaway = ledger.get('thermal_runaway', False)
        is_thermal_runaway = thermal_runaway.item() if isinstance(thermal_runaway, torch.Tensor) else bool(thermal_runaway)
        is_failure = (admm_status.item() == 2) if hasattr(admm_status, 'item') else (admm_status == 2)
        is_rupture = (torch.norm(admm_state) > strain_limit) or is_thermal_runaway or is_failure or yield_exceeded
        
        sheared_voxels = 0
        if is_rupture:
            # Physical shear of voxels in the PointerlessOctree upon exceeding yield limit
            sheared_voxels = self.terrain_octree.shear_voxels(stress_tensor, yield_limit=1.0)
            
            # Fracture recovery
            noise = harvest_honest_jitter(admm_state.shape, device=self.device, scaled=True)
            recovered_state = self.optimal_transport.nabla_psi_star(admm_state + noise)
        else:
            recovered_state = admm_state

        # --- Stage 7: Decoupled Polynomial CRT Reconstruction ---
        # Reconstruct continuous coordinates from the modular residues.
        x_hat = recovered_state * router_metric['pas_h'].mean()

        # --- Stage 8: Hyper-Ring Cycle Closure ---
        # Relational momentum integrals to classify soliton vs collapse.
        closure_gap = self.hyper_ring.compute_holonomy(x_hat, admr_output.mean(dim=0, keepdim=True) if admr_output.dim() > 1 else admr_output)
        
        # --- Stage 9: Track Deformation & Audience Projection ---
        # Push changes to Betti numbers (beta_0, beta_1).
        # In Voxelboxter, a high closure gap in a rupture state or yield exceedance
        # shears voxels in PointerlessOctree and increments Betti number (Delta beta_1 = 1).
        octree_betti = self.terrain_octree.compute_betti_numbers()
        betti_shift = 0
        gap_val = closure_gap.item() if hasattr(closure_gap, 'item') else float(closure_gap)
        new_b0 = octree_betti.get("betti_0", b0)
        new_b1 = max(b1, octree_betti.get("betti_1", b1))
        if is_rupture and (gap_val > 0.5 or yield_exceeded):
            betti_shift = 1 # We carved a new hole (Delta beta_1 = 1) in the terrain
            new_b1 = max(new_b1, b1 + 1)

        # Diegetic Autoeclectic Expression
        spectral_entropy = torch.tensor([[float(torch.clamp(1.0 - pas_h.mean(), 0.0, 1.0).item())]], device=self.device)
        diegetic_out = self.autoeclectic_head(x_hat.unsqueeze(0), spectral_entropy=spectral_entropy)

        # Return updated physics state
        return {
            "c_out": x_hat,
            "betti_shift": betti_shift,
            "betti_0": new_b0,
            "betti_1": new_b1,
            "pas_h": pas_h.detach(),
            "rupture": bool(is_rupture),
            "closure_gap": gap_val,
            "damped_spin_precession": float(self.damped_spin_precession.item()),
            "delta_psh": float(delta_psh),
            "psh_invariant_active": bool(delta_psh < 1e-4),
            "attention_integrity": bool(attn_integrity.all().item()),
            "diegetic_response": diegetic_out.squeeze(0).detach(),
            "calm_total_veto": total_veto,
            "calm_gauge": gauge_val,
            "calm_constraints": calm_constraints.squeeze(0).detach(),
            "diegetic_affordance_ok": diegetic_affordance_ok,
            "valence_hunger": float(valence_hunger.mean().item()),
            "valence_metrics": self.valence_drive.get_metrics()
        }

    def step(self, track_state: Dict[str, Any], controller_input: Optional[Dict[str, Any]] = None, vehicle_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Step method alias for the 9-stage physics loop.
        """
        v_state = vehicle_state or {}
        c_input = controller_input if isinstance(controller_input, dict) else {'throttle': 0.5, 'yaw': 0.0}
        return self.process_input(vehicle_state=v_state, track_state=track_state, controller_input=c_input)

    def ingest_knowledge_dyad(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        """
        Fuses multi-modal inputs (e.g., Image semantics + Textual descriptions)
        via DataAssociationLayer into unified coprime polynomial residues.
        """
        return self.data_association(text_emb, image_emb)

    def evaluate_attention_diagnostics(self) -> Dict[str, Any]:
        """
        Returns live attention health, Birkhoff doubly-stochastic integrity, and CALM veto states.
        """
        return {
            "birkhoff_integrity": bool(self.modular_attention.validate_structural_integrity().all().item()),
            "warmstart_mix": float(self.modular_attention.warmstart_mix.item()),
            "num_breathers": self.modular_attention.num_breathers,
            "active_functionals": self.modular_attention.K
        }
