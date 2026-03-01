
# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

"""
Main GyroidicFluxReasoner model integrating all components.

Integrates Phase 1-3 implementations:
- Phase 1: Constraint probe operators, cyclic traversal, failure tokens
- Phase 2: Hyper-ring closure, persistence obstruction, soliton stability
- Phase 3: Structural irreducibility, gyroidic differentiation, continuous co-primality, meta-invariant

Author: William Matthew Bryant
Created: January 2026
GDPO Reference: arXiv:2601.05242
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
import numpy as np

from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.polynomial_crt import PolynomialCRT, PolynomialCRTKernelDetector
from src.core.pressure_typing import StructuralPressure
from src.core.decoupled_polynomial_crt import DecoupledPolynomialCRT
from src.models.polynomial_embeddings import PolynomialFunctionalEmbedder
from src.models.modular_attention import ModularTransformerLayer
from src.models.introspection_head import GeometricSelfModelProbe, AggregateGeometricSelfModel
from src.models.resonance_cavity import ResonanceCavity
from src.topology.gyroid_covariance import SparseGyroidCovarianceProbe, SparseExplorerRouting
from src.topology.homology_pressure import HomologyPressure, WeightedBettiNumber
from src.optimization.codes_driver import CODES
from src.optimization.operational_admm import OperationalAdmm
from src.optimization.constraint_probe import ConstraintProbeOperator, ConstraintManifold
from src.core.failure_token import FailureToken
# Phase 3: Advanced Constraints
from src.core.structural_irreducibility import StructuralIrreducibilityChecker, EvidenceModuleProjection
from src.topology.gyroid_differentiation import GyroidFlowConstraint, ForbiddenSmoothingChecker
from src.core.continuous_coprimality import ContinuousCoprimality
from src.core.meta_invariant import MetaInvariant
from src.core.orchestrator import UniversalOrchestrator
# Phase 1: Garbled Output Repair Components
from src.core.spectral_coherence_repair import SpectralCoherenceCorrector, BezoutCoefficientRefresh
from src.core.chern_simons_gasket import ChernSimonsGasket, SolitonStabilityHealer
from src.core.love_invariant_protector import LoveInvariantProtector, SoftSaturatedGates
from src.core.veto_subspace import VetoSubspace, VetoResult


class GyroidicFluxReasoner(nn.Module):
    """
    Complete Gyroidic Sparse Covariance Flux Reasoner.
    
    Integrates:
        - Modular residue embeddings
        - Multi-field attention with Birkhoff projection
        - CRT reconstruction (with optional GDPO decoupling)
        - Geometric introspection
        - Sparse gyroid covariance probes (GCVE)
        - Homology pressure on obstruction cycles
        - Resonance cavity memory
    
    Author: William Matthew Bryant
    GDPO Reference: arXiv:2601.05242
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        graph_dim: int = 256,
        num_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_functionals: int = 5,
        poly_degree: int = 4,
        poly_basis: str = 'chebyshev',
        dropout: float = 0.1,
        use_introspection: bool = True,
        use_gyroid_probes: bool = True,
        use_resonance: bool = True,
        use_gdpo: bool = True,
        learnable_weights: bool = True,
        kl_weight: float = 0.01,
        use_admm: bool = True,
        admm_rho: float = 2.0,
        admm_steps: int = 50,
        use_saturation: bool = False
    ):
        """
        Args:
            text_dim: Text embedding dimension
            graph_dim: Graph embedding dimension
            num_dim: Numerical feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            num_functionals: Number of polynomial functionals
            poly_degree: Polynomial degree
            poly_basis: Polynomial basis type ('chebyshev', 'legendre', 'hermite')
            dropout: Dropout rate
            use_introspection: Enable geometric introspection
            use_gyroid_probes: Enable gyroid covariance probes
            use_resonance: Enable resonance cavity
            use_gdpo: Enable GDPO decoupled CRT reconstruction (recommended)
            learnable_weights: Use learnable per-prime weights with GDPO
            kl_weight: KL regularization coefficient for resonance cavity
            use_admm: Enable Hybrid Physics-ADMM refinement
            admm_rho: ADMM penalty parameter
            admm_steps: Number of ADMM iterations
            use_saturation: Enable Symbolic Saturated Gates
        """
        
        # Author: William Matthew Bryant
        super().__init__()
        # --- FORCED REPAIR INJECTION ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = self.device
        poly_degree = 13 # Aligning with your gyroid_state.pt
        # -------------------------------

        # Polynomial functional configuration
        # Ensure config matches the saturation request
        self.poly_config = PolynomialCoprimeConfig(
            k=num_functionals,
            degree=poly_degree,
            basis_type=poly_basis,
            learnable=True,
            use_saturation=use_saturation
        )
        self.K = self.poly_config.k
        self.D = self.poly_config.degree + 1
        
        # Multi-modal embedder (polynomial-based)
        self.embedder = PolynomialFunctionalEmbedder(
            text_dim=text_dim,
            graph_dim=graph_dim,
            num_dim=num_dim,
            hidden_dim=hidden_dim,
            poly_config=self.poly_config,
            use_saturation=use_saturation
        )
        
        # Modular transformer layers
        self.layers = nn.ModuleList([
            ModularTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_functionals=self.K,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Polynomial CRT reconstruction (GDPO-enhanced or standard)
        self.use_gdpo = use_gdpo
        self.kl_weight = kl_weight
        
        if use_gdpo:
            self.crt = DecoupledPolynomialCRT(
                poly_config=self.poly_config,
                use_gdpo=True,
                learnable_weights=learnable_weights
            )
        else:
            self.crt = PolynomialCRT(
                poly_config=self.poly_config
            )
        
        self.crt_kernel = PolynomialCRTKernelDetector(threshold=0.5)
        
        # Optional components
        self.use_introspection = use_introspection
        self.use_gyroid_probes = use_gyroid_probes
        self.use_resonance = use_resonance
        
        if use_introspection:
            self.introspection = AggregateGeometricSelfModel(
                hidden_dim=hidden_dim,
                num_probe_dims=64,
                probe_types=['moral', 'uncertainty', 'creative', 'metacognitive']
            )
        
        if use_gyroid_probes:
            self.gyroid_probe = SparseGyroidCovarianceProbe(
                hidden_dim=hidden_dim,
                window_size=32,
                violation_threshold=0.5
            )
            self.sparse_explorer = SparseExplorerRouting(
                walk_length=8,
                num_walks=5
            )
        
        if use_resonance:
            self.resonance_cavity = ResonanceCavity(
                hidden_dim=hidden_dim,
                num_modes=64,
                poly_config=self.poly_config,
                violation_weight=0.5 # Default scaling
            )
        
        # Pressure modules
        self.homology_pressure_fn = HomologyPressure(
            cycle_weight=1.0,
            persistence_weight=2.0,
            coherence_weight=0.5
        )
        self.weighted_betti = WeightedBettiNumber()
        
        # Hybrid ADMM Components (Phase 5)
        self.use_admm = use_admm
        if use_admm:
            # Dimension of coefficient vector c: K * D
            self.coeff_dim = self.K * self.D
            
            # KAGH Surrogate (Forward Model)
            self.kagh_surrogate = KAGHBlock(
                n_in=self.coeff_dim,
                n_out=self.coeff_dim, # Treating T as same dimension for now (auto-encoder style fidelity)
                width=128,
                depth=3,
                alpha=0.7
            )
            
            # CALM Predictor (Momentum)
            self.calm_predictor = CALM(
                dim=self.coeff_dim,
                history_len=8,
                hidden_dim=256
            )
            
            # Solver Config
            self.admm_rho = admm_rho
            self.admm_steps = admm_steps
            # Note: Solver instance is created per-forward or reused if stateless
            # We keep config here
            
            # History buffer for CALM: [batch, 8, dim]
            # In a real recurrent setting, this would be stateful.
            # Here we initialize fresh or manage state externally.
            # ADMM Solver / Primitive
            # "Operationalizes ADMM as an Inherent Primitive"
            # Phase 1: Support constraint probes
            self.admm_primitive = OperationalAdmm(
                rho=admm_rho,
                max_iters=admm_steps,
                zeta=0.05, # Default drift bound
                degree=poly_degree,
                use_constraint_probes=True,  # Phase 1: Enable constraint probes
                num_constraints=self.K  # One probe per functional
            )
            
            # Phase 1: Create constraint probes (optional, can be created on-demand)
            # Each constraint probe corresponds to a polynomial functional
            self.use_constraint_probes = True
            self.constraint_probes = None  # Will be created on first use if needed
            
            # For this implementation, we simulated history buffer as property
            self.register_buffer('calm_history', torch.zeros(1, 8, self.coeff_dim)) 

        # Evolutionary Trust Scalars: [K]
        # Updated by task contribution, not normalized advantage
        self.register_buffer('trust_scalars', torch.ones(self.K))
        self.trust_freeze_threshold = 0.9 # Freeze good groups
        self.mutation_rate = 0.1

        # Selection & Containment Pressures
        from src.core.polynomial_coprime import HypergraphOrthogonalityPressure
        from src.topology.homology_pressure import ResidueHomologyDrift
        
        self.selection_pressure_fn = HypergraphOrthogonalityPressure(k_order=3)
        self.homology_drift_tracker = ResidueHomologyDrift()
        
        # Phase 3: Advanced Constraints
        self.use_structural_irreducibility = True
        self.structural_irreducibility_checker = StructuralIrreducibilityChecker()
        self.evidence_modules: Optional[List[EvidenceModuleProjection]] = None
        
        self.use_gyroid_flow_constraint = True
        self.gyroid_flow_constraint = GyroidFlowConstraint()
        self.forbidden_smoothing_checker = ForbiddenSmoothingChecker()
        
        self.use_continuous_coprimality = True
        self.continuous_coprimality = ContinuousCoprimality(use_binary_quantization=True)
        
        self.use_meta_invariant = True
        self.meta_invariant = MetaInvariant()
        
        # Containment Budget: max allowed structural pressure before System 2 repair
        self.containment_budget = 0.5
        
        # Veto Subspace Coordinator (wraps existing veto systems)
        self.veto_subspace = VetoSubspace(
            calm_threshold=0.5,
            chiral_threshold=0.3,
            containment_budget=self.containment_budget
        )
        
        # Phase 4: Universal Orchestration (Deep Dynamics)
        self.orchestrator = UniversalOrchestrator(dim=hidden_dim)

        # Phase 1: Garbled Output Repair System
        self.spectral_coherence_corrector = SpectralCoherenceCorrector(device=device)
        self.bezout_refresh = BezoutCoefficientRefresh(
            num_functionals=self.K, 
            poly_degree=poly_degree, 
            device=device
        )
        self.chern_simons_gasket = ChernSimonsGasket(device=device)
        self.soliton_healer = SolitonStabilityHealer(device=device)
        self.love_protector = LoveInvariantProtector(love_dim=hidden_dim, device=device)
        self.soft_gates = SoftSaturatedGates(
            num_functionals=self.K,
            poly_degree=poly_degree,
            device=device
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def _create_constraint_probes(self, device: torch.device):
        """
        Create constraint probe operators for each polynomial functional.
        Phase 1: Initialize constraint probes with sparse covariance matrices.
        """
        if not self.use_admm:
            return
        
        probes = []
        
        for k in range(self.K):
            # Create sparse covariance matrix for constraint k
            # Use identity with small random perturbation for anisotropy
            cov_dim = self.D  # Dimension matches polynomial degree
            sparse_cov = torch.eye(cov_dim, device=device) * 0.5
            # Add small random anisotropy
            sparse_cov += torch.randn(cov_dim, cov_dim, device=device) * 0.1
            sparse_cov = (sparse_cov + sparse_cov.t()) / 2.0  # Symmetrize
            
            # Embedding function: identity for now (can be customized)
            embedding_fn = lambda r: r.reshape(-1, self.D) if r.numel() == self.D else r
            
            # Gyroid violation function: use gyroid probe if available
            if self.use_gyroid_probes:
                def gyroid_violation_fn(c):
                    # Evaluate the functional at residue class coordinates
                    # c: [batch, D]
                    phi_eval = self.poly_config.evaluate(c)
                    # Compute violation via the manifold primitive
                    violation = self.gyroid_probe.violation_fn(phi_eval)
                    return violation
            else:
                gyroid_violation_fn = lambda c: torch.zeros(c.shape[0], device=device)
            
            probe = ConstraintProbeOperator(
                constraint_index=k,
                sparse_covariance=sparse_cov,
                embedding_fn=embedding_fn,
                gyroid_violation_fn=gyroid_violation_fn,
                device=device
            )
            probes.append(probe)
        
        self.constraint_probes = nn.ModuleList(probes)
    
    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        graph_emb: Optional[torch.Tensor] = None,
        num_features: Optional[torch.Tensor] = None,
        anchors: Optional[torch.Tensor] = None,
        group_ids: Optional[torch.Tensor] = None,
        return_analysis: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (Ecology over Algebra).
        
        Args:
            text_emb: [batch, text_dim]
            graph_emb: [batch, K_seeds]
            num_features: [batch, num_dim]
            anchors: Optional [batch] ground truth for CRT reconstruction
            group_ids: Optional [batch] group assignments for GDPO normalization
            return_analysis: If True, return detailed analysis
            
        Returns:
            Dictionary with outputs and pressures
        """
        # 1. Multi-modal embedding to residue distributions
        embed_out = self.embedder(text_emb, graph_emb, num_features)
        residue_distributions = embed_out['residue_distributions']
        h = embed_out['fused_hidden']  # [batch, hidden_dim]
        
        # Add sequence dimension for transformer (simulate seq_len=1 for now if single vector)
        if h.dim() == 2:
            h = h.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # 2. GCVE: Sparse Gyroid Probes & Dynamic Attention Sparsification
        attention_mask = None
        total_topological_pressure = torch.zeros(h.shape[0], device=h.device)
        gcve_pressures = None
        gyroid_results = {}
        
        if self.use_gyroid_probes and h.shape[1] > 1:
            # Probe topology
            # Pass the saturated functional gate for fracture detection
            gyroid_results = self.gyroid_probe(h, phi_fn=self.poly_config.evaluate)
            total_topological_pressure = gyroid_results['total_pressure']
            gcve_pressures = gyroid_results['gcve_scores'] # [batch]
            
            # Upsample violation mask to sequence length
            # (Simple approximation: if window has violation, all tokens in it constitute a hotspot)
            # For now, we perform a simple expansion or mapping
            # In a real seq2seq model, we would map window indices to token indices.
            
            # Dynamic Sparsification logic:
            # Create [batch, seq_len, seq_len] mask
            # Allow: Local Attention (always) + High Violation to All (Dense)
            # Mask: Low Violation to Long Range (Sparsified)
            
            if h.shape[1] >= 32:
                # Proper dynamic sparsification based on gyroid violation scores
                batch_size, seq_len, hidden_dim = h.shape
                
                # Compute violation scores for each position
                violation_scores = torch.zeros(batch_size, seq_len, device=h.device)
                for i in range(seq_len):
                    pos_state = h[:, i, :]  # [batch, hidden_dim]
                    violation_score = self.gyroid_probe.compute_violation_score(pos_state)
                    violation_scores[:, i] = violation_score
                
                # Create attention mask based on violations
                attention_mask = torch.ones(batch_size, seq_len, seq_len, device=h.device)
                
                # Local attention window (always allowed)
                local_window = 8
                for i in range(seq_len):
                    start_idx = max(0, i - local_window // 2)
                    end_idx = min(seq_len, i + local_window // 2 + 1)
                    attention_mask[:, i, start_idx:end_idx] = 1.0
                
                # High violation positions get dense attention
                high_violation_threshold = 0.7
                high_violation_mask = violation_scores > high_violation_threshold
                for batch_idx in range(batch_size):
                    high_viol_positions = torch.where(high_violation_mask[batch_idx])[0]
                    for pos in high_viol_positions:
                        attention_mask[batch_idx, pos, :] = 1.0  # Dense attention from high violation
                        attention_mask[batch_idx, :, pos] = 1.0  # Dense attention to high violation
                
                # Low violation positions get sparsified long-range attention
                low_violation_threshold = 0.1
                low_violation_mask = violation_scores < low_violation_threshold
                for batch_idx in range(batch_size):
                    low_viol_positions = torch.where(low_violation_mask[batch_idx])[0]
                    for pos in low_viol_positions:
                        # Sparsify long-range connections (keep only every 4th position beyond local window)
                        for target_pos in range(seq_len):
                            if abs(target_pos - pos) > local_window // 2:
                                if (target_pos - pos) % 4 != 0:  # Sparsification pattern
                                    attention_mask[batch_idx, pos, target_pos] = 0.0
                
                # Store mask for use in attention layers
                self.dynamic_attention_mask = attention_mask
            
            # ABORT RECOVERY INTEGRATION
            # "Restart from evolutionary genomic phase homologies"
            # We use the Sparse Explorer to detect deep topological failures
            if self.sparse_explorer is not None:
                # 1. Scout violations
                scout_results = self.gyroid_probe.scout_violations(h, return_indices=True)
                violation_indices = scout_results.get('violation_indices', [])
                
                if len(violation_indices) > 0:
                    violation_indices = violation_indices.squeeze(-1) # Flatten
                    
                    # 2. Run Sparse Random Walks with Backtracking/Jumping
                    explorer_results = self.sparse_explorer.detect_local_cycles(
                        h.squeeze(0), # Assume batch=1 for demo or iterate
                        violation_indices
                    )
                    
                    total_aborts = explorer_results['total_aborts']
                    total_restarts = explorer_results['total_restarts']
                    instability_detected = any(explorer_results['instability_detected'])
                    
                    # 3. Decision: Abort & Restart?
                    # If too many reciprocity aborts (non-commutative divergence)
                    # OR if instability detected -> Trigger Evolutionary Restart
                    if total_aborts > 5 or instability_detected:
                         # "Restart from evolutionary genomic phase homologies"
                         # We mutate the polynomial config (genomic phase) to shake out of the deadlock
                         with torch.no_grad():
                             self.poly_config.mutate()
                             
                             # Reduce trust in current basis (Penalize)
                             self.trust_scalars *= 0.9
                             
                             # Optionally add a "Restart Penalty" to pressure
                             total_topological_pressure += 1.0 # Force containment pressure up

        # 3a. Transformer processing (Modular multi-field)
        for layer in self.layers:
            h = layer(h, mask=attention_mask, trust_scalars=self.trust_scalars)  # [batch, seq_len, hidden_dim]
        
        # Uses saturated polynomial gates and trust-based selection
        raw_residues = self.embedder.compute_expected_residues(residue_distributions) # [batch, K, D]
        
        # PHASE 1 REPAIR: Apply garbled output fixes
        
        # 1. Spectral Coherence Correction (fix consonant clustering)
        h_corrected = self.spectral_coherence_corrector.adaptive_coherence_correction(h)
        h = h_corrected
        
        # 2. Bezout Coefficient Refresh (fix CRT modulus drift)
        residue_distributions = self.bezout_refresh.apply_crt_correction(residue_distributions)
        
        # 3. Chern-Simons Gasket (plug logic leaks)
        # Use proper polynomial co-prime functional system (anti-lobotomy)
        if not hasattr(self, 'polynomial_config'):
            from core.polynomial_coprime import PolynomialCoprimeConfig
            self.polynomial_config = PolynomialCoprimeConfig(
                k=self.K,
                degree=4,  # Default polynomial degree
                basis_type='chebyshev',
                learnable=True,
                use_saturation=True,
                device=h.device
            )
        
        poly_coeffs = self.polynomial_config.get_coefficients_tensor()  # [K, D]
        residue_distributions = self.chern_simons_gasket.plug_logic_leak(
            residue_distributions, poly_coeffs
        )
        
        # 4. Love Invariant Protection (prevent scalarization)
        h_pooled_for_love = h.squeeze(1) if h.shape[1] == 1 else h.mean(dim=1)
        love_vector, love_diagnostics = self.love_protector.apply_love_protection(h_pooled_for_love)
        
        # 5. Soft Saturated Gates (prevent binary clipping)
        # Calculate PAS_h for adaptive hardening
        pas_h_val = self.poly_config.orth_pressure_fn.entropy_estimator(
            raw_residues.view(raw_residues.shape[0], -1)
        )['ergodic_entropy'].item() if hasattr(self.poly_config, 'orth_pressure_fn') else 0.5
        
        residue_distributions = self.soft_gates.apply_soft_saturation(
            residue_distributions, pas_h_val
        )
        
        # Apply trust weights and saturation
        # We assume residue_distributions are already symbolic if configured
        # Note: We rely on the CRT internal trust-weighting now, 
        # but we still scale them here for System 1 thresholding.
        symbolic_residues = residue_distributions # [batch, K, D]
        
        # Phase 3: Check structural irreducibility before CRT
        if self.use_structural_irreducibility:
            try:
                # Create evidence modules if not exists (simplified: one per functional)
                if self.evidence_modules is None:
                    self._create_evidence_modules(h.device)
                
                # Check irreducibility
                embedding_fn = lambda r: r.reshape(-1, self.K * self.D)
                irreducibility_result = self.structural_irreducibility_checker(
                    residue=symbolic_residues,
                    embedding_fn=embedding_fn,
                    evidence_modules=self.evidence_modules
                )
                
                # If reducible, trigger System 2 or mark for repair
                if not irreducibility_result['is_irreducible'].all():
                    # Mark reducible samples for System 2 repair
                    reducible_mask = ~irreducibility_result['is_irreducible']
                    failure_mask = failure_mask | reducible_mask
            except Exception:
                # If irreducibility check fails, continue without it
                pass
        
        # 3b. Measure Selection and Containment Pressure
        reconstruction_pressure_pre = self.crt.compute_reconstruction_pressure(
            symbolic_residues, anchors, group_ids,
            trust_scalars=self.trust_scalars
        )
        
        # Hypergraph Entropy (Symbolic Orthogonality)
        expected_symbols = self.embedder.compute_expected_residues(residue_distributions) 
        symbolic_pressure = self.selection_pressure_fn(expected_symbols.mean(dim=-1))
        
        selection_pressure = reconstruction_pressure_pre.mean() + symbolic_pressure
        
        # 3b. Universal Orchestration Check
        # Orchestrate logic based on current topological pressure and PAS_h
        # Calculate real PAS_h from symbolic residues
        with torch.no_grad():
             pas_h_val = self.poly_config.orth_pressure_fn.entropy_estimator(raw_residues.view(raw_residues.shape[0], -1))['ergodic_entropy'].item()
             # Calculate real coherence across functionals
             norm_symbols = raw_residues / (torch.norm(raw_residues, dim=-1, keepdim=True) + 1e-8)
             coherence_val = torch.norm(norm_symbols.mean(dim=1), dim=-1)

        # Real Pressure Gradient: Gradient of the containment pressure w.r.t the state
        pressure_grad = torch.autograd.grad(containment_pressure_total, h, retain_graph=True)[0]
        
        h_orchestrated, regime, routing = self.orchestrator(
            state=h,
            pressure_grad=pressure_grad,
            pas_h=pas_h_val,
            coherence=coherence_val
        )
        h = h_orchestrated # Apply logical primitives
        
        # Check for symbolic failure/conflict OR budget violation
        failure_mask = (reconstruction_pressure_pre > 0.5) 
        if self.use_gyroid_probes:
            failure_mask = failure_mask | (total_topological_pressure > self.containment_budget)
        
        # If in SERIOUSNESS, we increase the failure mask sensitivity
        if regime == "SERIOUSNESS":
            failure_mask = failure_mask | (reconstruction_pressure_pre > 0.2)
            
        # 3c. SYSTEM 2: Physics-ADMM Repair (Budget-Contingent)
        # "Repair only if pressure exceeds structural budget"
        if self.use_admm and failure_mask.any():
            # Get indices of failing samples
            fail_idx = torch.where(failure_mask)[0]
            
            # Prepare ADMM inputs for failed samples
            c_init = raw_residues[fail_idx].reshape(len(fail_idx), -1)
            
            # Run ADMM repair (Status: 0: REPAIRED, 1: ALTERNATIVE, 2: FAILURE)
            refined_c_flat, repair_status = self.admm_primitive(
                initial_c=c_init,
                forward_op=self.kagh_surrogate,
                gcve_pressure=total_topological_pressure[fail_idx]
            )
            
            # Update trust scalars: survivors freeze, failures mutate
            with torch.no_grad():
                # Get mutation bias from resonance cavity (heritable trust)
                if self.use_resonance:
                    mutation_bias = self.resonance_cavity.get_mutation_bias(expected_symbols.mean(dim=2)) # [B]
                    # Average bias for global scalars
                    m_bias = mutation_bias.mean()
                else:
                    m_bias = 1.0
                    
                # Evolutionary Mutation
                self.poly_config.mutate()
                
                # Signal Sovereignty & Failure Feedback
                # If repair status is FAILURE, reduce trust significantly
                failure_penalty = (repair_status == 2).float().mean() * 0.5
                self.trust_scalars *= (1.0 - (self.mutation_rate + failure_penalty) * m_bias) 
                self.trust_scalars = torch.clamp(self.trust_scalars, 0.0, 1.0)
                
                # Explicit Failure Prop: if status == 2, zero out residues to signal fracture
                refined_dist = refined_c_flat.reshape(len(fail_idx), self.K, self.D)
                failure_mask_final = (repair_status == 2).unsqueeze(-1).unsqueeze(-1)
                refined_dist = refined_dist * (1.0 - failure_mask_final.float())
                residue_distributions[fail_idx] = refined_dist
        
        # 4. CRT reconstruction (GDPO or standard)
        if self.use_gdpo and hasattr(self.crt, 'forward_decoupled'):
            reconstruction, crt_diagnostics = self.crt.forward(
                residue_distributions,
                group_ids=group_ids,
                trust_scalars=self.trust_scalars,
                return_diagnostics=True
            )
        else:
            reconstruction = self.crt(
                residue_distributions,
                trust_scalars=self.trust_scalars
            )
            crt_diagnostics = {}
        
        reconstruction_pressure = self.crt.compute_reconstruction_pressure(
            residue_distributions, anchors, group_ids
        )
        
        # 5. Detect CRT kernel violations
        violations, pressures = self.crt_kernel.detect_violations(
            self.crt, residue_distributions, anchors
        )
        
        # Build constraint graph and find cycles
        constraint_graph = self.crt_kernel.build_constraint_graph(
            residue_distributions, violations
        )
        cycles = self.crt_kernel.find_cycles(constraint_graph)
        
        # 6. Geometric introspection (optional)
        introspection_results = {}
        introspection_coherence = None
        
        if self.use_introspection:
            h_pooled = h.squeeze(1) if h.shape[1] == 1 else h.mean(dim=1)
            introspection_results = self.introspection(h_pooled, gcve_pressure=total_topological_pressure)
            
            # Compute coherence for moral probe (example)
            if 'moral' in introspection_results:
                moral_directions = introspection_results['moral']
                introspection_coherence = self.introspection.probe_head.compute_coherence(
                    moral_directions
                )
        
        # 7. Resonance cavity update (optional, GDPO-enhanced + GCVE)
        cavity_outputs = {}
        if self.use_resonance:
            introspection_dir = introspection_results.get('moral') if self.use_introspection else None
            
            # Get expected residues for storage
            expected_residues = self.embedder.compute_expected_residues(residue_distributions)
            
            # Update cavity with residue patterns AND topological violations
            # Use refined_residues (System 2 feedback) if ADMM ran
            refined_residues = residue_distributions if self.use_admm else None
            
            # INSTABILITY AWARENESS:
            # Calculate severity from SparseExplorer results (if available)
            instability_severity = 0.0
            if 'total_aborts' in locals():
                # Normalize aborts (5 is threshold, so 5 -> 1.0)
                instability_severity = min(1.0, total_aborts / 5.0)
            
            # Veto Subspace: evaluate recovery lattice
            _abort = total_aborts if 'total_aborts' in locals() else None
            veto_result = self.veto_subspace.evaluate(
                instability_severity=instability_severity,
                covariance_aborts=_abort,
                topological_pressure=total_topological_pressure.mean().item() if hasattr(total_topological_pressure, 'mean') else float(total_topological_pressure)
            )
            # Store diagnostics for downstream consumers
            self._last_veto_result = veto_result
            
            cavity_outputs = self.resonance_cavity(
                h,
                introspection_directions=introspection_dir,
                expected_residues=expected_residues,
                gcve_pressures=gcve_pressures, # Feed GCVE scores here
                reconstruction_pressure=reconstruction_pressure,
                refined_residues=refined_residues, # System 2 Ground Truth
                instability_severity=instability_severity
            )
            memory_state = cavity_outputs['memory_state']
        
        # 8. Pressure Calculations (Replacing legacy teleological 'Loss')
        
        # CRT Consistency Pressure: Measures symbolic reconstruction fidelity
        if self.learnable_weights:
            crt_pressure = self.crt.compute_decoupled_normalization(residue_distributions, anchors)
        else:
            crt_pressure = reconstruction_pressure.mean()
            
        # Homology Pressure: Detects obstructive cycles in the residue kernel
        homology_pressure = self.homology_pressure_fn(
            cycles=cycles,
            errors=pressures,
            introspection_coherence=introspection_coherence
        )
        
        # Invariant Pressure: Maximizes functional diversity/orthogonality
        invariant_pressure = self.poly_config.orthogonality_pressure()
        
        # Gyroid Violation Pressure: Detects local manifold defects
        gyroid_pressure = mean_violation.mean()
        
        # KL Pressure: Measures drift from the Resonance Cavity's heritable trust
        kl_pressure = torch.tensor(0.0, device=h.device)
        if self.use_gdpo and self.use_resonance:
            if cavity_outputs and cavity_outputs.get('residue_prior') is not None:
                residue_prior = cavity_outputs['residue_prior']
                prior_confidence = cavity_outputs['prior_confidence']
                diff = expected_residues - residue_prior.unsqueeze(0)
                kl_pressure = (prior_confidence.unsqueeze(0) * diff ** 2).mean()
            else:
                kl_pressure = torch.norm(expected_residues - expected_residues.mean(dim=0), p=2).mean()

        # Selection Pressure (S_Symbolic): Survival of the symbolic configuration
        selection_pressure_total = crt_pressure + self.kl_weight * kl_pressure + 0.1 * invariant_pressure
        
        # Containment Pressure (C_Repair): Structural tension in the physical embedding
        h_drift, h_trigger = self.homology_drift_tracker(constraint_graph)
        containment_pressure_total = 0.1 * homology_pressure + 0.01 * gyroid_pressure + 0.5 * h_drift
        
        # NOTE: total_pressure REMOVED to prevent teleological scalarization.
        # Pressures must remain independent domain tripwires.
        
        # Phase 3: Meta-Invariant Check
        if self.use_meta_invariant:
            try:
                # Compute H_1 dimension from constraint graph
                meta_result = self.meta_invariant(
                    current_h1_dim=torch.tensor(len(cycles), dtype=torch.float32),
                    residue_distribution=expected_residues,
                    graphs=[constraint_graph] if constraint_graph else None
                )
                # Log violation but don't fail - this is a monitoring invariant
            except Exception:
                # If meta-invariant check fails, continue without it
                pass
        
        # Phase 3: Continuous Co-Primality Check (optional, for diagnostics)
        if self.use_continuous_coprimality and self.K >= 2:
            try:
                # Check co-primality between functionals (sample pairs)
                for i in range(min(2, self.K)):
                    for j in range(i + 1, min(3, self.K)):
                        residue_i = expected_residues[:, i, :]  # [batch, D]
                        residue_j = expected_residues[:, j, :]  # [batch, D]
                        coprimality_result = self.continuous_coprimality(
                            residue_i, residue_j, check_asymptotic=False
                        )
                        # Use for diagnostics or selection pressure
            except Exception:
                # If co-primality check fails, continue without it
                pass
        
        # 9. Output (Projection)
        # Use squeezed h if sequence length is 1, else mean pool or use last token
        if h.shape[1] == 1:
            h_out = h.squeeze(1)
        else:
            h_out = h.mean(dim=1)
            
        output = self.output_proj(h_out)  # [batch, 1]
        
        # PHASE 1 REPAIR: Apply soliton healing to final output
        output_text = None  # In real implementation, this would be decoded text
        healed_residues = self.soliton_healer.heal_fractured_soliton(
            residue_distributions, output_text
        )
        
        results = {
            'output': output.squeeze(-1),
            'reconstruction': reconstruction,
            'selection_pressure': StructuralPressure(selection_pressure_total, 'selection'),
            'containment_pressure': StructuralPressure(containment_pressure_total, 'containment'),
            'crt_pressure': StructuralPressure(crt_pressure, 'symbolic'),
            'homology_pressure': StructuralPressure(homology_pressure, 'topological'),
            'gyroid_pressure': StructuralPressure(gyroid_pressure, 'geometric'),
            'lambda_min': gyroid_results.get('lambda_min', torch.tensor(0.0, device=h.device)),
            'trace_c': gyroid_results.get('trace_c', torch.tensor(1.0, device=h.device)),
            'h_drift': h_drift,
            'violations': violations,
            'num_cycles': len(cycles),
            # Phase 1 Repair Diagnostics
            'spectral_diagnostics': self.spectral_coherence_corrector.get_diagnostics(),
            'chern_simons_diagnostics': self.chern_simons_gasket.get_diagnostics(),
            'soliton_healing_diagnostics': self.soliton_healer.get_diagnostics(),
            'love_diagnostics': love_diagnostics,
            'soft_gates_diagnostics': self.soft_gates.get_diagnostics()
        }
        
        if return_analysis:
            results.update({
                'residue_distributions': residue_distributions,
                'introspection': introspection_results,
                'gyroid_results': gyroid_results,
                'constraint_graph': constraint_graph,
                'cycles': cycles,
                'crt_diagnostics': crt_diagnostics
            })
        
        return results
    
    def _create_evidence_modules(self, device: torch.device):
        """
        Create evidence module projections for structural irreducibility.
        Phase 3: Initialize evidence modules (simplified: one per functional).
        """
        if not self.use_structural_irreducibility:
            return
        
        modules = []
        constraint_dim = self.K * self.D
        
        for k in range(self.K):
            # Create evidence cluster from functional k
            # Simplified: use identity cluster
            evidence_cluster = torch.eye(self.D, device=device) * 0.5
            evidence_cluster += torch.randn(self.D, self.D, device=device) * 0.1
            
            # Projection dimension: use D (polynomial degree)
            projection_dim = self.D
            
            module = EvidenceModuleProjection(
                evidence_cluster=evidence_cluster,
                projection_dim=projection_dim,
                constraint_dim=constraint_dim
            )
            modules.append(module)
        
        self.evidence_modules = nn.ModuleList(modules)
    
    def inference(
        self,
        text_emb: Optional[torch.Tensor] = None,
        graph_emb: Optional[torch.Tensor] = None,
        num_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode (no pressure computation).
        
        Returns:
            - output: Model prediction
            - reconstruction: CRT reconstruction
            - confidence: Based on violation scores
        """
        with torch.no_grad():
            results = self.forward(
                text_emb=text_emb,
                graph_emb=graph_emb,
                num_features=num_features,
                anchors=None,
                group_ids=None,
                return_analysis=False
            )
            
            # Relative confidence based on selection pressure
            confidence = torch.exp(-results['selection_pressure'].value)
            
            return {
                'output': results['output'],
                'reconstruction': results['reconstruction'],
                'confidence': confidence
            }
    
    def _create_constraint_probes(self, device: torch.device):
        """
        Create constraint probe operators for each polynomial functional.
        Phase 1: Initialize constraint probes with sparse covariance matrices.
        """
        if not self.use_admm:
            return
        
        probes = []
        
        for k in range(self.K):
            # Create sparse covariance matrix for constraint k
            # Use identity with small random perturbation for anisotropy
            cov_dim = self.D  # Dimension matches polynomial degree
            sparse_cov = torch.eye(cov_dim, device=device) * 0.5
            # Add small random anisotropy
            sparse_cov += torch.randn(cov_dim, cov_dim, device=device) * 0.1
            sparse_cov = (sparse_cov + sparse_cov.t()) / 2.0  # Symmetrize
            
            # Embedding function: identity for now (can be customized)
            embedding_fn = lambda r: r.reshape(-1, self.D) if r.numel() == self.D else r
            
            # Gyroid violation function: use gyroid probe if available
            if self.use_gyroid_probes:
                def gyroid_violation_fn(c):
                    # Proper gyroid violation computation using the gyroid probe
                    # Compute violation score based on gyroid surface distance
                    batch_size = c.shape[0]
                    violation_scores = torch.zeros(batch_size, device=c.device)
                    
                    for i in range(batch_size):
                        constraint_state = c[i]  # Single constraint state
                        # Use the gyroid probe to compute proper violation
                        violation_score = self.gyroid_probe.compute_violation_score(constraint_state.unsqueeze(0))
                        violation_scores[i] = violation_score.squeeze()
                    
                    return violation_scores
            else:
                gyroid_violation_fn = lambda c: torch.zeros(c.shape[0], device=device)
            
            probe = ConstraintProbeOperator(
                constraint_index=k,
                sparse_covariance=sparse_cov,
                embedding_fn=embedding_fn,
                gyroid_violation_fn=gyroid_violation_fn,
                device=device
            )
            probes.append(probe)
        
        self.constraint_probes = nn.ModuleList(probes)
