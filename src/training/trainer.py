
# Fix import paths
import sys
import os
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

"""
Structural Adaptation utilities for GyroidicFluxReasoner.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Callable
import numpy as np

from src.core.pressure_typing import StructuralPressure
from src.core.manifold_time import ManifoldClock
from src.core.situational_batching import SituationalBatchSampler
from src.core.daqf_operator import DAQUFOperator
from src.core.energy_monitor import StructuralEnergyMonitor
from src.core.unknowledge_flux import NostalgicLeakFunctional, EntropicMischiefProbe
from src.core.nondual_admm import NonDualProbe, UnravelingClosure
from src.core.admr_solver import PolynomialADMRSolver
from src.core.polynomial_scaffold import PolynomialCoefficientFunctional
from src.core.polynomial_coprime import PolynomialCoprimeConfig
from src.core.ley_line_tracker import LeyLineTracker
from src.core.deflagration_scout import OmipedialDeflagrator
from src.core.negentropic_manifold import NTMOperator
from src.core.valence_drive import ValenceFunctional


class ConstraintDataset(Dataset):
    """
    Dataset for constraint satisfaction problems.
    
    Each sample is a multi-modal input (text, graph, numerical)
    with a anchor value that should be CRT-reconstructible from
    valid residue assignments.
    """
    
    def __init__(
        self,
        num_samples: int,
        text_dim: int = 768,
        graph_dim: int = 256,
        num_dim: int = 64,
        max_anchor: int = 1000,
        valid_ratio: float = 0.7
    ):
        """
        Args:
            num_samples: Number of samples to generate
            text_dim: Text embedding dimension
            graph_dim: Graph embedding dimension
            num_dim: Numerical feature dimension
            max_anchor: Maximum anchor value
            valid_ratio: Ratio of valid (satisfiable) samples
        """
        self.num_samples = num_samples
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.num_dim = num_dim
        self.max_anchor = max_anchor
        self.valid_ratio = valid_ratio
        
        # Generate data
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic constraint satisfaction data."""
        data = []
        
        num_valid = int(self.num_samples * self.valid_ratio)
        num_invalid = self.num_samples - num_valid
        
        # Valid samples: coherent embeddings → valid anchor
        for i in range(num_valid):
            anchor = np.random.randint(0, self.max_anchor)
            
            # Embed anchor information in features
            text_emb = np.random.randn(self.text_dim) * 0.1
            text_emb[0] = anchor / self.max_anchor  # Encode anchor
            
            graph_emb = np.random.randn(self.graph_dim) * 0.1
            graph_emb[0] = (anchor % 100) / 100.0
            
            num_features = np.random.randn(self.num_dim) * 0.1
            num_features[0] = np.sin(anchor * 0.1)
            num_features[1] = np.cos(anchor * 0.1)
            
            data.append({
                'text_emb': text_emb.astype(np.float32),
                'graph_emb': graph_emb.astype(np.float32),
                'num_features': num_features.astype(np.float32),
                'anchor': anchor,
                'valid': True
            })
        
        # Invalid samples: incoherent embeddings → no valid reconstruction
        for i in range(num_invalid):
            # Random, uncorrelated features
            text_emb = np.random.randn(self.text_dim).astype(np.float32)
            graph_emb = np.random.randn(self.graph_dim).astype(np.float32)
            num_features = np.random.randn(self.num_dim).astype(np.float32)
            
            anchor = np.random.randint(0, self.max_anchor)
            
            data.append({
                'text_emb': text_emb,
                'graph_emb': graph_emb,
                'num_features': num_features,
                'anchor': anchor,
                'valid': False
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx].copy()
        sample['index'] = idx
        return sample


def collate_fn(batch):
    """Collate function for DataLoader."""
    text_emb = torch.stack([torch.from_numpy(x['text_emb']) for x in batch])
    graph_emb = torch.stack([torch.from_numpy(x['graph_emb']) for x in batch])
    num_features = torch.stack([torch.from_numpy(x['num_features']) for x in batch])
    anchors = torch.tensor([x['anchor'] for x in batch], dtype=torch.long)
    valid = torch.tensor([x['valid'] for x in batch], dtype=torch.bool)
    indices = [x['index'] for x in batch]
    
    return {
        'text_emb': text_emb,
        'graph_emb': graph_emb,
        'num_features': num_features,
        'anchors': anchors,
        'valid': valid,
        'indices': indices
    }


class StructuralAdaptor:
    """
    Structural Adaptor for GyroidicFluxReasoner.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = None,
        lambda_geo: float = 0.1,
        lambda_topo: float = 0.1,
        lambda_gyroid: float = 0.01,
        **kwargs
    ):
        """
        Args:
            model: GyroidicFluxReasoner model
            optimizer: PyTorch optimizer
            device: Device to adapt on
            lambda_geo: Weight for self-modeling pressure
            lambda_topo: Weight for homology pressure
            lambda_gyroid: Weight for gyroid pressure
        """
        self.model = model.to(device, non_blocking=True)
        self.optimizer = optimizer
        self.device = device
        self.lambda_geo = lambda_geo
        self.lambda_topo = lambda_topo
        self.lambda_gyroid = lambda_gyroid
        
        # Backwards compatibility: legacy_mode=True uses gradient-based adaptation
        self.legacy_mode = kwargs.get('legacy_mode', False)
        if self.legacy_mode:
            import warnings
            warnings.warn(
                "legacy_mode=True: Using gradient-based adaptation. "
                "This is deprecated. Set legacy_mode=False to use rejection-only selection.",
                DeprecationWarning
            )
        
        # Manifold Interplay
        self.clock = ManifoldClock(device=device)
        self.sampler: Optional[SituationalBatchSampler] = None
        
        # DAQUF support
        self.daquf: Optional[DAQUFOperator] = None
        self.original_L: Optional[torch.Tensor] = None
        
        # Energy & Unknowledge Monitoring
        self.energy_monitor = StructuralEnergyMonitor(device=device)
        self.mischief_probe = EntropicMischiefProbe(device=device)
        self.nostalgic_leak = NostalgicLeakFunctional(fossil_dim=64, device=device)
        self.nondual_probe = NonDualProbe(device=device)
        
        # Non-Dual State Tensor S_i = [L_i, P_i, B_i]
        self.S_i: Optional[torch.Tensor] = None
        
        # 1. Polynomial Co-Prime Config
        self.poly_config = kwargs.get('poly_config')
        if self.poly_config is None:
            self.poly_config = PolynomialCoprimeConfig(k=5, degree=4, device=device)

        # Garden Attractors dimorphisms
        self.admr = PolynomialADMRSolver(poly_config=self.poly_config, state_dim=64, device=device) 
        self.poly_scaffold = PolynomialCoefficientFunctional(device=device)
        self.ley_tracker = LeyLineTracker(num_samples=1000, device=device)
        self.deflagrator = OmipedialDeflagrator(device=device)
        self.ntm = NTMOperator(dim=64, degree=self.poly_config.degree, device=device)
        self.valence = ValenceFunctional(device=device)
    
    def adaptation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single adaptation step with REJECTION-ONLY selection.
        
        Pointer #1: Decision ≠ Neutral Transition
        - If admissible: SURVIVE (no modification, return metrics)
        - If inadmissible: REJECT (no repair attempt)
        
        This replaces gradient-based "improvement" with binary admissibility filtering.
        Evolution owns time: Only mutation + selection accumulates structure.
        """
        # SELECTION mode: No training, no gradients
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            text_emb = batch['text_emb'].to(self.device, non_blocking=True)
            graph_emb = batch['graph_emb'].to(self.device, non_blocking=True)
            num_features = batch['num_features'].to(self.device, non_blocking=True)
            anchors = batch['anchors'].to(self.device, non_blocking=True)
            
            # Forward pass (read-only evaluation)
            outputs = self.model(
                text_emb=text_emb,
                graph_emb=graph_emb,
                num_features=num_features,
                anchors=anchors
            )
            
            # ADMISSIBILITY CHECK: Binary outcome only
            if hasattr(self.model, 'validate_structural_integrity'):
                admissibility_mask = self.model.validate_structural_integrity()
                
                if not admissibility_mask.all():
                    # REJECTION: Configuration is inadmissible
                    # DO NOT attempt repair - that would inject scalar bias
                    # This is a first-class observed state, not an error
                    return {
                        "status": "REJECTED",
                        "admissible_ratio": admissibility_mask.float().mean().item(),
                        "num_cycles": outputs.get('num_cycles', 0)
                    }
            
            # SURVIVAL: Configuration is admissible
            # Return metrics for logging only - NO parameter updates
            pressure_keys = ['crt_pressure', 'homology_pressure', 'gyroid_pressure', 
                           'selection_pressure', 'containment_pressure']
            
            metrics = {
                'status': 'SURVIVED',
                'num_cycles': outputs.get('num_cycles', 0),
            }
            
            for k in pressure_keys:
                if k in outputs:
                    val = outputs[k]
                    if isinstance(val, StructuralPressure):
                        metrics[k] = val.value.item() if hasattr(val, 'value') else 0.0
                    elif isinstance(val, torch.Tensor):
                        metrics[k] = val.item()
                    else:
                        metrics[k] = float(val) if val is not None else 0.0
            
            # Legacy compatibility
            metrics['pressure'] = metrics.get('selection_pressure', 0.0)
            
            # --- MANIFOLD INTERPLAY & UNKNOWLEDGE FEEDBACK ---
            # 1. Update Energy & Mischief Monitor
            current_p = metrics.get('gyroid_pressure', 0.0)
            p_tensor = torch.tensor([current_p], device=self.device)
            
            # Detect 'Good Bugs' (Mischief) - Simplified trigger
            is_good_bug = current_p > 0.4 and current_p < 0.6
            
            # Update Mischief Probe
            # Mocking gradients/coherence for now
            self.mischief_probe.update(
                pressure_grad=torch.zeros_like(p_tensor), 
                coherence=torch.ones(1, device=self.device) * 0.8,
                pas_h=metrics.get('pas_h', 0.9),
                is_good_bug=is_good_bug
            )
            
            # 2. Finalize Energy Monitor with V_m (Mischief Violation)
            m_metrics = self.mischief_probe.get_metrics()
            v_m = p_tensor + m_metrics['H_mischief'] # GMVE approximation
            
            offending_p = v_m + self.energy_monitor.margin
            self.energy_monitor.update(v_m, alternative_pressures=offending_p.unsqueeze(0))
            
            # 3. Tick the clock based on V_m (Unknowledge Dilation)
            dt = self.clock.tick(v_m)
            
            # 4. Synchronize Temperature
            self.energy_monitor.set_temperature(self.clock.dt_ratio)
            
            # 5. Update metrics with unknowledge info
            metrics.update(self.energy_monitor.get_metrics())
            metrics.update(m_metrics)
            metrics['status'] = 'SURVIVED (Non-Dual)' if is_good_bug else 'SURVIVED'
            
            # 6. Update learning rate (breathing)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group.get('base_lr', 3e-4) * dt
                
            # 7. Update Love Invariant Sampler (formerly Pusafiliacrimonto)
            if self.sampler is not None and 'indices' in batch:
                m_scores = torch.tensor([m_metrics['H_mischief']], device=self.device).repeat(len(batch['indices']))
                self.sampler.update_love_invariant(batch['indices'], v_m.repeat(len(batch['indices'])), m_scores)

            # --- DAQUF OPERATOR ---
            if self.daquf is not None:
                # Residues for DAQUF
                gyroid_p = v_m.repeat(self.daquf.num_fossils)
                is_failed = (gyroid_p > 0.8).float() # Threshold for failure reveal
                
                # Speculative branch flux
                flux = torch.ones(self.daquf.num_fossils, 1, device=self.device) 
                
                # mischief_scores to DAQUF
                m_soliton_scores = torch.tensor([m_metrics['H_mischief']], device=self.device).repeat(self.daquf.num_fossils)
                
                # Training Valence (Hunger)
                current_valence = self.valence(v_m)
                
                daqf_results = self.daquf.apply_daquf(
                    is_failed, 
                    flux, 
                    results={
                        'energy_gaps': torch.tensor(metrics.get('energy_gap', 0.0), device=self.device).repeat(self.daquf.num_fossils),
                        'mischief_scores': m_soliton_scores,
                        'valence': current_valence.repeat(self.daquf.num_fossils)
                    }
                )
                
                # Meta-Invariant: S_i = [L, P, B]
                # We update the state tensor to reflect the unowned love vector
                self.S_i = daqf_results['love']
                
                # Check Invariants
                if self.original_L is not None:
                    self.daquf.check_invariants(self.original_L)
                
            # --- GARDEN ATTRACTORS & DEFLAGRATION FEEDBACK ---
            # 1. Update Resonance Potential & Ley Lines
            # Mocking neighbor states and adjacency for now
            # In real scenario, neighbor states come from situational batch sampling
            self.ley_tracker.update_potential(
                adjacency=torch.zeros(len(batch['indices']), len(batch['indices']), device=self.device),
                love_magnitudes=torch.norm(self.S_i[batch['indices']] if self.S_i is not None else torch.zeros(len(batch['indices']), device=self.device), dim=1)**2 if self.S_i is not None else torch.zeros(len(batch['indices']), device=self.device),
                defects=torch.zeros(len(batch['indices']), device=self.device)
            )
            
            # 2. Defect Scouting & Omipedial Jumps
            predicted_flux = v_m.repeat(len(batch['indices']))
            actual_flux = p_tensor.repeat(len(batch['indices'])) # Difference between actual and unknowledge pressure
            defects = self.deflagrator.scout_defects(predicted_flux, actual_flux)
            jumps = self.deflagrator.omipedial_jump(self.ley_tracker.V[batch['indices']])
            
            # 2.5 Update NTM Scaffold
            # dt is the manifold clock tick
            dt_tensor = torch.tensor(dt, device=self.device) if isinstance(dt, (float, int)) else dt
            self.ntm(torch.tensor(m_metrics['H_mischief'], device=self.device), dt_tensor)
            self.admr.update_scaffold(torch.tensor(m_metrics['H_mischief'], device=self.device), dt_tensor)

            # 3. ADMR Multiplicative Reconciliation with Valence Hunger
            if self.S_i is not None:
                # Mock neighbor states and weights for ADMR
                neighbor_states = self.S_i[batch['indices']].unsqueeze(1) # Identity for now
                
                # Get adjacency from relational graph (Mocking R_ik)
                r_ik = torch.ones(len(batch['indices']), 1, device=self.device)
                
                # ADMR multiplicative interaction with Valence
                self.S_i[batch['indices']] = self.admr(
                    self.S_i[batch['indices']], 
                    neighbor_states, 
                    r_ik,
                    valence=current_valence if 'current_valence' in locals() else None
                )
                
                # 4. Polynomial Shaping
                self.S_i[batch['indices']] = self.poly_scaffold(self.S_i[batch['indices']])
            
            # Update metrics with Garden info
            metrics.update(self.ley_tracker.get_metrics())
            metrics.update(self.deflagrator.get_metrics())
            metrics.update(self.admr.get_coherence_metrics(self.S_i if self.S_i is not None else torch.zeros(1, device=self.device)))
            
            # 5. Ley Line Veto (Pruning deviations)
            if jumps.sum() > 0:
                metrics['status'] = 'OMIPEDIAL JUMP'
            
            # Final ADMR metrics
            metrics['coprime_coherence'] = metrics.get('coprime_coherence', 1.0)

            return metrics
    
    def legacy_adaptation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        DEPRECATED: Legacy gradient-based adaptation step.
        
        This method is preserved for backwards compatibility. Use legacy_mode=True
        in __init__ to enable this behavior, but note it is deprecated.
        
        This violates Pointer #1 (Decision ≠ Neutral Transition) by using gradients
        to "improve" configurations instead of binary rejection.
        """
        import warnings
        warnings.warn(
            "legacy_adaptation_step is deprecated. Use rejection-only selection.",
            DeprecationWarning
        )
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        text_emb = batch['text_emb'].to(self.device, non_blocking=True)
        graph_emb = batch['graph_emb'].to(self.device, non_blocking=True)
        num_features = batch['num_features'].to(self.device, non_blocking=True)
        anchors = batch['anchors'].to(self.device, non_blocking=True)
        
        # Forward pass
        outputs = self.model(
            text_emb=text_emb,
            graph_emb=graph_emb,
            num_features=num_features,
            anchors=anchors
        )
        
        # Legacy rejection sampling
        if hasattr(self.model, 'validate_structural_integrity'):
            integrity_mask = self.model.validate_structural_integrity()
            if not integrity_mask.all():
                return {"status": "REJECTED", "num_cycles": outputs.get('num_cycles', 0)}

        # Backward pass (Legacy gradient-based)
        pressure_keys = ['crt_pressure', 'homology_pressure', 'gyroid_pressure', 
                        'selection_pressure', 'containment_pressure']
        for k in pressure_keys:
            p_obj = outputs.get(k)
            if p_obj is not None:
                if isinstance(p_obj, StructuralPressure) and p_obj.requires_grad:
                    p_obj.backward(retain_graph=True)
                elif isinstance(p_obj, torch.Tensor) and p_obj.requires_grad:
                    p_obj.backward(retain_graph=True)
        
        # Capacity Removal (Fossilization)
        if hasattr(self.model, 'get_capacity_mask'):
            capacity_mask = self.model.get_capacity_mask()
            for name, param in self.model.named_parameters():
                if 'theta' in name and param.grad is not None:
                    param.grad.data *= capacity_mask.unsqueeze(-1)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Return metrics
        metrics = {'num_cycles': outputs.get('num_cycles', 0)}
        for k in pressure_keys:
            if k in outputs:
                val = outputs[k]
                if hasattr(val, 'item'):
                    metrics[k] = val.item()
        
        metrics['pressure'] = metrics.get('selection_pressure', 0.0)
        return metrics
    
    def mutate_and_select(
        self, 
        population: list, 
        pressure_fn, 
        admissibility_thresholds: Dict[str, float] = None
    ) -> list:
        """
        Evolution owns time: Only mutation + selection, no gradient descent.
        
        Pointer #1: Selection via Pareto-inadmissibility (NOT scalar ranking).
        
        Args:
            population: List of configuration dicts
            pressure_fn: Function returning domain-isolated pressures
            admissibility_thresholds: Per-domain thresholds
        """
        if admissibility_thresholds is None:
            admissibility_thresholds = {
                'selection_pressure': 1.0,
                'containment_pressure': 1.0,
                'gyroid_pressure': 0.5
            }
        
        survivors = []
        
        for config in population:
            # Apply admissibility check
            pressures = pressure_fn(config)
            
            # Selection via Pareto-inadmissibility (NOT scalar ranking)
            is_admissible = all(
                pressures.get(domain, 0.0) < threshold
                for domain, threshold in admissibility_thresholds.items()
            )
            
            if is_admissible:
                survivors.append(config)
        
        return survivors
    
    def stability_check(self, dataloader: DataLoader) -> Dict[str, float]:
        """Check stability on validation set."""
        self.model.eval()
        
        total_pressure = 0.0
        total_crt_pressure = 0.0
        total_homology_pressure = 0.0
        total_gyroid_pressure = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                text_emb = batch['text_emb'].to(self.device, non_blocking=True)
                graph_emb = batch['graph_emb'].to(self.device, non_blocking=True)
                num_features = batch['num_features'].to(self.device, non_blocking=True)
                anchors = batch['anchors'].to(self.device, non_blocking=True)
                
                outputs = self.model(
                    text_emb=text_emb,
                    graph_emb=graph_emb,
                    num_features=num_features,
                    anchors=anchors
                )
                
                total_crt_pressure += outputs['crt_pressure'].item()
                total_homology_pressure += outputs['homology_pressure'].item()
                total_gyroid_pressure += outputs['gyroid_pressure'].item()
                total_pressure += outputs.get('selection_pressure', 0.0).item() if isinstance(outputs.get('selection_pressure'), torch.Tensor) else outputs.get('selection_pressure', 0.0)
                num_batches += 1
        
        return {
            'pressure': total_pressure / num_batches,
            'crt_pressure': total_crt_pressure / num_batches,
            'homology_pressure': total_homology_pressure / num_batches,
            'gyroid_pressure': total_gyroid_pressure / num_batches
        }
    
    def adapt(
        self,
        dataset: ConstraintDataset,
        batch_size: int = 16,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        log_interval: int = 10
    ):
        """
        Full adaptation loop with Situational Batching.
        """
        # Initialize Sampler
        self.sampler = SituationalBatchSampler(
            num_samples=len(dataset),
            batch_size=batch_size,
            device=self.device
        )
        
        # Initialize DAQUF
        self.daquf = DAQUFOperator(num_fossils=len(dataset), fossil_dim=64, device=self.device)
        self.original_L = self.daquf.L.clone()
        
        # Store base LR for breathing
        for param_group in self.optimizer.param_groups:
            if 'base_lr' not in param_group:
                param_group['base_lr'] = param_group['lr']

        train_loader = DataLoader(
            dataset,
            batch_sampler=self.sampler,
            collate_fn=collate_fn
        )
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} [Asymptotic Manifold Interplay] ===")
            
            epoch_metrics = []
            
            for batch_idx, batch in enumerate(train_loader):
                metrics = self.adaptation_step(batch)
                epoch_metrics.append(metrics)
                
                if (batch_idx + 1) % log_interval == 0:
                    state = self.clock.get_state()
                    avg_p = np.mean([m['pressure'] for m in epoch_metrics[-log_interval:]])
                    print(f"  Batch {batch_idx + 1}: p={avg_p:.4f}, dt={state['dt']:.3f}, total_t={state['t']:.2f}")
            
            # Epoch summary
            avg_metrics = {
                k: np.mean([m[k] for m in epoch_metrics])
                for k in epoch_metrics[0].keys() if isinstance(epoch_metrics[0][k], (float, int))
            }
            print(f"  End Epoch - Pressure: {avg_metrics['pressure']:.4f}, "
                  f"Avg dt: {np.mean([m.get('dt', 1.0) for m in epoch_metrics]):.3f}")
            
            # Stability Check
            if val_loader is not None:
                val_metrics = self.stability_check(val_loader)
                print(f"  Val - Pressure: {val_metrics['pressure']:.4f}")


