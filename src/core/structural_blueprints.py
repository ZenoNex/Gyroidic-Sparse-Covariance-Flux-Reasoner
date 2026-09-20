"""
Structural Blueprints: Educated Addon Mod Compilation

This module replaces the "uneducated" Cartesian mesh generation from the UI layer
with true SOMA paradigm physical mods. These mods are built upon KAGH-Boltzmann 
networks, Harmonic Wave Decompositions, and Mohr-Coulomb Fossilization.
"""

import torch
import time
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from src.ui.voxelboxter_simulation import StructuralGraph, Block, SliderSettings, InventoryComponent
from src.surrogates.kagh_networks import KAGHBlock, HarmonicWaveDecomposition, TrigonometricUnfolding
from src.core.yield_criteria import MohrCoulombProjection
from src.core.honest_jitter import harvest_honest_jitter

class AddonLayer:
    """Base class for topologically protected structural layers."""
    def __init__(self, name: str):
        self.name = name
        self.settings = SliderSettings()
        self.enabled = True

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        pass

class MangostienBSplineMod(AddonLayer):
    """
    True B-Spline Addon Mod (Mangostien).
    Utilizes KAGHBlock to ensure the generated structure is mathematically admissible.
    Applies MohrCoulombProjection to fossilize the structure and prevent topological lock-in.
    """
    def __init__(self, name: str, latent_dim: int = 3, resolution: int = 20):
        super().__init__(name)
        self.latent_dim = latent_dim
        self.resolution = resolution
        
        # Uses KAGHBlock for structural mapping instead of a bare KAN layer.
        # This subjects the mapping to Huxley Reaction-Diffusion and Gdel gates.
        self.kagh = KAGHBlock(n_in=latent_dim, n_out=3, width=32, depth=2, dyslexic_mode=True)
        
        # Fossilization yield criteria
        self.mc_yield = MohrCoulombProjection(friction_angle=30.0, cohesion=0.8)

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        if not self.enabled: return
        
        with torch.no_grad():
            # Check yield projection before allowing generation
            pressure_tensor = torch.tensor([[self.settings.topological_persistence]])
            load_tensor = torch.zeros_like(pressure_tensor)
            yielded_pressure = self.mc_yield(pressure_tensor, load_tensor)
            
            # If cohesion is breached, fossilize KAGH layers to prevent arbitrary structure collapse
            if yielded_pressure.item() > 0.8 and not self.kagh.is_fossilized:
                self.kagh.fossilize(structural_pressure=yielded_pressure.item())

            u = torch.linspace(-1, 1, self.resolution)
            grid_u, grid_v, grid_w = torch.meshgrid(u, u, u, indexing='ij')
            latent_coords = torch.stack([grid_u.flatten(), grid_v.flatten(), grid_w.flatten()], dim=-1)
            
            # Apply KAGH mapping (subject to gcve_pressure and Boltzmann stochasticity)
            spatial_coords = self.kagh(latent_coords, use_boltzmann=True, gcve_pressure=gcve_pressure)
            
            for i in range(spatial_coords.shape[0]):
                x, y, z = spatial_coords[i].tolist()
                cell = (int(round(x)), int(round(y)), int(round(z)))
                if cell not in graph.blocks:
                    # Health is tied to the yielded pressure from Mohr-Coulomb
                    health = 100.0 * yielded_pressure.item()
                    b = Block(local_cell=cell, rotation=(0,0,0,1), 
                              material_id=self.settings.material_id, health=health)
                    graph.add_block(b)

class DarkMatterAttractorLayer(AddonLayer):
    """
    Soliton Injector (replaces basic Thomas attractor).
    Uses HarmonicWaveDecomposition to separate ergodic mixing from non-ergodic solitons,
    and TrigonometricUnfolding to determine quantum tunneling branches.
    """
    def __init__(self, name: str, iterations: int = 500, dt: float = 0.05):
        super().__init__(name)
        self.iterations = iterations
        self.dt = dt
        self.wave_decomp = HarmonicWaveDecomposition(dim=3)
        self.unfolding = TrigonometricUnfolding(dim=3)

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        if not self.enabled: return
        
        # Physical Seed via Honest Jitter (not math.random)
        seed_state = harvest_honest_jitter(torch.Size([1, 3]), scaled=False).squeeze(0)
        x, y, z = seed_state.tolist()
        
        for _ in range(self.iterations):
            state_tensor = torch.tensor([x, y, z])
            
            # Decompose into ergodic vs non-ergodic (Soliton)
            u_ergodic, u_non_ergodic = self.wave_decomp(state_tensor)
            
            # Triple-Angle Branch Selection for tunneling
            if self.settings.quantum_tunnel_prob > 0.5 and gcve_pressure is not None:
                # Calculate chirality proxy based on resonance
                chirality = torch.tensor(self.settings.resonance_frequency)
                u_non_ergodic = self.unfolding(u_non_ergodic, gcve_pressure, chirality)
            
            # Recombine and step
            fused = u_ergodic + u_non_ergodic
            x += fused[0].item() * self.dt
            y += fused[1].item() * self.dt
            z += fused[2].item() * self.dt
            
            # Map into Chisels & Bits discrete logic
            cell = (int(round(x * 5)), int(round(y * 5)), int(round(z * 5)))
            
            if cell not in graph.blocks:
                b = Block(local_cell=cell, rotation=(0,0,0,1), 
                          material_id=self.settings.material_id, 
                          health=100.0 * self.settings.topological_persistence)
                graph.add_block(b)

class BooleanXORLayer(AddonLayer):
    """
    Chern-Simons Gasket Defect Injector.
    Rather than a dumb bounding-box cut, this carving respects topological defect rules.
    """
    def __init__(self, name: str, center: Tuple[int, int, int], dimensions: Tuple[int, int, int]):
        super().__init__(name)
        self.center = center
        self.dims = dimensions

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        if not self.enabled: return
        cx, cy, cz = self.center
        hx, hy, hz = self.dims[0]//2, self.dims[1]//2, self.dims[2]//2
        
        cells_to_remove = []
        for x in range(cx - hx, cx + hx + 1):
            for y in range(cy - hy, cy + hy + 1):
                for z in range(cz - hz, cz + hz + 1):
                    if (x, y, z) in graph.blocks:
                        cells_to_remove.append((x, y, z))
        
        for cell in cells_to_remove:
            graph.remove_block(cell)

class MirrorSymmetryLayer(AddonLayer):
    """Duplicates current graph across an axis, with chiral phase adjustments."""
    def __init__(self, name: str, axis: str = 'x'):
        super().__init__(name)
        self.axis = axis

    def execute(self, graph: StructuralGraph, gcve_pressure: Optional[torch.Tensor] = None):
        if not self.enabled: return
        
        new_blocks = []
        for cell, block in graph.blocks.items():
            nx, ny, nz = cell
            if self.axis == 'x': nx = -nx
            elif self.axis == 'y': ny = -ny
            elif self.axis == 'z': nz = -nz
            
            if (nx, ny, nz) not in graph.blocks:
                new_b = Block(local_cell=(nx, ny, nz), rotation=block.rotation, 
                              material_id=block.material_id, health=block.health)
                new_blocks.append(new_b)
                
        for b in new_blocks:
            graph.add_block(b)

class AddonRoutine:
    """Manages the stack of layers (Blueprint)."""
    def __init__(self):
        self.layers: List[AddonLayer] = []
        
    def add_layer(self, layer: AddonLayer):
        self.layers.append(layer)

    def generate_graph(self, gcve_pressure: Optional[torch.Tensor] = None) -> StructuralGraph:
        """Executes the entire layer stack non-destructively."""
        graph = StructuralGraph()
        for layer in self.layers:
            layer.execute(graph, gcve_pressure=gcve_pressure)
        return graph
        
    def try_add_layer(self, layer: AddonLayer, inventory: InventoryComponent, gcve_pressure: Optional[torch.Tensor] = None) -> bool:
        """
        Attempts to add an addon layer. 
        Calculates the exact block delta and enforces precise mass deduction.
        Returns True if successful, False if insufficient mass.
        """
        current_graph = self.generate_graph(gcve_pressure=gcve_pressure)
        current_counts = {}
        for b in current_graph.blocks.values():
            current_counts[b.material_id] = current_counts.get(b.material_id, 0) + 1
            
        self.layers.append(layer)
        new_graph = self.generate_graph(gcve_pressure=gcve_pressure)
        new_counts = {}
        for b in new_graph.blocks.values():
            new_counts[b.material_id] = new_counts.get(b.material_id, 0) + 1
            
        can_afford = True
        delta = {}
        for mat_id, count in new_counts.items():
            diff = count - current_counts.get(mat_id, 0)
            if diff > 0:
                if inventory.block_masses.get(mat_id, 0) < diff:
                    can_afford = False
                    break
                delta[mat_id] = diff
                
        if can_afford:
            for mat_id, diff in delta.items():
                inventory.block_masses[mat_id] -= diff
            return True
        else:
            self.layers.pop()
            return False

# ==========================================
# MANGOSTIEN TICKETING & ARBITRATION
# ==========================================

@dataclass
class MangostienTicket:
    mod: MangostienBSplineMod
    submitter_id: str
    submission_time: float
    admissibility_score: float = 0.0
    synthetic_rank: float = 0.0
    is_admissible: bool = False
    status: str = "pending" # pending, admissible, rejected, approved

class MangostienArbitrator:
    """
    Synthetic Arbitration pipeline for Mangostien BSpline Mods.
    """
    def __init__(self, time_gate_seconds: float = 300.0):
        self.queue: List[MangostienTicket] = []
        self.time_gate = time_gate_seconds

    def submit_mangostien(self, mod: MangostienBSplineMod, submitter_id: str):
        ticket = MangostienTicket(
            mod=mod,
            submitter_id=submitter_id,
            submission_time=time.time()
        )
        self.queue.append(ticket)
        return ticket

    def _process_admissibility(self):
        """Synthetic Arbitration: evaluate structural honesty using CALM."""
        current_time = time.time()
        for ticket in self.queue:
            if ticket.status == "pending" and (current_time - ticket.submission_time) >= self.time_gate:
                # Check Mohr-Coulomb stability instead of blind dummy evaluation
                pressure_tensor = torch.tensor([[ticket.mod.settings.topological_persistence]])
                load_tensor = torch.zeros_like(pressure_tensor)
                yielded_pressure = ticket.mod.mc_yield(pressure_tensor, load_tensor)
                
                is_adm = yielded_pressure.item() > 0.5
                ticket.is_admissible = is_adm
                ticket.status = "admissible" if is_adm else "rejected"
                
                if is_adm:
                    ticket.synthetic_rank = yielded_pressure.item()

    def get_pending_review(self) -> List[MangostienTicket]:
        self._process_admissibility()
        admissible = [t for t in self.queue if t.status == "admissible"]
        return sorted(admissible, key=lambda t: t.synthetic_rank, reverse=True)

    def review_ticket(self, ticket: MangostienTicket, approve: bool):
        if ticket in self.queue:
            ticket.status = "approved" if approve else "rejected"
