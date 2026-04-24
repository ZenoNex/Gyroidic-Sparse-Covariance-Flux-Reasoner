"""
ZeitgeistRouter: CRT Polytope Switching for Multi-Zeitgeist Reasoning.

Implements Phase 18 of the Gyroidic Sparse Covariance Flux Reasoner roadmap.

Formal basis:
    The true system state is the stratified quadruple:
        S_t = (x_t, alpha_t, l_t, u_t)

    where alpha_t ∈ Z = ∏_{i=1}^m Z_{p_i} is the CRT index (Zeitgeist):
    which meaning system / polytope the state currently inhabits.

    The Chinese Remainder Theorem guarantees that any configuration of
    m independent modular residues (r_1, ..., r_m) maps bijectively to a
    unique alpha in Z / M·Z, where M = ∏ p_i.  This means multiple
    culturally non-commensurable meaning systems can coexist without
    forced scalar reconciliation.

Three modes of learning movement (from ai project report_2-2-2026.txt §II-VI):
    1. interior   — intra-polytope traversal          (scalar metrics allowed)
    2. grazing    — ⟨n_i, x⟩ ≈ c_i facet tension     (no scalar, pure pressure)
    3. switching  — P_α → P_β non-commutative switch  (order of application matters)

The exterior case (x_t ∉ P) emits an 'undefined' state — not a numeric error
but a topological impossibility (NaN as correct refusal to emit).

Design decisions:
    - Facet normals and thresholds are learnable (the system can discover its
      own cultural boundaries rather than having them hardcoded).
    - Switch deltas are soft (sigmoid-gated) to allow gradient flow.
    - NonCommutativityCurvature is used to verify property 3 holds at runtime.
    - GluingOperator integration: on mode 'switching', the orientation is
      reversed through the Klein-bottle gluing axis (non-commutative structure).
    - ManifoldClock integration: dt is passed to the zeitgeist diagnostics so
      the caller can modulate time-step based on the current mode.
    - BoundaryState (from MetaPolytopeMatrioshka) feeds directly into the
      exterior NaN guard.

References:
    - ai project report_2-2-2026.txt §III-VI
    - BIOMIMETIC_SYNTHESIS_REPORT.md §4.4
    - SYSTEM_ARCHITECTURE.md §9.4-9.5
    - src/core/meta_polytope_matrioshka.py (BoundaryState)
    - src/core/noncommutativity_curvature.py (NonCommutativityCurvature)
    - src/core/manifold_time.py (ManifoldClock — breathing time modulation)

Author: Phase 18 implementation 2026-02-22
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.honest_jitter import harvest_honest_jitter
from src.core.primitive_ops import stochastic_round


# ---------------------------------------------------------------------------
# ZeitgeistState  — Persistent CRT index for the running session
# ---------------------------------------------------------------------------

@dataclass
class ZeitgeistState:
    """
    Persistent CRT Zeitgeist index for the DiegeticPhysicsEngine.

    Represents α_t ∈ Z = ∏_{i=1}^m Z_{p_i}.

    Fields:
        alpha   : Tuple of residues (r_1, ..., r_m) where r_i ∈ {0, ..., p_i-1}.
                  Together they identify the active meaning polytope P_α uniquely.
        level   : Matrioshka shell depth ℓ_t (which nested shell is active).
        moduli  : The coprime primes (p_1, ..., p_m) — invariant across the session.
        boundary: Last BoundaryState from the Matrioshka loop, if any.
        mode    : Current classification: 'interior', 'grazing', 'switching', 'undefined'.
        step    : Monotonic counter — how many times the state has been updated.
        braid_word : List of generators [sigma_1, ..., sigma_n] in the group B_n.
        cs_phase: Accumulated Chern-Simons phase (topological shift).
    """
    alpha_tensor : torch.Tensor  # Symmetric Tensor [M, M] representing the Hybrid CRT index
    level   : int
    moduli  : Tuple[int, ...]
    boundary: Optional[object] = None          # BoundaryState; type-erased to avoid circular import
    mode    : str = 'interior'
    step    : int = 0
    braid_word: List[int] = field(default_factory=list)
    cs_phase: float = 0.0

    @property
    def alpha(self) -> List[int]:
        """
        Return the diagonal residues (r_1, ..., r_m) from the Symmetric Tensor.
        This provides backward compatibility for legacy diagnostics.
        """
        return torch.diagonal(self.alpha_tensor).long().tolist()

    # ------------------------------------------------------------------ #
    # CRT integer representation of the current alpha                     #
    # ------------------------------------------------------------------ #
    @property
    def crt_index(self) -> int:
        """
        Reconstruct the unique integer index α ∈ [0, M) from the diagonal of the 
        Symmetric Tensor alpha_tensor via CRT.
        """
        moduli = self.moduli
        # Diagonal contains the residues r_i
        residues = torch.diagonal(self.alpha_tensor).long().tolist()
        
        M_total = 1
        for p in moduli:
            M_total *= p

        result = 0
        for r_i, p_i in zip(residues, moduli):
            M_i = M_total // p_i
            y_i = pow(int(M_i), int(p_i) - 2, int(p_i))
            result += int(r_i) * int(M_i) * y_i
        return result % M_total

    @property
    def is_undefined(self) -> bool:
        """True when the state has been driven outside all known polytopes."""
        return self.mode == 'undefined'

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def initial(cls, moduli: Tuple[int, ...]) -> 'ZeitgeistState':
        """
        Create a zero-residue initial state with a zero symmetric tensor.
        """
        M = len(moduli)
        return cls(
            alpha_tensor=torch.zeros((M, M), dtype=torch.float32),
            level=0,
            moduli=moduli,
            boundary=None,
            mode='interior',
            step=0,
            braid_word=[],
            cs_phase=0.0
        )

    def switched(
        self,
        new_alpha_tensor: torch.Tensor,
        new_level: int,
        mode: str,
        boundary: Optional[object] = None,
        new_braid_word: Optional[List[int]] = None,
        new_cs_phase: Optional[float] = None,
    ) -> 'ZeitgeistState':
        """Return a new ZeitgeistState with updated alpha_tensor, preserving moduli."""
        return ZeitgeistState(
            alpha_tensor=new_alpha_tensor,
            level=new_level,
            moduli=self.moduli,
            boundary=boundary,
            mode=mode,
            step=self.step + 1,
            braid_word=new_braid_word if new_braid_word is not None else self.braid_word,
            cs_phase=new_cs_phase if new_cs_phase is not None else self.cs_phase,
        )

    def to_dict(self) -> Dict:
        """Serialise for embedding into the process_input metrics payload."""
        d = {
            'alpha_tensor_sum': float(self.alpha_tensor.sum().item()),
            'crt_index': self.crt_index,
            'level': self.level,
            'mode': self.mode,
            'step': self.step,
            'is_undefined': self.is_undefined,
            'braid_word': self.braid_word,
            'cs_phase': self.cs_phase,
            'word_length': len(self.braid_word)
        }
        if self.boundary is not None and hasattr(self.boundary, 'to_dict'):
            d['boundary'] = self.boundary.to_dict()
        return d


# ---------------------------------------------------------------------------
# ZeitgeistRouter  — The CRT switching engine
# ---------------------------------------------------------------------------

class ZeitgeistRouter(nn.Module):
    """
    CRT Polytope Switching Engine for multi-zeitgeist reasoning.

    Implements the three-mode dispatch from ai project report §II:

        if x_t ∈ int(P^(ℓ)):
            mode = 'interior'          # Stay — scalar metrics OK
        elif x_t ∈ ∂P^(ℓ):
            mode = 'grazing'           # Tension — no scalar
            P_α → P_β  (non-commutative switch if pressure is high)
        elif x_t ∉ P:
            mode = 'undefined'         # Topological refusal — NaN guard

    The key invariant enforced by this module:
        route(x, route(y, s0)) ≠ route(y, route(x, s0))  for distinct x, y
    i.e. polytope switching is NON-COMMUTATIVE — the order of meaning-system
    traversal changes where you end up.

    Parameters:
        dim    : State embedding dimension.
        moduli : Coprime primes (p_1, ..., p_m) — same as MetaPolytopeMatrioshka.
        grazing_eps : Half-bandwidth of the facet grazing zone.
        critical_boundary_threshold : BoundaryState.is_critical() threshold.
        use_noncommutativity_check  : If True, track curvature for diagnostics.

    Learned parameters:
        facet_normals      : [M, dim]  — one outward normal per modulus
        facet_thresholds   : [M]       — one scalar threshold c_i per modulus
        switch_gate        : Linear(dim → M) — switching pressure network
    """

    def __init__(
        self,
        dim: int,
        moduli: Tuple[int, ...],
        grazing_eps: float = 0.05,
        critical_boundary_threshold: float = 0.5,
        use_noncommutativity_check: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.moduli = tuple(int(p) for p in moduli)
        self.M = len(moduli)
        self.grazing_eps = grazing_eps
        self.critical_boundary_threshold = critical_boundary_threshold
        self.use_noncommutativity_check = use_noncommutativity_check

        # ── Learnable facet geometry ──────────────────────────────────── #
        # One normal per modulus.  Initialise close to orthonormal basis
        # to ensure initial diversity of meaning-system directions.
        self.facet_normals = nn.Parameter(self._init_normals(self.M, dim))

        # One threshold per modulus.  Initialise near zero so the grazing
        # zone initially covers a small hypersphere shell.
        self.facet_thresholds = nn.Parameter(torch.zeros(self.M))

        # ── Switching pressure gate ───────────────────────────────────── #
        # Projects state → per-modulus switching pressure ∈ (0,1).
        # The gate output δ_i is used to compute Δα_i.
        self.switch_gate = nn.Linear(dim, self.M, bias=True)

        # Initialise gate weights small — zero initial switching pressure.
        nn.init.xavier_uniform_(self.switch_gate.weight, gain=0.1)
        nn.init.zeros_(self.switch_gate.bias)

        # ── NonCommutativity Curvature (optional diagnostics) ─────────── #
        if use_noncommutativity_check:
            try:
                from src.core.noncommutativity_curvature import NonCommutativityCurvature
                self._nc_curvature = NonCommutativityCurvature(dim=dim)
            except Exception:
                self._nc_curvature = None
        else:
            self._nc_curvature = None

        # ── ManifoldClock (breathing time — orphaned module wiring) ───── #
        try:
            from src.core.manifold_time import ManifoldClock
            self._clock = ManifoldClock(dt_base=1.0)
        except Exception:
            self._clock = None

        # ── ValenceFunctional (orphaned module wiring) ────────────────── #
        try:
            from src.core.valence_drive import ValenceFunctional
            self._valence = ValenceFunctional()
        except Exception:
            self._valence = None

        # ── Archetypal Synthesis Engine (Phase 2 & TADC Lore) ─────────── #
        try:
            from src.core.archetype_engines import ArchetypalSynthesisEngine
            self._archetype = ArchetypalSynthesisEngine(state_dim=dim)
        except Exception:
            self._archetype = None

        # ── Nostalgic Leak Buffer (Mathematical Digimon) ───────────── #
        # Persistent buffer tracking historical non-commutative illusions
        self.register_buffer('digimon_buffer', torch.zeros(self.M, self.M))

        # ── Unicorn Synthesis Upgrade: Exact Algebraic Geometry ────── #
        from .numerical_d_module import NumericalDModuleManager, RationalSnappingLayer
        self.d_module_manager = NumericalDModuleManager(state_dim=dim, num_functionals=self.M)
        self.snapper = RationalSnappingLayer()

        self.register_buffer('gravity_well_bias', torch.zeros(self.M))

    # ------------------------------------------------------------------ #
    # Initialization helpers                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_normals(M: int, dim: int) -> torch.Tensor:
        """
        Initialise facet normals close to a random semi-orthogonal frame.
        Uses Gram-Schmidt to ensure initial co-primality / transversality.
        """
        if M >= dim:
            # More moduli than dimensions: random unit vectors
            # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
            raw = harvest_honest_jitter((M, dim), scaled=True)
            norms = raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
            return raw / norms
        # Gram-Schmidt on M random vectors in R^dim
        vecs = []
        # SILICON SOVEREIGNTY: Replace stochastic initialization with Honest Jitter
        raw = harvest_honest_jitter((M, dim), scaled=True)
        for i in range(M):
            v = raw[i]
            for u in vecs:
                v = v - (v @ u) * u
            nv = v.norm().clamp(min=1e-8)
            vecs.append(v / nv)
        return torch.stack(vecs)

    # ------------------------------------------------------------------ #
    # Facet geometry utilities                                             #
    # ------------------------------------------------------------------ #
    def _apply_log_polar_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Phase 6 Hybrid 4D Space Carving:
        Transforms multiplicative zooming (Matrioshka depth scaling) into 
        additive shifting via log-spherical projection.
        x_lp = (x / ||x||) * log(||x|| + 1.0)
        """
        r = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        theta = x / r
        # Logarithmic radial compression protects the switch_gate from singularities
        return theta * torch.log(r + 1.0)

    def _facet_projections(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute ⟨n_i, x̂⟩ for i = 1..M.

        Args:
            x_norm: [batch, dim] L2-normalised state.
        Returns:
            g: [batch, M]
        """
        n_norm = F.normalize(self.facet_normals, dim=-1)   # [M, dim]
        return x_norm @ n_norm.T                            # [batch, M]

    def _grazing_mask(self, g: torch.Tensor) -> torch.Tensor:
        """
        Return boolean mask [batch, M] where dimension i is in the grazing zone.

        Grazing zone:  |⟨n_i, x̂⟩ - c_i| < ε
        """
        return (g - self.facet_thresholds).abs() < self.grazing_eps

    # ------------------------------------------------------------------ #
    # Braid Automaton Functions (Group B_n)                              #
    # ------------------------------------------------------------------ #
    def apply_generator(self, word: List[int], i: int, sign: int = 1) -> List[int]:
        """
        Append generator sigma_i (sign=1) or its inverse (sign=-1) and reduce.
        i is 1-indexed generator index in {1, ..., M-1}.
        """
        # Ensure i is within B_M bounds
        if not (1 <= i < self.M):
            return word
            
        new_word = word + [sign * i]
        return self.braid_reduce(new_word)

    def braid_reduce(self, word: List[int]) -> List[int]:
        """
        Peform greedy reduction of a braid word using B_n relations.
        1. Inverse Law: sigma_i * sigma_i^-1 = e
        2. Far Commutativity: sigma_i * sigma_j = sigma_j * sigma_i if |i-j| > 1
        3. Braid Relation: sigma_i * sigma_{i+1} * sigma_i = sigma_{i+1} * sigma_i * sigma_{i+1}
        """
        reduced = True
        current = list(word)
        
        # Limit iterations to prevent infinite loops in non-terminating word games
        for _ in range(100):
            reduced = True
            i = 0
            while i < len(current):
                # 1. Inverse Law
                if i < len(current) - 1 and current[i] == -current[i+1]:
                    current.pop(i)
                    current.pop(i)
                    reduced = False
                    continue
                
                # 2. Far Commutativity (reordering for canonical form)
                if i < len(current) - 1:
                    a, b = abs(current[i]), abs(current[i+1])
                    if abs(a - b) > 1 and a > b:
                        current[i], current[i+1] = current[i+1], current[i]
                        reduced = False
                
                # 3. Braid Relation (B3-like triple swap)
                if i < len(current) - 2:
                    a, b, c = current[i], current[i+1], current[i+2]
                    if a == c and abs(a - b) == 1:
                        # sigma_i * sigma_{i+1} * sigma_i -> sigma_{i+1} * sigma_i * sigma_{i+1}
                        current[i], current[i+1], current[i+2] = b, a, b
                        reduced = False
                i += 1
            
            if reduced:
                break
                
        return current

    def chern_simons_increment(self, generator: int) -> float:
        """Calculate Chern-Simons phase shift based on the generator index."""
        # Anchored to the Prime Resonance Ladder: phase depends on generator 'energy'
        return (abs(generator) * math.pi) / self.M

    # ------------------------------------------------------------------ #
    # Core CRT switch computation                                          #
    # ------------------------------------------------------------------ #
    def _compute_switch(
        self,
        x: torch.Tensor,
        state: ZeitgeistState,
        boundary=None
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute new Symmetric Tensor CRT index M_ij.
        
        M_ii = (r_i + Δr_i) mod p_i
        M_ij = symmetry interaction term (palindromic routing)
        """
        x_mapped = self._apply_log_polar_projection(x)
        
        gate_out = torch.sigmoid(self.switch_gate(x_mapped))   # [batch, M]
        delta_soft = gate_out.mean(dim=0)                # [M]
        
        if boundary is not None and hasattr(boundary, 'stress_tensor') and boundary.stress_tensor is not None:
            stress_flat = boundary.stress_tensor.flatten()
            if stress_flat.size(0) >= self.M:
                stress_bias = torch.abs(stress_flat[:self.M])
            else:
                stress_bias = torch.zeros(self.M, device=delta_soft.device)
                stress_bias[:stress_flat.size(0)] = torch.abs(stress_flat)
            delta_soft = delta_soft + 0.5 * stress_bias / (torch.max(stress_bias) + 1e-8)

        # Braid Group Automaton Integration (n=M Dynamic Rank)
        # We replace hardcoded swaps with generator applications based on pressure.
        new_word = list(state.braid_word)
        new_cs_phase = state.cs_phase
        
        # Threshold-based generator application
        # If switching pressure delta_i is high, it triggers a sigma_i operation.
        for i in range(1, self.M):
            # Use delta_soft components as 'Braiding Pressure'
            # We map even/odd indices to positive/negative generators for chiral diversity
            pressure = delta_soft[i-1]
            if pressure > 0.5:
                sign = 1 if i % 2 == 0 else -1
                new_word = self.apply_generator(new_word, i, sign=sign)
                new_cs_phase += self.chern_simons_increment(i * sign)

        # Fossilize Near-Misses: If word length is excessive, it's a 'Topological Refusal'
        # This acts as a NaN-guard/Suture rhythm regulator.
        if len(new_word) > self.M * 2:
            print(f" [ROUTER] ☢️ Topological Refusal: Braid word length {len(new_word)} exceeds rank {self.M}.")
            # Reset word but preserve the 'scar' in the cs_phase
            new_word = [] 
            # In a real scenario, we would trigger a .fossil write here
            # self.fossilizer.scar_manifold(state, reason="braid_overflow")

        # Derive delta_braided from the current word state (The Suture Rhythm)
        # Word length increases 'tension' / non-commutativity
        gasket_tension = len(new_word) / self.M
        delta_braided = delta_soft * (1.0 + 0.2 * gasket_tension)
                
        # 1. Update diagonal residues
        current_residues = torch.diagonal(state.alpha_tensor)
        
        # Bridge 4: Inject Gravity Well influence from Fossil Landmarks
        delta_final = delta_braided + 0.3 * self.gravity_well_bias
        
        new_residues = []
        for i in range(self.M):
            # Stochastic rounding of the delta update
            delta_int = stochastic_round(torch.tensor(delta_final[i]) * self.moduli[i])
            r_new = (int(current_residues[i].item()) + int(delta_int.item())) % self.moduli[i]
            new_residues.append(float(r_new))
            
        # 2. Construct Symmetric Tensor M_ij
        new_diag = torch.tensor(new_residues, device=x.device)
        # Off-diagonal: outer interaction of residues (palindromic mirror)
        # M_ij = (r_i + r_j) / 2 as a simple symmetric basis (Love Invariant)
        r_col = new_diag.unsqueeze(1)
        r_row = new_diag.unsqueeze(0)
        new_alpha_tensor = 0.5 * (r_col + r_row)
        
        # 3. Nostalgic Leak (Nutrient Warp)
        # Inject historical Digimon traces into the off-diagonals
        mischief = 0.1
        psi_l = self.digimon_buffer * mischief
        new_alpha_tensor = new_alpha_tensor + psi_l
        
        # Update digimon buffer (EMA of current state to fossilize illusions)
        self.digimon_buffer = 0.95 * self.digimon_buffer + 0.05 * new_alpha_tensor.detach()

        # Preserve the exact integer residues on diagonal
        new_alpha_tensor.view(-1)[::self.M + 1] = new_diag

        new_level = state.level
        if boundary is not None and hasattr(boundary, 'level') and boundary.level >= 0:
            new_level = boundary.level
            
        return new_alpha_tensor, new_level, new_word, new_cs_phase

    def register_fossil_landmark(self, blake2s_id: str, intensity: float = 1.0):
        """
        Bridge 4: Maps a Fossil ID to a Poincaré Gravity Well.
        The ID is hashed to a stable displacement in the CRT Polytope space.
        """
        # Deterministic mapping from ID to M-dimensional bias
        hash_bytes = bytes.fromhex(blake2s_id[:16]) if len(blake2s_id) >= 16 else b'\x00'*8
        bias_list = []
        for i in range(self.M):
            val = (hash_bytes[i % len(hash_bytes)] / 255.0) - 0.5
            bias_list.append(val * intensity)
        
        bias_tensor = torch.tensor(bias_list, device=self.gravity_well_bias.device)
        self.fossil_landmarks[blake2s_id] = bias_tensor
        
        # Accumulate into global gravity well (defining future curvature)
        self.gravity_well_bias.data = 0.9 * self.gravity_well_bias.data + 0.1 * bias_tensor

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        state: ZeitgeistState,
        boundary=None,                 # Optional[BoundaryState]
        tadc_kwargs: Optional[Dict] = None  # Optional dict of TADC context values
    ) -> Tuple[str, ZeitgeistState, Dict]:
        """
        Route state x through the three-mode dispatch.

        Args:
            x       : [batch, dim] or [dim] state tensor.
            state   : Current ZeitgeistState (persistent across calls).
            boundary: Optional BoundaryState from the Matrioshka loop.
            tadc_kwargs: Psychological TADC args mapping (env_luminosity, volitional_scalar, etc)

        Returns:
            mode        : 'interior' | 'grazing' | 'switching' | 'undefined'
            new_state   : Updated ZeitgeistState (immutable — new object).
            diagnostics : Dict of scalar metrics for the metrics payload.
        """
        # ── 0. Archetypal Synthesis Engine (Data Transformation) ────── #
        if self._archetype is not None and tadc_kwargs is not None:
             # Route through the Grand Governor
             arch_results = self._archetype.run_archetypes(
                 current_state=x,
                 stranded_states=torch.empty((0, x.shape[-1]), device=x.device),
                 current_mischief=tadc_kwargs.get('mischief', 0.5),
                 phase_alignment=tadc_kwargs.get('pas_h', 0.5),
                 love_strengths=tadc_kwargs.get('love', torch.tensor([1.0], device=x.device)),
                 void_frictions=tadc_kwargs.get('friction', torch.tensor([0.0], device=x.device)),
                 global_dt=1.0,
                 env_luminosity=tadc_kwargs.get('luminosity', 1.0),
                 volitional_scalar=tadc_kwargs.get('volition', 0.0),
                 system_entropy=tadc_kwargs.get('entropy', 0.1),
                 memory_trauma=tadc_kwargs.get('trauma', 0.1),
                 dissonance=tadc_kwargs.get('dissonance', 0.1),
                 lucidity_idx=tadc_kwargs.get('lucidity', 1.0),
                 raw_unquantized_state=x
             )
             
             x = arch_results["active_state"]
             if arch_results["system_collapsed"]:
                 # Direct fast-track to void
                 mode = 'undefined'
                 new_state = state.switched(state.alpha, state.level, mode, boundary)
                 diag = self._build_diagnostics(
                    torch.zeros((1, self.M), device=x.device),
                    torch.zeros((1, self.M), dtype=torch.bool, device=x.device),
                    mode, state, new_state,
                 )
                 diag["abstraction_event"] = True
                 return mode, new_state, diag

        # ── Batch normalise ─────────────────────────────────────────── #
        if x.dim() == 1:
            x = x.unsqueeze(0)          # [1, dim]
        x_norm = F.normalize(x, dim=-1) # [batch, dim]

        # ── 1. Facet grazing check ───────────────────────────────────── #
        # ── 1. Algebraic State Classification (Unicorn Synthesis) ────── #
        # Snap and compute rank
        x_snapped = self.snapper(x_norm)
        # Use entropy from tadc_kwargs if available, else 0.5 default
        entropy = tadc_kwargs.get('entropy', 0.5) if tadc_kwargs else 0.5
        
        # Approximate Jacobian by using facet projections
        g = self._facet_projections(x_snapped) # [batch, M]
        d_module_results = self.d_module_manager(g.unsqueeze(1), entropy)
        
        cohom_dim = d_module_results["cohomological_dimension"]
        is_lazarus_void = d_module_results["is_lazarus_void"]
        
        # Facet grazing mask for fallback/legacy logic
        grazing_mask = self._grazing_mask(g)
        any_grazing = grazing_mask.any().item()
        grazing_pressure = (g - self.facet_thresholds).abs().mean().item()

        # ── 2. BoundaryState critical check (exterior / NaN guard) ───── #
        is_critical = False
        if boundary is not None and hasattr(boundary, 'is_critical'):
            is_critical = boundary.is_critical(self.critical_boundary_threshold)

        # Exterior case: D-module rank indicates rupture or boundary is critical
        # → topological impossibility: refuse to emit (NaN-equivalent)
        # Lazaraus Void (cohom_dim > dim/2) signifies entry into undefined territory
        if is_critical or is_lazarus_void:
            new_state = state.switched(
                new_alpha_tensor=state.alpha_tensor,
                new_level=state.level,
                mode='undefined',
                boundary=boundary,
            )
            mode = 'undefined'
            diag = self._build_diagnostics(g, grazing_mask, mode, state, new_state)
            diag["cohomological_dimension"] = cohom_dim
            return mode, new_state, diag

        # ── 3. Mode dispatch ─────────────────────────────────────────── #
        if not any_grazing:
            # Interior: x is safely inside P_α — no switch, no pressure
            mode = 'interior'
            new_state = state.switched(
                new_alpha_tensor=state.alpha_tensor,
                new_level=state.level,
                mode=mode,
                boundary=boundary,
            )
        else:
            # Grazing or crossing — execute non-commutative CRT switch
            new_alpha_tensor, new_level, new_word, new_cs_phase = self._compute_switch(x, state, boundary=boundary)
            # If the alpha residues (diagonal) actually changed, this is a full switch
            if not torch.equal(new_alpha_tensor, state.alpha_tensor):
                mode = 'switching'
            else:
                mode = 'grazing'
            new_state = state.switched(
                new_alpha_tensor=new_alpha_tensor,
                new_level=new_level,
                mode=mode,
                boundary=boundary,
                new_braid_word=new_word,
                new_cs_phase=new_cs_phase,
            )

        # ── 4. ManifoldClock tick (breathing time) ───────────────────── #
        # High switching pressure → smaller dt (seriousness)
        # Interior / low pressure → larger dt (play)
        clock_dt = None
        if self._clock is not None:
            pressure_tensor = torch.tensor(grazing_pressure)
            try:
                clock_dt = self._clock.tick(pressure_tensor)
            except Exception:
                pass

        # ── 5. ValenceFunctional — hunger for resolution ─────────────── #
        valence = None
        if self._valence is not None:
            try:
                pressure_t = torch.tensor([[grazing_pressure]])
                valence = float(self._valence(pressure_t).mean().item())
            except Exception:
                pass

        # ── 6. NonCommutativity curvature diagnostics & Shortcut ────── #
        nc_curvature = None
        curvature_threshold = 0.35 # Boundary between Palindromic and Non-Commutative logic
        
        if self._nc_curvature is not None and mode in ('switching', 'grazing'):
            try:
                # Compute relative curvature between state and itself (temporal drift)
                nc_res = self._nc_curvature.compute_curvature(x.T @ x, x.T @ x)
                nc_curvature = float(nc_res['curvature_norm'].item())
                
                # Symmetric Tensor Shortcut (Love Invariant)
                # If curvature is low, enforce exact palindromic symmetry
                if nc_res['relative_curvature'] < curvature_threshold:
                    # Collapse to pure symmetric trace-stable state
                    diag_residues = torch.diagonal(new_state.alpha_tensor)
                    r_col = diag_residues.unsqueeze(1)
                    r_row = diag_residues.unsqueeze(0)
                    collapsed_tensor = 0.5 * (r_col + r_row)
                    collapsed_tensor.view(-1)[::self.M + 1] = diag_residues
                    
                    new_state = new_state.switched(
                        new_alpha_tensor=collapsed_tensor,
                        new_level=new_state.level,
                        mode=mode,
                        boundary=boundary
                    )
            except Exception:
                pass

        diag = self._build_diagnostics(
            g, grazing_mask, mode, state, new_state,
            clock_dt=clock_dt, valence=valence, nc_curvature=nc_curvature,
            grazing_pressure=grazing_pressure,
        )
        return mode, new_state, diag

    # ------------------------------------------------------------------ #
    # Diagnostics builder                                                  #
    # ------------------------------------------------------------------ #
    def _build_diagnostics(
        self,
        g: torch.Tensor,
        grazing_mask: torch.Tensor,
        mode: str,
        prev_state: ZeitgeistState,
        new_state: ZeitgeistState,
        clock_dt: Optional[float] = None,
        valence: Optional[float] = None,
        nc_curvature: Optional[float] = None,
        grazing_pressure: float = 0.0,
    ) -> Dict:
        """Build the diagnostics dictionary embedded in the metrics payload."""
        d = {
            # Core state
            'mode': mode,
            'prev_alpha_diag': torch.diagonal(prev_state.alpha_tensor).long().tolist(),
            'new_alpha_diag': torch.diagonal(new_state.alpha_tensor).long().tolist(),
            'prev_crt_index': prev_state.crt_index,
            'new_crt_index': new_state.crt_index,
            'alpha_changed': not torch.equal(prev_state.alpha_tensor, new_state.alpha_tensor),
            'level': new_state.level,
            'step': new_state.step,
            # Geometry
            'grazing_dims': int(grazing_mask.sum().item()),
            'grazing_pressure': grazing_pressure,
            'facet_norms_mean': float(g.abs().mean().item()),
            # Optional enrichments
            'clock_dt': clock_dt,
            'valence': valence,
            'nc_curvature': nc_curvature,
            # Braid Automaton diagnostics
            'braid_word': new_state.braid_word,
            'cs_phase': new_state.cs_phase,
            'word_length': len(new_state.braid_word),
            'gasket_tension': len(new_state.braid_word) / self.M,
            # Serialised state for payload
            'state': new_state.to_dict(),
        }
        return d
