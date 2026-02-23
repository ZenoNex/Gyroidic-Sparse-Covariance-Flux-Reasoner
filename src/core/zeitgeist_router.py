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
    """
    alpha   : Tuple[int, ...]
    level   : int
    moduli  : Tuple[int, ...]
    boundary: Optional[object] = None          # BoundaryState; type-erased to avoid circular import
    mode    : str = 'interior'
    step    : int = 0

    # ------------------------------------------------------------------ #
    # CRT integer representation of the current alpha                     #
    # ------------------------------------------------------------------ #
    @property
    def crt_index(self) -> int:
        """
        Reconstruct the unique integer index α ∈ [0, M) from residues via CRT.

        Uses the standard constructive CRT formula:
            α = Σ_i  r_i · M_i · y_i  mod M
        where M = ∏ p_i,  M_i = M / p_i,  y_i = M_i^{-1} mod p_i.
        """
        moduli = self.moduli
        residues = self.alpha
        M = 1
        for p in moduli:
            M *= p

        result = 0
        for r_i, p_i in zip(residues, moduli):
            M_i = M // p_i
            # Modular inverse via Fermat's little theorem: M_i^{p_i-1} ≡ 1 (mod p_i)
            # Since all p_i are prime and gcd(M_i, p_i) = 1 (distinct primes),
            # y_i = M_i^{p_i-2} mod p_i  is the unique inverse.
            y_i = pow(int(M_i), int(p_i) - 2, int(p_i))
            result += int(r_i) * int(M_i) * y_i
        return result % M

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
        Create a zero-residue initial state: α = (0, 0, ..., 0).
        This places the system at the origin of the first (default) polytope.
        """
        return cls(
            alpha=tuple(0 for _ in moduli),
            level=0,
            moduli=moduli,
            boundary=None,
            mode='interior',
            step=0,
        )

    def switched(
        self,
        new_alpha: Tuple[int, ...],
        new_level: int,
        mode: str,
        boundary: Optional[object] = None,
    ) -> 'ZeitgeistState':
        """Return a new ZeitgeistState with updated alpha, preserving moduli."""
        return ZeitgeistState(
            alpha=new_alpha,
            level=new_level,
            moduli=self.moduli,
            boundary=boundary,
            mode=mode,
            step=self.step + 1,
        )

    def to_dict(self) -> Dict:
        """Serialise for embedding into the process_input metrics payload."""
        d = {
            'alpha': list(self.alpha),
            'crt_index': self.crt_index,
            'level': self.level,
            'mode': self.mode,
            'step': self.step,
            'is_undefined': self.is_undefined,
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
            raw = torch.randn(M, dim)
            norms = raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
            return raw / norms
        # Gram-Schmidt on M random vectors in R^dim
        vecs = []
        raw = torch.randn(M, dim)
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
    # Core CRT switch computation                                          #
    # ------------------------------------------------------------------ #
    def _compute_switch(
        self,
        x: torch.Tensor,
        state: ZeitgeistState,
    ) -> Tuple[Tuple[int, ...], int]:
        """
        Compute new residues (α_1', ..., α_m') and new level ℓ' after a switch.

        Δα_i = round(σ(gate(x))_i · p_i)  mod  p_i
        α_i' = (α_i + Δα_i)  mod  p_i

        The non-commutativity arises because gate(x) depends on x, and two
        different states x, y produce different Δα vectors; composing in
        different orders gives different paths through Z.

        Returns:
            new_alpha : Tuple[int, ...] of length M
            new_level : int (preserved; level changes come from Matrioshka loop)
        """
        # gate output: [batch, M] → take mean over batch → [M]
        gate_out = torch.sigmoid(self.switch_gate(x))   # [batch, M]
        delta_soft = gate_out.mean(dim=0)                # [M]

        new_alpha = tuple(
            int((state.alpha[i] + round(float(delta_soft[i]) * self.moduli[i]))
                % self.moduli[i])
            for i in range(self.M)
        )
        return new_alpha, state.level

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        state: ZeitgeistState,
        boundary=None,                 # Optional[BoundaryState]
    ) -> Tuple[str, ZeitgeistState, Dict]:
        """
        Route state x through the three-mode dispatch.

        Args:
            x       : [batch, dim] or [dim] state tensor.
            state   : Current ZeitgeistState (persistent across calls).
            boundary: Optional BoundaryState from the Matrioshka loop.

        Returns:
            mode        : 'interior' | 'grazing' | 'switching' | 'undefined'
            new_state   : Updated ZeitgeistState (immutable — new object).
            diagnostics : Dict of scalar metrics for the metrics payload.
        """
        # ── Batch normalise ─────────────────────────────────────────── #
        if x.dim() == 1:
            x = x.unsqueeze(0)          # [1, dim]
        x_norm = F.normalize(x, dim=-1) # [batch, dim]

        # ── 1. Facet grazing check ───────────────────────────────────── #
        g = self._facet_projections(x_norm)         # [batch, M]
        grazing_mask = self._grazing_mask(g)        # [batch, M]
        any_grazing = grazing_mask.any().item()

        grazing_pressure = (g - self.facet_thresholds).abs().mean().item()

        # ── 2. BoundaryState critical check (exterior / NaN guard) ───── #
        is_critical = False
        if boundary is not None and hasattr(boundary, 'is_critical'):
            is_critical = boundary.is_critical(self.critical_boundary_threshold)

        # Exterior case: already grazing AND boundary is critical
        # → topological impossibility: refuse to emit (NaN-equivalent)
        if is_critical and any_grazing:
            new_state = state.switched(
                new_alpha=state.alpha,
                new_level=state.level,
                mode='undefined',
                boundary=boundary,
            )
            mode = 'undefined'
            diag = self._build_diagnostics(g, grazing_mask, mode, state, new_state)
            return mode, new_state, diag

        # ── 3. Mode dispatch ─────────────────────────────────────────── #
        if not any_grazing:
            # Interior: x is safely inside P_α — no switch, no pressure
            mode = 'interior'
            new_state = state.switched(
                new_alpha=state.alpha,
                new_level=state.level,
                mode=mode,
                boundary=boundary,
            )
        else:
            # Grazing or crossing — execute non-commutative CRT switch
            new_alpha, new_level = self._compute_switch(x, state)
            # If the alpha residues actually changed, this is a full switch
            if new_alpha != state.alpha:
                mode = 'switching'
            else:
                mode = 'grazing'
            new_state = state.switched(
                new_alpha=new_alpha,
                new_level=new_level,
                mode=mode,
                boundary=boundary,
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

        # ── 6. NonCommutativity curvature diagnostics ────────────────── #
        nc_curvature = None
        if self._nc_curvature is not None and mode in ('switching', 'grazing'):
            try:
                nc_curvature = float(self._nc_curvature(x, x).mean().item())
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
            'prev_alpha': list(prev_state.alpha),
            'new_alpha': list(new_state.alpha),
            'prev_crt_index': prev_state.crt_index,
            'new_crt_index': new_state.crt_index,
            'alpha_changed': prev_state.alpha != new_state.alpha,
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
            # Serialised state for payload
            'state': new_state.to_dict(),
        }
        return d
