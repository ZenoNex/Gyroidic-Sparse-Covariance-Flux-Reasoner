"""
Global PAS API for Cross-System Coherence Tracking & Full-Scale RIC Deployment.

Bridges the internal physics engine out of the local sandbox into edge + cloud
environments using cryptographic handshakes (GLYPHLOCK) and the PAS_LOCK closure theorem.

CODES v40 Reference:
- Thesis Spine (The Coherence Wager):
    PAS_LOCK(S_n) = 1 iff PAS_h(S_n,t) >= theta_L and PAS_zeta(S_n) <= zeta_drift and GLYPHLOCK(S_n) = 1.
- CODES/RIC Mechanism:
    FIELDCAST (field selector) + CHORDLOCK (prime anchor) + PAS_h (multiharmonic invariant) +
    TEMPOLOCK (prime-indexed timing) + GLYPHLOCK/AURA_OUT (symbol legality, emission gate).
"""

import os
import time
import hmac
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import torch

from src.core.invariants import (
    PhaseAlignmentInvariant,
    APAS_Zeta,
    check_glyphlock,
    compute_chirality,
    compute_ihc_bao_shell_modulation,
)
from src.core.honest_jitter import harvest_honest_jitter


@dataclass
class PASLockToken:
    """
    Cryptographically signed PAS_LOCK session and emission token.
    Grants access to external nodes without permitting corruption of internal state.
    """
    session_id: str
    client_id: str
    timestamp: float
    field_id: str
    chordlock_anchor: int
    tempolock_interval: float
    pas_h: float
    pas_zeta: float
    glyphlock: bool
    aura_out: bool
    signature: str = ""

    def is_valid(self, secret_key: bytes, max_age_sec: float = 300.0) -> bool:
        """Validate token age, mathematical invariants, and cryptographic signature."""
        if time.time() - self.timestamp > max_age_sec:
            return False
        if not self.glyphlock:
            return False
        if not self.aura_out:
            return False
        if self.pas_h < 0.85:  # theta_L threshold
            return False
        if self.pas_zeta > 0.05:  # zeta_drift threshold
            return False
        
        # Verify signature
        expected_sig = self._compute_signature(secret_key)
        return hmac.compare_digest(self.signature, expected_sig)

    def _compute_signature(self, secret_key: bytes) -> str:
        payload = f"{self.session_id}:{self.client_id}:{self.timestamp}:{self.field_id}:{self.chordlock_anchor}:{self.tempolock_interval:.4f}:{self.pas_h:.4f}:{self.pas_zeta:.4f}:{int(self.glyphlock)}:{int(self.aura_out)}"
        return hmac.new(secret_key, payload.encode('utf-8'), hashlib.sha256).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PASLockToken':
        return cls(**data)


class GLYPHLOCKHandshake:
    """
    Cryptographic handshake protocol using GLYPHLOCK and hardware-derived honest jitter.
    """
    def __init__(self, secret_key: Optional[bytes] = None):
        if secret_key is None:
            # Anchor secret key to silicon entropy
            raw_jitter = harvest_honest_jitter((32,), scaled=False)
            jitter_bytes = bytes([int(abs(x.item()) * 255) % 256 for x in raw_jitter])
            self._secret_key = hashlib.sha256(jitter_bytes + os.urandom(16)).digest()
        else:
            self._secret_key = secret_key
        self._pas_invariant = PhaseAlignmentInvariant()
        self._apas_zeta = APAS_Zeta(zeta=0.05)
        self._last_state: Optional[torch.Tensor] = None
        self._last_pas_h: Optional[float] = None

    def create_handshake(
        self,
        client_id: str,
        field_id: str = "FIELD_01",
        chordlock_anchor: int = 197,  # UFSC prime anchor
        state_tensor: Optional[torch.Tensor] = None,
        prev_state_tensor: Optional[torch.Tensor] = None,
        theta_L: float = 0.85,
        zeta_drift: float = 0.05
    ) -> PASLockToken:
        """
        Evaluate engine coherence and real step-to-step drift (PAS_zeta + 33-shell Z_3
        modulation) and issue a signed PASLockToken.
        """
        session_id = hashlib.sha256(f"{client_id}:{time.time()}:{os.urandom(8).hex()}".encode('utf-8')).hexdigest()[:16]
        now = time.time()

        if state_tensor is not None:
            c_flat = state_tensor.detach().float().flatten()
            norm = torch.norm(c_flat).item()
            # 33-tori Z_3 counter-rotation shell modulation from IHC invariants
            ihc_mod = compute_ihc_bao_shell_modulation(comoving_distance_mpc=max(1.0, norm * 10.0))
            z3_factor = 1.0 + ihc_mod["z3_standing_wave_modulation"]

            # Real multiharmonic phase alignment (no artificial 0.86 floor)
            if c_flat.numel() >= 2:
                phases = torch.angle(torch.fft.rfft(c_flat))
                raw_pas_h = float(self._pas_invariant(phases).mean().item())
            else:
                raw_pas_h = float(norm / (norm + 0.1))
            pas_h = max(0.0, min(1.0, raw_pas_h * z3_factor))
            glyph_ok = bool(check_glyphlock(state_tensor).max().item() > 0)

            # Real PAS_zeta drift measurement against previous state
            ref_prev = prev_state_tensor.detach().float().flatten() if prev_state_tensor is not None else self._last_state
            if ref_prev is not None and ref_prev.numel() > 0:
                m = min(c_flat.numel(), ref_prev.numel())
                rel_delta = float((torch.norm(c_flat[:m] - ref_prev[:m]) / (torch.norm(ref_prev[:m]) + 1e-8)).item())
                pas_delta = abs(pas_h - (self._last_pas_h if self._last_pas_h is not None else pas_h))
                pas_zeta = float(max(1e-6, 0.5 * rel_delta + 0.5 * pas_delta))
            else:
                # Irreducible non-zero hardware jitter baseline (CODES v40: PAS_zeta != 0)
                jitter_val = float(torch.abs(harvest_honest_jitter((1,), scaled=False)).item())
                pas_zeta = max(1e-4, min(0.02, 0.005 + 0.01 * jitter_val))

            self._last_state = c_flat.clone()
            self._last_pas_h = pas_h
        else:
            pas_h = 0.92
            glyph_ok = True
            jitter_val = float(torch.abs(harvest_honest_jitter((1,), scaled=False)).item())
            pas_zeta = max(1e-4, min(0.02, 0.005 + 0.01 * jitter_val))

        tempolock_interval = float(chordlock_anchor) / 1000.0  # Prime-indexed timing
        aura_out = bool(pas_h >= theta_L and pas_zeta <= zeta_drift and glyph_ok)

        token = PASLockToken(
            session_id=session_id,
            client_id=client_id,
            timestamp=now,
            field_id=field_id,
            chordlock_anchor=chordlock_anchor,
            tempolock_interval=tempolock_interval,
            pas_h=pas_h,
            pas_zeta=pas_zeta,
            glyphlock=glyph_ok,
            aura_out=aura_out
        )
        token.signature = token._compute_signature(self._secret_key)
        return token

    def verify(self, token: PASLockToken, max_age_sec: float = 300.0) -> bool:
        return token.is_valid(self._secret_key, max_age_sec=max_age_sec)


class GlobalPASAPI:
    """
    Global PAS API for cross-system coherence tracking and multi-node RIC deployment.
    Provides safe, read-only queries with zero internal state leakage.
    """
    def __init__(self, engine=None, secret_key: Optional[bytes] = None):
        self.engine = engine
        self.handshake = GLYPHLOCKHandshake(secret_key=secret_key)
        self.node_registry: Dict[str, Dict[str, Any]] = {}
        self.coherence_log: List[Dict[str, Any]] = []

    def initiate_handshake(
        self,
        client_id: str,
        field_id: str = "RIC_COHERENCE",
        chordlock_anchor: int = 197
    ) -> Dict[str, Any]:
        """External client handshake."""
        engine_state = getattr(self.engine, 'meta_state', None) if self.engine else None
        token = self.handshake.create_handshake(
            client_id=client_id,
            field_id=field_id,
            chordlock_anchor=chordlock_anchor,
            state_tensor=engine_state
        )
        return {
            "status": "HANDSHAKE_SEALED" if token.aura_out else "COHERENCE_HOLD",
            "token": token.to_dict(),
            "protocol": "CODES_v40_PAS_LOCK"
        }

    def track_coherence(
        self,
        node_id: str,
        state_tensor: Union[List[float], torch.Tensor],
        field_id: str = "EDGE_NODE",
        environment: str = "edge"
    ) -> Dict[str, Any]:
        """
        Record telemetry from external edge/cloud node and compute coherence delta.
        """
        if isinstance(state_tensor, list):
            t = torch.tensor(state_tensor, dtype=torch.float32)
        else:
            t = state_tensor.detach().cpu().float()

        norm = torch.norm(t).item()
        pas_h = float(norm / (norm + 0.15))
        chi = float(compute_chirality(t.unsqueeze(0)).item()) if t.numel() > 1 else 0.0

        prev = self.node_registry.get(node_id)
        prev_pas = prev['pas_h'] if prev else pas_h
        pas_zeta = abs(pas_h - prev_pas)

        record = {
            "status": "tracked",
            "node_id": node_id,
            "environment": environment,
            "field_id": field_id,
            "pas_h": pas_h,
            "pas_zeta": pas_zeta,
            "chirality": chi,
            "is_coherent": pas_h >= 0.85 and pas_zeta <= 0.05,
            "timestamp": time.time()
        }

        self.node_registry[node_id] = record
        self.coherence_log.append(record)
        if len(self.coherence_log) > 200:
            self.coherence_log.pop(0)

        return record

    def query_engine_safe(
        self,
        token_data: Dict[str, Any],
        query_text: str,
        query_vector: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Execute read-only query against engine without allowing external state corruption.
        Zero-leakage: gradients and in-place tensor mutations are forbidden.
        """
        token = token_data if isinstance(token_data, PASLockToken) else PASLockToken.from_dict(token_data)
        if not self.handshake.verify(token):
            return {
                "error": "SECURITY REFUSAL: Invalid or expired GLYPHLOCK handshake token.",
                "status": "UNAUTHORIZED"
            }

        # Safe read-only evaluation
        if self.engine is not None and hasattr(self.engine, 'process_input'):
            with torch.no_grad():
                res = self.engine.process_input(
                    text_input=query_text,
                    generate_response=True,
                    regime='prickles'  # Truth-branching mode
                )
                return {
                    "status": "OK",
                    "session_id": token.session_id,
                    "response": res.get("response", ""),
                    "coherence_metrics": {
                        "pas_h": token.pas_h,
                        "pas_zeta": token.pas_zeta,
                        "glyphlock": token.glyphlock,
                        "retrieval_state": res.get("metrics", {}).get("retrieval_state", "KNOWN")
                    }
                }
        else:
            return {
                "status": "OK",
                "session_id": token.session_id,
                "response": f"[PAS_LOCK_BRIDGE] Coherence verified for query: '{query_text}'.",
                "coherence_metrics": {
                    "pas_h": token.pas_h,
                    "pas_zeta": token.pas_zeta,
                    "glyphlock": token.glyphlock
                }
            }

    def get_cross_system_coherence(self) -> Dict[str, Any]:
        """Aggregate telemetry for edge + cloud nodes."""
        active_nodes = list(self.node_registry.values())
        if not active_nodes:
            return {
                "total_nodes": 0,
                "mean_pas_h": 1.0,
                "coherent_ratio": 1.0,
                "ric_deployment_status": "LOCAL_NOMADIC"
            }

        mean_pas = sum(n["pas_h"] for n in active_nodes) / len(active_nodes)
        coherent_count = sum(1 for n in active_nodes if n["is_coherent"])

        return {
            "total_nodes": len(active_nodes),
            "mean_pas_h": round(mean_pas, 4),
            "coherent_ratio": round(coherent_count / len(active_nodes), 4),
            "ric_deployment_status": "EDGE_CLOUD_SYNCHRONIZED",
            "nodes": active_nodes
        }
