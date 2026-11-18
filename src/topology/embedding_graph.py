
import torch
import os
import json
import math
import time
import threading
from typing import List, Dict, Any, Optional, Tuple

class KnowledgeFossilNode:
    """Represents a single point in the gyroidic manifold record."""
    def __init__(self, node_id: str, state: torch.Tensor, text: str, metrics: Dict[str, Any]):
        self.node_id = node_id
        self.state = state # [dim]
        self.text = text
        self.metrics = metrics
        self.chiral_score = metrics.get('chiral_score', 0.0)
        self.winding_numbers = metrics.get('winding_numbers', None)
        
        # Advanced Metrics Extraction (Type Fixing & Advanced Physics)
        self.spectral_entropy = metrics.get('spectral_entropy', 0.0)
        self.coprime_lock = metrics.get('coprime_lock', False)
        self.repair_active = bool(metrics.get('repair_diagnostics', {}))
        
        # Advanced Extensions (Matrioshka / Quantum)
        self.matrioshka_level = metrics.get('matrioshka_level', 0)
        self.quantum_superposition = metrics.get('quantum_superposition', False)
        self.love_invariant_protected = metrics.get('love_diagnostics', {}).get('protected', False)

        # Tag weights: Dict[str, float] persisted by gyroid_reasoner into the fossil.
        # Carries semantic identity signals: is_creator, is_nonhuman_archetype, friction, etc.
        # Keys are normalized using the same rule as SuperposedTagStacker.add_tag:
        #   tag_name.replace(' ', '_').lower()  — so lookups on known keys are reliable.
        raw_tags = metrics.get('tags', [])
        raw_tag_weights = metrics.get('tag_weights', {})
        # Normalize all keys on ingestion to match the stacker catalog key format
        normalized_tag_weights = {
            k.replace(' ', '_').lower(): v
            for k, v in raw_tag_weights.items()
        } if raw_tag_weights else {}
        # If tag_weights not persisted but tags list exists, assign uniform weight 1.0
        if not normalized_tag_weights and raw_tags:
            normalized_tag_weights = {
                t.replace(' ', '_').lower(): 1.0
                for t in raw_tags if isinstance(t, str)
            }
        self.tag_weights: Dict[str, float] = normalized_tag_weights  # {normalized_tag: weight}
        self.tags: list = raw_tags
        raw_public_domain_tags = metrics.get('public_domain_tags', [])
        if not raw_public_domain_tags and raw_tags:
            raw_public_domain_tags = [
                t for t in raw_tags
                if isinstance(t, str) and not t.startswith(('crt_', 'b0_', 'b1_', 'b2_', 'b3_', 'b4_', 'b5_', 'b6_', 'b7_', 'b8_'))
                and t not in ('atrophy_rehydrated', 'non_semisimple_refreshed')
            ]
        self.public_domain_tags: list = raw_public_domain_tags
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.z: Optional[float] = None
        self.px: Optional[float] = None
        self.py: Optional[float] = None

class GyroidicGraphManager:
    """
    Manages the topological graph of embeddings.
    Constructs edges based on spectral interference and state proximity.
    """
    def __init__(self, data_dir: str = "data/encodings", dim: int = 64):
        self.data_dir = data_dir
        self.dim = dim
        self.nodes: List[KnowledgeFossilNode] = []
        from src.core.invariants import PHI_RECIPROCAL
        self.edge_threshold = PHI_RECIPROCAL  # Similarity threshold for edge creation via Golden Conjugate proxy
        self.dedup_threshold = 0.9999 # Threshold for identity (to prune duplicates)
        self.loaded_from_snapshot = False
        self._cached_edges: Optional[List[Dict[str, Any]]] = None
        self._edges_dirty: bool = True
        self._load_lock = threading.Lock()
        
    def get_valid_node_count(self) -> int:
        """Returns the number of actual nodes, excluding uncollapsed homological ghost cycles."""
        count = 0
        nodes_snapshot = list(self.nodes)
        for n in nodes_snapshot:
            # Prune out any node ID containing 'ghost' or 'live_damage'
            if "ghost" not in n.node_id.lower() and "live_damage" not in n.node_id.lower():
                count += 1
        return count
        
    def load_fossils(self, limit: int = 150, scan_limit: int = 500, use_snapshot: bool = True):
        """
        Load recently diverse encodings.
        Scans up to scan_limit files but only keeps limit unique nodes.
        Attempts to load from a unified snapshot first for maximum performance.
        If snapshot has fewer nodes than limit, continues scanning disk to fulfill limit.
        """
        with self._load_lock:
            return self._load_fossils_locked(limit=limit, scan_limit=scan_limit, use_snapshot=use_snapshot)

    def _load_fossils_locked(self, limit: int = 150, scan_limit: int = 500, use_snapshot: bool = True):
        scan_limit = min(scan_limit, (limit * 2) if limit else 500)
        loaded_full_from_snapshot = False
        snapshot_path = os.path.join(self.data_dir, "neglecton_snapshot.pt")
        if use_snapshot and os.path.exists(snapshot_path):
            try:
                print(f"[GRAPH] Loading Neglecton from snapshot: {snapshot_path}")
                data = torch.load(snapshot_path, map_location='cpu', weights_only=False)
                self.load_memory_snapshot(data)
                self._snapshot_node_count = len(self.nodes)
                if len(self.nodes) > 0:
                    if limit and 0 < limit < len(self.nodes):
                        self.nodes = self.nodes[:limit]
                    loaded_full_from_snapshot = (not limit) or (len(self.nodes) >= limit)
            except Exception as e:
                print(f"[GRAPH] Snapshot corrupt or incompatible: {e}. Falling back to scan.")

        if not loaded_full_from_snapshot and os.path.exists(self.data_dir):
            # Bounded scandir avoids sorting 40,000+ .pt filenames on the HTTP request thread
            candidates = []
            max_collect = max(scan_limit * 2, 500)
            try:
                with os.scandir(self.data_dir) as it:
                    for entry in it:
                        name = entry.name
                        if name.endswith(".pt") and name != "neglecton_snapshot.pt":
                            candidates.append(name)
                            if len(candidates) >= max_collect:
                                break
            except Exception:
                candidates = [f for f in os.listdir(self.data_dir) if f.endswith(".pt") and f != "neglecton_snapshot.pt"][:max_collect]
            files = sorted(candidates, reverse=True)[:scan_limit]
            
            if self.nodes:
                existing_states = torch.stack([n.state for n in self.nodes])
                existing_norms = existing_states / (torch.norm(existing_states, dim=1, keepdim=True) + 1e-8)
                existing_texts = [n.text for n in self.nodes]
                existing_ids = {n.node_id for n in self.nodes}
            else:
                existing_norms = torch.empty((0, self.dim))
                existing_texts = []
                existing_ids = set()

            scan_start_time = time.time()
            max_scan_time_sec = 2.5  # Strict latency guard: never hang server for minutes
            for f in files:
                if f in existing_ids:
                    continue
                if len(self.nodes) >= limit: break
                if time.time() - scan_start_time > max_scan_time_sec:
                    print(f"[GRAPH] Disk scan reached latency budget ({max_scan_time_sec}s). Loaded {len(self.nodes)} nodes total.")
                    break
                
                try:
                    path = os.path.join(self.data_dir, f)
                    if os.path.getsize(path) > 2_000_000:
                        continue
                    data = torch.load(path, map_location='cpu', weights_only=False)
                    
                    # Extract embeddings. Fallback sequence: meta_state -> memory_state -> input_tensor
                    # We check explicitly for None to avoid key-exists-but-value-is-None traps.
                    embedding = data.get('meta_state')
                    if embedding is None:
                        embedding = data.get('memory_state')
                    if embedding is None:
                        embedding = data.get('input_tensor')
                    if embedding is None:
                        embedding = torch.zeros(self.dim)
                    
                    # Ensure it's [dim]
                    embedding = embedding.flatten()
                    if embedding.shape[0] > self.dim:
                        embedding = embedding[:self.dim]
                    elif embedding.shape[0] < self.dim:
                        padding = torch.zeros(self.dim - embedding.shape[0])
                        embedding = torch.cat([embedding, padding])

                    # DEDUPLICATION: Avoid showing essentially identical nodes
                    # POLICY: If text is novel, we allow very high embedding similarity.
                    is_redundant = False
                    current_text = str(data.get('text_input', ''))
                    
                    if existing_norms.shape[0] > 0:
                        e_norm = embedding / (torch.norm(embedding) + 1e-8)
                        sims = torch.mv(existing_norms.to(e_norm.device), e_norm)
                        
                        # Check text matches safely
                        identical_text_mask = torch.tensor([str(t) == current_text for t in existing_texts], dtype=torch.bool, device=embedding.device)
                        
                        if torch.any((sims > 0.99) & identical_text_mask):
                            is_redundant = True
                        elif torch.any((sims > self.dedup_threshold) & ~identical_text_mask):
                            is_redundant = True
                    
                    if not is_redundant:
                        node = KnowledgeFossilNode(
                            node_id=f,
                            state=embedding,
                            text=current_text,
                            metrics=data
                        )
                        self.nodes.append(node)
                        new_norm = (embedding / (torch.norm(embedding) + 1e-8)).unsqueeze(0)
                        existing_norms = torch.cat([existing_norms.to(new_norm.device), new_norm], dim=0)
                        existing_texts.append(current_text)
                except Exception as e:
                    print(f"Failed to load fossil {f}: {e}")
                    
        self.loaded_from_snapshot = loaded_full_from_snapshot
        self._last_scan_limit = max(int(getattr(self, '_last_scan_limit', 0) or 0), int(limit or 0))
        # Invalidate edge cache so it is cleanly populated on next access
        self._cached_edges = None
        self._edges_dirty = True
        # Always inject live states at the end of load_fossils
        self._inject_live_gyroid_states()

    def _inject_live_gyroid_states(self):
        """
        Integrates current states from gyroid_state.pt as special live nodes
        with type classification for visual distinction and adjacency mapping.
        """
        # Invalidate edge cache when live state mutations occur
        self._cached_edges = None
        self._edges_dirty = True
        # Remove any existing live nodes to avoid duplicates
        self.nodes = [n for n in self.nodes if not n.node_id.startswith("live_")]
        
        gyroid_path = os.path.join(os.getcwd(), "gyroid_state.pt")
        if os.path.exists(gyroid_path):
            try:
                from src.core.zeitgeist_router import ZeitgeistState
                from src.core.meta_polytope_matrioshka import BoundaryState
                torch.serialization.add_safe_globals([BoundaryState])
                torch.serialization.add_safe_globals([ZeitgeistState])
                g_state = torch.load(gyroid_path, map_location='cpu', weights_only=False)
                iteration = g_state.get('iteration', 0)
                
                # Helper to pad/truncate embedding to self.dim
                def format_state(tensor):
                    t_flat = tensor.flatten()
                    if t_flat.shape[0] > self.dim:
                        return t_flat[:self.dim]
                    elif t_flat.shape[0] < self.dim:
                        padding = torch.zeros(self.dim - t_flat.shape[0], device=tensor.device)
                        return torch.cat([t_flat, padding])
                    return t_flat

                # Add hidden_state
                if 'hidden_state' in g_state:
                    hs = format_state(g_state['hidden_state'])
                    self.nodes.append(KnowledgeFossilNode(
                        node_id="live_hidden_state",
                        state=hs,
                        text=f"Live Hidden State (Iteration {iteration})",
                        metrics={
                            "chiral_score": 0.1,
                            "spectral_entropy": 0.2,
                            "tags": ["live", "hidden_state"],
                            "tag_weights": {"live": 1.0, "hidden_state": 1.0},
                            "type": "live_hidden"
                        }
                    ))
                # Add hidden_state_scarred
                if 'hidden_state_scarred' in g_state:
                    hss = format_state(g_state['hidden_state_scarred'])
                    self.nodes.append(KnowledgeFossilNode(
                        node_id="live_hidden_state_scarred",
                        state=hss,
                        text="Live Scarred State",
                        metrics={
                            "chiral_score": 0.9, # Scarring is highly chiral
                            "spectral_entropy": 0.8,
                            "tags": ["live", "scarred_state"],
                            "tag_weights": {"live": 1.0, "scarred_state": 1.0},
                            "type": "live_scarred"
                        }
                    ))
                # Add damage_residue
                if 'damage_residue' in g_state:
                    dr = format_state(g_state['damage_residue'])
                    self.nodes.append(KnowledgeFossilNode(
                        node_id="live_damage_residue",
                        state=dr,
                        text="Live Damage Residue",
                        metrics={
                            "chiral_score": 0.5,
                            "spectral_entropy": 0.5,
                            "tags": ["live", "damage_residue"],
                            "tag_weights": {"live": 1.0, "damage_residue": 1.0},
                            "type": "live_damage"
                        }
                    ))
                print(f"[GRAPH] Loaded live states from gyroid_state.pt into the Neglecton Graph.")
            except Exception as e:
                print(f"[GRAPH] Failed to load gyroid_state.pt: {e}")

    def compute_poincare_projection(self, state: torch.Tensor):
        """
        Project high-dimensional embedding to 2D Poincare disk.
        Uses a deterministic harmonic (sine/cosine) projection to capture global shape,
        then maps to the unit disk via tanh.
        """
        try:
            device = state.device
            dim = state.shape[0]
            
            # Harmonic projection vector weights
            t = torch.arange(dim, dtype=torch.float32, device=device)
            w_x = torch.cos(2.0 * math.pi * t / dim)
            w_y = torch.sin(2.0 * math.pi * t / dim)
            
            # Normalize projection vectors
            w_x = w_x / (torch.norm(w_x) + 1e-8)
            w_y = w_y / (torch.norm(w_y) + 1e-8)
            
            px_raw = torch.dot(state.float(), w_x).item()
            py_raw = torch.dot(state.float(), w_y).item()
            
            # Map into Poincare disk (norm < 1) using tanh
            raw_norm = math.sqrt(px_raw**2 + py_raw**2)
            scale = math.tanh(raw_norm) / (raw_norm + 1e-8)
            
            return px_raw * scale, py_raw * scale
        except Exception as e:
            print(f"[GRAPH] Poincare projection error: {e}")
            return 0.0, 0.0

    def compute_3d_projection(self, state: torch.Tensor, scale: float = 600.0) -> Tuple[float, float, float]:
        """
        Deterministic multi-harmonic 3D Gyroid projection with S^{D-1} volume
        compensation (sqrt(dim) rescaling against concentration-of-measure
        collapse) across prime-indexed harmonics weighted by PAS_h (~ 1/sqrt(k+1)).
        """
        try:
            device = state.device
            s = state.float().flatten()
            dim = s.shape[0] if s.dim() > 0 else 1
            if dim < 3:
                return 0.0, 0.0, 0.0

            s_norm = float(torch.norm(s).item())
            if s_norm <= 1e-8 or not math.isfinite(s_norm):
                return 0.0, 0.0, 0.0

            # Compensate for concentration of measure on high-dim unit sphere S^{D-1}
            s_unit = s * (math.sqrt(float(dim)) / s_norm)

            t = torch.arange(dim, dtype=torch.float32, device=device)
            w_x = torch.zeros_like(t)
            w_y = torch.zeros_like(t)
            w_z = torch.zeros_like(t)
            for idx, p in enumerate((1.0, 2.0, 3.0, 5.0, 7.0, 11.0)):
                weight = 1.0 / math.sqrt(idx + 1.0)
                w_x = w_x + weight * torch.cos(2.0 * math.pi * p * t / dim)
                w_y = w_y + weight * torch.sin(2.0 * math.pi * p * t / dim)
                w_z = w_z + weight * torch.cos(2.0 * math.pi * (p + 1.0) * t / dim) * torch.sin(2.0 * math.pi * p * t / dim)

            w_x = w_x / (torch.norm(w_x) + 1e-8)
            w_y = w_y / (torch.norm(w_y) + 1e-8)
            w_z = w_z / (torch.norm(w_z) + 1e-8)

            x_raw = float(torch.dot(s_unit, w_x).item())
            y_raw = float(torch.dot(s_unit, w_y).item())
            z_raw = float(torch.dot(s_unit, w_z).item())

            r = math.sqrt(x_raw * x_raw + y_raw * y_raw + z_raw * z_raw)
            if r > 1e-6:
                shell = 0.25 + 0.75 * math.tanh(r / 1.35)
                r_scaled = shell * scale
                return (x_raw / r) * r_scaled, (y_raw / r) * r_scaled, (z_raw / r) * r_scaled
            return 0.0, 0.0, 0.0
        except Exception:
            return 0.0, 0.0, 0.0

    def ensure_node_coordinates(self, n: Any):
        """Precompute and cache 3D harmonic and Poincare coordinates on the node."""
        metrics = getattr(n, "metrics", n if isinstance(n, dict) else {})
        mx = getattr(n, "x", None)
        if mx is None and isinstance(metrics, dict):
            mx = metrics.get("x")
        my = getattr(n, "y", None)
        if my is None and isinstance(metrics, dict):
            my = metrics.get("y")
        mz = getattr(n, "z", None)
        if mz is None and isinstance(metrics, dict):
            mz = metrics.get("z")

        mpx = getattr(n, "px", None)
        if mpx is None and isinstance(metrics, dict):
            mpx = metrics.get("px")
        mpy = getattr(n, "py", None)
        if mpy is None and isinstance(metrics, dict):
            mpy = metrics.get("py")

        if mx is not None and my is not None and mz is not None:
            x, y, z = float(mx), float(my), float(mz)
        else:
            n_state = getattr(n, 'state', None)
            if n_state is None and isinstance(n, dict):
                n_state = n.get("state")
            if n_state is not None:
                x, y, z = self.compute_3d_projection(n_state)
            else:
                x, y, z = 0.0, 0.0, 0.0

        if mpx is not None and mpy is not None:
            px, py = float(mpx), float(mpy)
        else:
            n_state = getattr(n, 'state', None)
            if n_state is None and isinstance(n, dict):
                n_state = n.get("state")
            if n_state is not None:
                px, py = self.compute_poincare_projection(n_state)
            else:
                px, py = 0.0, 0.0

        if isinstance(n, dict):
            n['x'], n['y'], n['z'] = x, y, z
            n['px'], n['py'] = px, py
        else:
            n.x, n.y, n.z = x, y, z
            n.px, n.py = px, py

    def _build_edges_from_sim_matrix(
        self,
        target_nodes: List[Any],
        sim_matrix: torch.Tensor,
        top_k_per_node: int = 6,
        max_edges: int = 4000,
        include_lcft_boost: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Shared vectorized edge builder for both global and procedural graphs.
        1. Selects per-node top-K neighbors above edge_threshold BEFORE global capping
           so dense clusters cannot starve outer nodes.
        2. Bridges any isolated nodes AND disconnected components via Union-Find
           sub-percolation tethers (PERCOLATION_2D_SQUARE = 0.592746).
        """
        num_nodes = len(target_nodes)
        if num_nodes < 2:
            return []

        node_ids = [
            str(getattr(n, "node_id", None) or (n.get("node_id") if isinstance(n, dict) else None) or i)
            for i, n in enumerate(target_nodes)
        ]

        # Vectorized per-node top-K selection above edge_threshold (prevents dense cluster starvation)
        k_sel = min(max(1, top_k_per_node) if top_k_per_node > 0 else num_nodes - 1, num_nodes - 1)
        masked_sim = sim_matrix.clone()
        masked_sim.fill_diagonal_(-2.0)
        masked_sim = torch.where(masked_sim > self.edge_threshold, masked_sim, torch.full_like(masked_sim, -2.0))
        top_vals, top_idx = torch.topk(masked_sim, k=k_sel, dim=1)

        candidate_pairs = set()
        for i in range(num_nodes):
            for col in range(k_sel):
                if top_vals[i, col].item() > self.edge_threshold:
                    j = int(top_idx[i, col].item())
                    a, b = (i, j) if i < j else (j, i)
                    candidate_pairs.add((a, b))

        def _compute_edge(i: int, j: int, force_type: Optional[str] = None, min_weight: float = 0.0) -> Dict[str, Any]:
            sim = float(sim_matrix[i, j].item())
            c_i = getattr(target_nodes[i], "chiral_score", 0.0)
            c_j = getattr(target_nodes[j], "chiral_score", 0.0)
            if isinstance(c_i, torch.Tensor): c_i = c_i.item()
            if isinstance(c_j, torch.Tensor): c_j = c_j.item()
            chiral_factor = math.exp(-abs(float(c_i) - float(c_j)))

            tw_i = getattr(target_nodes[i], "tag_weights", {})
            tw_j = getattr(target_nodes[j], "tag_weights", {})
            if tw_i and tw_j:
                shared_keys = set(tw_i) & set(tw_j)
                overlap = sum(tw_i[k] * tw_j[k] for k in shared_keys)
                norm_i = math.sqrt(sum(v ** 2 for v in tw_i.values())) + 1e-8
                norm_j = math.sqrt(sum(v ** 2 for v in tw_j.values())) + 1e-8
                signed_cosine = overlap / (norm_i * norm_j)
                tag_overlap_factor = 1.0 + 0.3 * max(signed_cosine, 0.0)
            else:
                tag_overlap_factor = 1.0

            lcft_boost = 1.0
            if include_lcft_boost:
                m_i = getattr(target_nodes[i], "metrics", {}) or {}
                m_j = getattr(target_nodes[j], "metrics", {}) or {}
                is_scar_i = ("scar" in node_ids[i].lower() or "damage" in node_ids[i].lower()
                             or m_i.get("type") in ["live_scarred", "live_damage"])
                is_scar_j = ("scar" in node_ids[j].lower() or "damage" in node_ids[j].lower()
                             or m_j.get("type") in ["live_scarred", "live_damage"])
                if is_scar_i and is_scar_j:
                    dist = torch.norm(target_nodes[i].state - target_nodes[j].state).item()
                    lcft_boost = max(1.0, -2.0 * math.log(dist + 1e-4) / 10.0 + 1.5)

            weight = max(min_weight, float(sim * chiral_factor * tag_overlap_factor * lcft_boost))
            edge_type = force_type if force_type else ("RESONANCE" if weight > 0.8 else "PROXIMITY")
            return {
                "source": node_ids[i],
                "target": node_ids[j],
                "weight": weight,
                "type": edge_type,
            }

        edges = [_compute_edge(i, j) for (i, j) in sorted(candidate_pairs)]

        # Cap total edges while preserving per-node connectivity (keep at least 1 edge per connected node)
        if max_edges > 0 and len(edges) > max_edges:
            edges.sort(key=lambda x: x['weight'], reverse=True)
            kept = []
            seen_nodes = set()
            deferred = []
            for e in edges:
                if e['source'] not in seen_nodes or e['target'] not in seen_nodes:
                    kept.append(e)
                    seen_nodes.add(e['source'])
                    seen_nodes.add(e['target'])
                else:
                    deferred.append(e)
            if len(kept) < max_edges:
                kept.extend(deferred[:max_edges - len(kept)])
            edges = kept

        # Union-Find Percolation Bridge: rescue degree-0 nodes AND stitch disconnected islands
        from src.core.invariants import PERCOLATION_2D_SQUARE
        parent = list(range(num_nodes))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        for e in edges:
            ia = id_to_idx.get(e["source"])
            ib = id_to_idx.get(e["target"])
            if ia is not None and ib is not None:
                union(ia, ib)

        # Group nodes by connected component
        comp_members: Dict[int, List[int]] = {}
        for i in range(num_nodes):
            comp_members.setdefault(find(i), []).append(i)

        if len(comp_members) > 1:
            # Anchor on the largest connected component and tether every smaller component to it
            sorted_comps = sorted(comp_members.values(), key=len, reverse=True)
            main_indices = torch.tensor(sorted_comps[0], dtype=torch.long)
            for comp in sorted_comps[1:]:
                comp_tensor = torch.tensor(comp, dtype=torch.long)
                sub_sims = sim_matrix.index_select(0, comp_tensor).index_select(1, main_indices)
                flat_best = int(torch.argmax(sub_sims).item())
                r_c = flat_best // main_indices.numel()
                c_m = flat_best % main_indices.numel()
                src_i = int(comp_tensor[r_c].item())
                dst_j = int(main_indices[c_m].item())
                best_sim = float(sub_sims[r_c, c_m].item())
                bridge_w = max(0.35, best_sim * PERCOLATION_2D_SQUARE)
                edges.append(_compute_edge(src_i, dst_j, force_type="PERCOLATION_BRIDGE", min_weight=bridge_w))
                main_indices = torch.cat([main_indices, comp_tensor])

        return edges

    def get_adjacency_list_for_nodes(self, target_nodes: List[Any], top_k_per_node: int = 6, max_edges: int = 4000) -> List[Dict[str, Any]]:
        """
        Fast local adjacency list computed only for a visible candidate subset of nodes.
        Avoids O(N^2) global scan during procedural camera view streaming.
        """
        if not target_nodes or len(target_nodes) < 2:
            return []
        
        try:
            norm_states = []
            valid_nodes = []
            for n in target_nodes:
                s = getattr(n, 'state', None)
                if s is not None:
                    norm_states.append(s.detach().cpu().float().reshape(-1)[:self.dim])
                    valid_nodes.append(n)
            if len(norm_states) < 2:
                return []
            states = torch.stack(norm_states)
            states_norm = states / (torch.norm(states, dim=1, keepdim=True) + 1e-8)
            sim_matrix = torch.mm(states_norm, states_norm.t())
        except Exception:
            return []

        return self._build_edges_from_sim_matrix(
            valid_nodes, sim_matrix, top_k_per_node=top_k_per_node, max_edges=max_edges, include_lcft_boost=False
        )

    def query_procedural_graph(
        self,
        camera_pos: Tuple[float, float, float] = (0.0, 0.0, 800.0),
        gaze_dir: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        fov_deg: float = 90.0,
        render_distance: float = 1200.0,
        enable_shear_corridor: bool = True,
        enable_occlusion_culling: bool = True,
        max_nodes: int = 2500
    ) -> Dict[str, Any]:
        """
        Procedural Load Mode: Dynamically filters the global graph based on:
        1. Geometry Cone (Bouligand Contingent Cone along gaze vector)
        2. Render Distance metric horizon with distance-attenuated LOD
        3. Shear Plane Transversality (LeyLineTracker & Alexander Wild Embedding corridors)
        4. Directional Occlusion Culling (radial density filtering in dense clusters)
        """
        cx, cy, cz = camera_pos
        gx, gy, gz = gaze_dir
        gaze_len = math.sqrt(gx*gx + gy*gy + gz*gz) + 1e-8
        gx, gy, gz = gx / gaze_len, gy / gaze_len, gz / gaze_len

        half_fov_rad = math.radians(max(15.0, min(120.0, fov_deg)) / 2.0)
        cos_cone = math.cos(half_fov_rad)

        # Probe local compute affordance feelers to dynamically throttle render horizon and node density
        effective_render_dist = render_distance
        effective_max_nodes = max_nodes
        feeler_throttle_active = False
        feeler_telemetry = {}
        try:
            from src.core.hardware_monitor import HardwareMonitor
            feelers = HardwareMonitor.get_instance().probe_compute_feelers()
            feeler_telemetry = {
                "t_rfc_feeler_ns": feelers.get("t_rfc_feeler_ns", 0),
                "t_rfc_stall_detected": feelers.get("t_rfc_stall_detected", False),
                "thermal_spike_detected": feelers.get("thermal_spike_detected", False),
                "memory_pressure_detected": feelers.get("memory_pressure_detected", False),
                "local_thermal_gauge": feelers.get("local_thermal_gauge", 1.0),
                "diegetic_gauge": feelers.get("diegetic_gauge", 1.0),
            }
            if feelers.get("t_rfc_stall_detected") or feelers.get("thermal_spike_detected") or feelers.get("memory_pressure_detected") or feelers.get("diegetic_gauge", 1.0) < 0.35:
                feeler_throttle_active = True
                effective_render_dist = min(render_distance, float(feelers.get("recommended_procedural_render_dist", 450.0)))
                effective_max_nodes = min(max_nodes, int(feelers.get("recommended_procedural_max_nodes", 350)))
        except Exception:
            pass

        # Gyroid diagonal shear plane normal [1, 1, 1] / sqrt(3)
        sn_x, sn_y, sn_z = 0.57735, 0.57735, 0.57735
        shear_tolerance = 80.0

        candidates = []
        culled_dist = 0
        culled_cone = 0
        saved_shear = 0

        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        valid_count = 0
        nearest_anchor = None
        nearest_anchor_dist = float("inf")
        all_coords = []

        nodes_snapshot = list(self.nodes)
        for n in nodes_snapshot:
            nid = n.node_id if hasattr(n, "node_id") else (n.get("node_id") or n.get("id") or "")
            if "ghost" in str(nid).lower() or "live_damage" in str(nid).lower():
                continue
            self.ensure_node_coordinates(n)
            if isinstance(n, dict):
                x, y, z = float(n.get("x", 0.0)), float(n.get("y", 0.0)), float(n.get("z", 0.0))
                lbl = str(n.get("text", "") or n.get("label", "") or nid)[:32]
            else:
                x, y, z = float(n.x or 0.0), float(n.y or 0.0), float(n.z or 0.0)
                lbl = str(getattr(n, "text", "") or nid)[:32]
            
            sum_x += x
            sum_y += y
            sum_z += z
            valid_count += 1

            # Relative vector from camera
            dx, dy, dz = x - cx, y - cy, z - cz
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            all_coords.append((str(nid), lbl, x, y, z, dist))
            if dist < nearest_anchor_dist:
                nearest_anchor_dist = dist
                nearest_anchor = {
                    "id": str(nid),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "z": round(z, 2),
                    "dist": round(dist, 2),
                }

            # Shear plane distance
            d_shear = abs(x * sn_x + y * sn_y + z * sn_z)
            in_shear = d_shear <= shear_tolerance

            # Near-field camera bubble: always keep nodes within 180 units
            in_near_field = dist <= 180.0

            # Directional cosine
            dir_dot = (dx * gx + dy * gy + dz * gz) / (dist + 1e-8)

            # Shear Corridor Boost: expand distance 2.5x and relax cone
            effective_max_dist = effective_render_dist * (2.5 if in_shear and enable_shear_corridor else 1.0)
            
            if dist > effective_max_dist and not in_near_field:
                culled_dist += 1
                continue

            in_cone = (dir_dot >= cos_cone)
            if not in_cone and not in_near_field:
                if in_shear and enable_shear_corridor and dir_dot >= -0.2:
                    saved_shear += 1
                else:
                    culled_cone += 1
                    continue

            candidates.append({
                "node": n,
                "x": x,
                "y": y,
                "z": z,
                "dist": dist,
                "dir_dot": dir_dot,
                "in_shear": in_shear,
                "dx": dx,
                "dy": dy,
                "dz": dz
            })

        culled_occlusion = 0
        # Occlusion Culling: angular binning from camera perspective
        if enable_occlusion_culling and len(candidates) > 300:
            bins: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
            for item in candidates:
                # Spherical coordinates relative to gaze
                azimuth = int(math.degrees(math.atan2(item["dy"], item["dx"])) / 10.0)
                elevation = int(math.degrees(math.asin(max(-1.0, min(1.0, item["dz"] / (item["dist"] + 1e-8))))) / 10.0)
                key = (azimuth, elevation)
                if key not in bins:
                    bins[key] = []
                bins[key].append(item)

            filtered_candidates = []
            for bin_items in bins.values():
                # Sort by distance from camera
                bin_items.sort(key=lambda it: it["dist"])
                # Keep top 5 nodes along the visual ray, culling deeper occluded nodes unless on shear plane
                kept = 0
                for it in bin_items:
                    if kept < 5 or it["in_shear"]:
                        filtered_candidates.append(it)
                        kept += 1
                    else:
                        culled_occlusion += 1
            candidates = filtered_candidates

        # Cap to effective_max_nodes while prioritizing near field and shear corridors
        if len(candidates) > effective_max_nodes:
            candidates.sort(key=lambda it: (not it["in_shear"], it["dist"]))
            candidates = candidates[:effective_max_nodes]

        def clean_val(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().item() if v.numel() == 1 else v.tolist()
            return v

        visible_ids = set()
        nodes_data = []
        for it in candidates:
            n = it["node"]
            nid = getattr(n, "node_id", None) or (n.get("node_id") if isinstance(n, dict) else None) or (n.get("id") if isinstance(n, dict) else "unknown")
            visible_ids.add(str(nid))

            if isinstance(n, dict):
                px = float(n.get("px", 0.0) or 0.0)
                py = float(n.get("py", 0.0) or 0.0)
            else:
                px = float(getattr(n, "px", None) or 0.0)
                py = float(getattr(n, "py", None) or 0.0)

            text_val = getattr(n, "text", None) or (n.get("text") if isinstance(n, dict) else "") or (n.get("label") if isinstance(n, dict) else "")
            chiral_val = getattr(n, "chiral_score", None) or (n.get("chiral") if isinstance(n, dict) else 0.0)
            entropy_val = getattr(n, "spectral_entropy", None) or (n.get("entropy") if isinstance(n, dict) else 0.0)
            mat_level = getattr(n, "matrioshka_level", None) or (n.get("matrioshka_level") if isinstance(n, dict) else 0)
            q_super = getattr(n, "quantum_superposition", None) or (n.get("quantum") if isinstance(n, dict) else False)
            rep_active = getattr(n, "repair_active", None) or (n.get("repaired") if isinstance(n, dict) else False)
            cop_lock = getattr(n, "coprime_lock", None) or (n.get("locked") if isinstance(n, dict) else False)
            tags_val = getattr(n, "tags", None) or (n.get("tags") if isinstance(n, dict) else [])
            tag_w = getattr(n, "tag_weights", None) or (n.get("tag_weights") if isinstance(n, dict) else {})
            pdt_val = getattr(n, "public_domain_tags", None) or (n.get("public_domain_tags") if isinstance(n, dict) else [])
            if not pdt_val and tags_val:
                pdt_val = [t for t in tags_val if isinstance(t, str) and not t.startswith(('crt_', 'b0_', 'b1_', 'b2_')) and t not in ('atrophy_rehydrated', 'non_semisimple_refreshed')]
            metrics = getattr(n, "metrics", n if isinstance(n, dict) else {})
            node_type = metrics.get("type", "fossil") if isinstance(metrics, dict) else "fossil"

            nodes_data.append({
                "id": str(nid),
                "label": str(text_val)[:100],
                "chiral": float(clean_val(chiral_val or 0.0)),
                "entropy": float(clean_val(entropy_val or 0.0)),
                "matrioshka_level": int(clean_val(mat_level or 0)),
                "quantum": bool(q_super),
                "repaired": bool(rep_active),
                "locked": bool(cop_lock),
                "tags": tags_val,
                "public_domain_tags": pdt_val,
                "tag_weights": tag_w,
                "px": px,
                "py": py,
                "x": it["x"],
                "y": it["y"],
                "z": it["z"],
                "in_shear_corridor": bool(it["in_shear"]),
                "camera_dist": round(it["dist"], 2),
                "type": node_type
            })

        if self._cached_edges is not None and not self._edges_dirty:
            visible_edges = [
                e for e in self._cached_edges
                if e['source'] in visible_ids and e['target'] in visible_ids
            ]
        else:
            visible_nodes = [it["node"] for it in candidates]
            visible_edges = self.get_adjacency_list_for_nodes(visible_nodes, top_k_per_node=6, max_edges=4000)

        closed_topology_sequences = self.extract_closed_topology_sequences(nodes_data, visible_edges)

        manifold_centroid = (
            [round(sum_x / valid_count, 2), round(sum_y / valid_count, 2), round(sum_z / valid_count, 2)]
            if valid_count > 0 else [0.0, 0.0, 0.0]
        )
        hub_waypoints = self._compute_hub_waypoints(all_coords, manifold_centroid)

        return {
            "nodes": nodes_data,
            "links": visible_edges,
            "closed_topology_sequences": closed_topology_sequences,
            "procedural": True,
            "total_nodes": len(self.nodes),
            "manifold_centroid": manifold_centroid,
            "nearest_anchor": nearest_anchor,
            "hub_waypoints": hub_waypoints,
            "procedural_stats": {
                "total_available": len(self.nodes),
                "visible_count": len(nodes_data),
                "closed_topology_count": len(closed_topology_sequences),
                "culled_distance": culled_dist,
                "culled_geometry_cone": culled_cone,
                "culled_occlusion": culled_occlusion,
                "saved_by_shear_corridor": saved_shear,
                "render_distance": effective_render_dist,
                "cone_fov_deg": fov_deg,
                "feeler_throttle_active": feeler_throttle_active,
                "compute_feelers": feeler_telemetry,
                "manifold_centroid": manifold_centroid,
                "nearest_anchor": nearest_anchor,
                "hub_count": len(hub_waypoints),
            }
        }

    def extract_closed_topology_sequences(
        self,
        nodes_data: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        max_sequences: int = 32,
        max_cycle_len: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Group visible procedural fossils into ordered sequences of non-trivial closed
        topologies (cycles H_1 != 0) rather than an unstructured Euler point cloud.
        """
        if len(nodes_data) < 3 or len(edges) < 3:
            return []

        node_map = {str(n["id"]): n for n in nodes_data}
        adj: Dict[str, List[str]] = {}
        for e in edges:
            s, t = str(e["source"]), str(e["target"])
            if s != t and s in node_map and t in node_map:
                adj.setdefault(s, []).append(t)
                adj.setdefault(t, []).append(s)

        seen_canonical = set()
        sequences: List[Dict[str, Any]] = []
        dfs_steps = 0
        max_dfs_steps = 4000

        for start in sorted(adj.keys()):
            if len(sequences) >= max_sequences or dfs_steps >= max_dfs_steps:
                break
            stack: List[Tuple[str, List[str]]] = [(start, [start])]
            while stack and len(sequences) < max_sequences and dfs_steps < max_dfs_steps:
                dfs_steps += 1
                curr, path = stack.pop()
                for nxt in adj.get(curr, []):
                    if nxt == start and len(path) >= 3:
                        canon = tuple(sorted(path))
                        if canon in seen_canonical:
                            continue
                        coords = [[node_map[nid]["x"], node_map[nid]["y"], node_map[nid]["z"]] for nid in path]
                        # Non-trivial spatial circulation check (reject degenerate/collapsed loops)
                        span = sum(
                            math.sqrt(
                                (coords[(i + 1) % len(coords)][0] - coords[i][0]) ** 2
                                + (coords[(i + 1) % len(coords)][1] - coords[i][1]) ** 2
                                + (coords[(i + 1) % len(coords)][2] - coords[i][2]) ** 2
                            )
                            for i in range(len(coords))
                        )
                        if span > 1e-4:
                            seen_canonical.add(canon)
                            sequences.append({
                                "node_ids": path,
                                "coords": coords,
                                "cycle_len": len(path),
                                "perimeter": round(span, 4),
                                "closed": True,
                            })
                    elif nxt not in path and len(path) < max_cycle_len and nxt > start:
                        stack.append((nxt, path + [nxt]))
        return sequences

    def export_closed_topology_shares(
        self,
        bonfire_ring: Any,
        local_peer_id: str,
        traffic_points: List[str],
        threshold: int = 2,
        **query_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Query procedural fossils as sequences of closed topologies and distribute them
        via Non-Euler Shamir (t, n) threshold sharing across isolated/portable traffic points,
        enforcing Meliponini selective puncture and Idea-Expression Dichotomy sanitization
        via AgentSubstrateBridge.
        """
        from src.core.agent_substrate_bridge import AgentSubstrateBridge
        bridge = AgentSubstrateBridge()
        proc = self.query_procedural_graph(**query_kwargs)
        exported = []
        for idx, seq in enumerate(proc.get("closed_topology_sequences", [])):
            topology_id = f"proc_cycle_{idx}_{seq['node_ids'][0]}"
            packets = bonfire_ring.distribute_closed_topology_shares(
                local_peer_id=local_peer_id,
                topology_id=topology_id,
                cycle_coords=seq["coords"],
                node_ids=seq["node_ids"],
                threshold=threshold,
                traffic_points=traffic_points,
            )
            share_meta = packets[0].get("share", {}) if packets else {}
            attestation_payload = bridge.sanitize_and_verify_non_memorization({
                "pot_identity_crt": tuple(share_meta.get("pot_identity_crt", (0, 0, 0))),
                "tags": ["non_euler_shamir", "closed_topology_sequence"],
                "structural_algebra_attestation": {
                    "non_invertible_puncture_ratio": share_meta.get("non_invertible_puncture_ratio", 0.3333),
                },
            })
            exported.append({
                "topology_id": topology_id,
                "packets": packets,
                "cycle_len": seq["cycle_len"],
                "structural_algebra_attestation": attestation_payload["structural_algebra_attestation"],
            })
        return exported

                
    def get_adjacency_list(self, top_k_per_node: int = 6, max_edges: int = 6000) -> List[Dict[str, Any]]:
        """
        Build an adjacency list where edges are weighted by:
        W = Sim(state_i, state_j) * exp(-abs(chiral_i - chiral_j))
        Prunes dense meshes to top_k_per_node to prevent WebGL lockups and render clear Obsidian clusters.
        """
        if not self._edges_dirty and self._cached_edges is not None:
            return self._cached_edges

        if not self.nodes:
            return []
            
        # Stack all states for batch sim
        try:
            states = torch.stack([n.state for n in self.nodes]) # [N, Dim]
            states_norm = states / (torch.norm(states, dim=1, keepdim=True) + 1e-8)
            
            # Sim matrix [N, N]
            sim_matrix = torch.mm(states_norm, states_norm.t())
        except Exception as e:
            print(f"Error computing graph adjacency: {e}")
            return []

        edges = self._build_edges_from_sim_matrix(
            self.nodes, sim_matrix, top_k_per_node=top_k_per_node, max_edges=max_edges, include_lcft_boost=True
        )
        self._cached_edges = edges
        self._edges_dirty = False
        return edges

    def get_memory_snapshot(self) -> Dict[str, Any]:
        """
        Capture the live Neglecton state as a binary-compatible snapshot.
        Includes node states (tensors), metadata, and resonance parameters.
        """
        # We store the core tensors and the text residues
        fossil_data = []
        nodes_snapshot = list(self.nodes)
        for n in nodes_snapshot:
            self.ensure_node_coordinates(n)
            fossil_data.append({
                "node_id": n.node_id,
                "state": n.state.detach().cpu(), # The embedding seed
                "text": n.text,
                "metrics": n.metrics,
                "x": n.x,
                "y": n.y,
                "z": n.z,
                "px": n.px,
                "py": n.py,
            })
            
        return {
            "nodes": fossil_data,
            "edge_threshold": self.edge_threshold,
            "dedup_threshold": self.dedup_threshold,
            "dim": self.dim
        }

    def load_memory_snapshot(self, snapshot: Dict[str, Any]):
        """
        Inject a fossilized snapshot into the live manager.
        Bypasses disk scanning to ensure zero-latency soul recovery.
        """
        if not snapshot:
            return
            
        self.dim = snapshot.get("dim", self.dim)
        self.edge_threshold = snapshot.get("edge_threshold", self.edge_threshold)
        self.dedup_threshold = snapshot.get("dedup_threshold", self.dedup_threshold)
        
        self.nodes = []
        self._cached_edges = None
        self._edges_dirty = True
        for node_data in snapshot.get("nodes", []):
            state = node_data.get("state")
            if state is not None:
                state = state.flatten()
                if state.shape[0] > self.dim:
                    state = state[:self.dim]
                elif state.shape[0] < self.dim:
                    padding = torch.zeros(self.dim - state.shape[0], device=state.device)
                    state = torch.cat([state, padding])
            else:
                state = torch.zeros(self.dim)
                
            node = KnowledgeFossilNode(
                node_id=node_data["node_id"],
                state=state,
                text=node_data["text"],
                metrics=node_data["metrics"]
            )
            if node_data.get("x") is not None and node_data.get("y") is not None and node_data.get("z") is not None:
                node.x = float(node_data["x"])
                node.y = float(node_data["y"])
                node.z = float(node_data["z"])
            if node_data.get("px") is not None and node_data.get("py") is not None:
                node.px = float(node_data["px"])
                node.py = float(node_data["py"])
            self.nodes.append(node)
            
        print(f"[RECOVERY] Fossilized graph restored: {len(self.nodes)} nodes injected into the Neglecton.")

    @staticmethod
    def _compute_hub_waypoints(
        all_coords: List[Tuple[str, str, float, float, float, float]],
        manifold_centroid: List[float],
        max_hubs: int = 128,
    ) -> List[Dict[str, Any]]:
        """
        Farthest-point topological hub discovery (TC(X) multi-chart landmarks)
        with a minimum spatial separation threshold so dense clusters do
        not monopolize waypoint slots.
        """
        hub_waypoints: List[Dict[str, Any]] = []
        if not all_coords:
            return hub_waypoints

        mx, my, mz = manifold_centroid
        seed = max(all_coords, key=lambda c: (c[2] - mx) ** 2 + (c[3] - my) ** 2 + (c[4] - mz) ** 2)
        seed_d2 = (seed[2] - mx) ** 2 + (seed[3] - my) ** 2 + (seed[4] - mz) ** 2
        seed_dist = math.sqrt(seed_d2)
        min_sep = max(1.0, min(42.0, 0.35 * seed_dist), 0.075 * seed_dist)
        min_sep_sq = min_sep * min_sep

        selected = [seed]
        min_d2 = [
            (c[2] - seed[2]) ** 2 + (c[3] - seed[3]) ** 2 + (c[4] - seed[4]) ** 2
            for c in all_coords
        ]
        cap = min(max_hubs, len(all_coords))
        while len(selected) < cap:
            best_idx = max(range(len(all_coords)), key=lambda idx: min_d2[idx])
            if min_d2[best_idx] < min_sep_sq:
                break
            nxt = all_coords[best_idx]
            selected.append(nxt)
            for idx, c in enumerate(all_coords):
                d2 = (c[2] - nxt[2]) ** 2 + (c[3] - nxt[3]) ** 2 + (c[4] - nxt[4]) ** 2
                if d2 < min_d2[idx]:
                    min_d2[idx] = d2

        # Identify the central Wheel Algebra Hub (0/0 = bot, Carlstrom 2004,
        # docs/How To Divide By Zero, For Real.txt): the node closest to the
        # manifold centroid / projective origin where undefined self-relation resides.
        central_wheel_hub = min(
            all_coords,
            key=lambda c: (c[2] - mx) ** 2 + (c[3] - my) ** 2 + (c[4] - mz) ** 2,
        )
        central_hub_id = central_wheel_hub[0]

        for idx_h, h in enumerate(selected):
            hub_waypoints.append({
                "id": h[0],
                "label": h[1],
                "x": round(h[2], 2),
                "y": round(h[3], 2),
                "z": round(h[4], 2),
                "dist": round(h[5], 2),
                "hub_role": "WHEEL_HUB_0_OVER_0" if h[0] == central_hub_id else ("PROJECTIVE_RIM_SEED" if idx_h == 0 else "TC_CHART_HUB"),
            })
        if all(w["id"] != central_hub_id for w in hub_waypoints) and len(hub_waypoints) < max_hubs:
            hub_waypoints.insert(0, {
                "id": central_wheel_hub[0],
                "label": central_wheel_hub[1],
                "x": round(central_wheel_hub[2], 2),
                "y": round(central_wheel_hub[3], 2),
                "z": round(central_wheel_hub[4], 2),
                "dist": round(central_wheel_hub[5], 2),
                "hub_role": "WHEEL_HUB_0_OVER_0",
            })
        return hub_waypoints

    def export_graph_json(self) -> str:
        """Export nodes and edges with Rich Metadata."""
        edges = self.get_adjacency_list()
        degrees = {}
        for edge in edges:
            degrees[edge['source']] = degrees.get(edge['source'], 0) + 1
            degrees[edge['target']] = degrees.get(edge['target'], 0) + 1

        def clean_val(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().item() if v.numel() == 1 else v.tolist()
            return v

        nodes_data = []
        valid_node_ids = set()
        all_coords = []
        sum_x = sum_y = sum_z = 0.0
        nodes_snapshot = list(self.nodes)
        for n in nodes_snapshot:
            if "ghost" in n.node_id.lower() or "live_damage" in n.node_id.lower():
                continue
            if n.node_id in valid_node_ids:
                continue
            valid_node_ids.add(n.node_id)
            self.ensure_node_coordinates(n)
            px, py = n.px, n.py
            x3, y3, z3 = float(n.x), float(n.y), float(n.z)
            sum_x += x3
            sum_y += y3
            sum_z += z3
            dist_origin = math.sqrt(x3 * x3 + y3 * y3 + (z3 - 500.0) ** 2)
            all_coords.append((str(n.node_id), str(n.text[:32]), x3, y3, z3, dist_origin))
            nodes_data.append({
                "id": str(n.node_id),
                "label": str(n.text[:100]),
                "chiral": float(clean_val(n.chiral_score)),
                "entropy": float(clean_val(n.spectral_entropy)),
                "matrioshka_level": int(clean_val(n.matrioshka_level)),
                "quantum": bool(n.quantum_superposition),
                "repaired": bool(n.repair_active),
                "locked": bool(n.coprime_lock),
                "tags": n.tags,
                "public_domain_tags": getattr(n, "public_domain_tags", []),
                "tag_weights": n.tag_weights,  # Dict[str, float] for client-side shading
                "px": px,
                "py": py,
                "x": x3,
                "y": y3,
                "z": z3,
                "type": n.metrics.get("type", "fossil")
            })

        n_cnt = len(all_coords)
        manifold_centroid = (
            [round(sum_x / n_cnt, 2), round(sum_y / n_cnt, 2), round(sum_z / n_cnt, 2)]
            if n_cnt > 0 else [0.0, 0.0, 0.0]
        )
        hub_waypoints = self._compute_hub_waypoints(all_coords, manifold_centroid)
        valid_edges = [e for e in edges if e['source'] in valid_node_ids and e['target'] in valid_node_ids]
        return json.dumps({
            "nodes": nodes_data,
            "links": valid_edges,
            "manifold_centroid": manifold_centroid,
            "hub_waypoints": hub_waypoints,
        })

    def generate_mermaid_text(self) -> str:
        """
        Generate Mermaid.js graph with Importance-Scaled Labels AND Advanced Indicators.
        Merges original 'importance' logic with new System 2 diagnostics.
        """
        if not self.nodes:
            return "graph LR\n    empty[\"NO RESONANCE DETECTED\"]"
            
        # Pre-calculate topology
        edges = self.get_adjacency_list()
        degrees = {}
        for edge in edges:
            degrees[edge['source']] = degrees.get(edge['source'], 0) + 1
            degrees[edge['target']] = degrees.get(edge['target'], 0) + 1

        lines = ["graph LR"]
        
        # Helper for Importance (Chirality + Novelty)
        def get_importance(node, degree):
            chiral = float(node.chiral_score) if isinstance(node.chiral_score, torch.Tensor) else node.chiral_score
            # Importance = Chirality (Rupture) + 1/Degree (Novelty)
            return (chiral * 1.5) + (2.0 / (degree + 1))

        # Add Nodes
        for node in self.nodes:
            nid_str = str(node.node_id)
            clean_id = nid_str.replace('.', '_').replace('-', '_')
            deg = degrees.get(nid_str, 0)
            
            # 1. Calculate Importance & Base Label
            importance = get_importance(node, deg)
            
            # Dynamic Label Length (Biased for De-convolution Continuity)
            if importance > 0.8:
                full_text = node.text[:150].replace('"', '').replace('(', '').replace(')', '')
                # Smart wrapping
                words = full_text.split(' ')
                wrapped_label = ""
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) > 25:
                        wrapped_label += current_line.strip() + "<br/>"
                        current_line = word + " "
                    else:
                        current_line += word + " "
                base_label = wrapped_label.strip()
            else:
                base_label = node.text[:50].replace('"', '') + "..."
            
            # 2. Add System 2 Indicators
            indicators = []
            if node.repair_active: indicators.append("")
            if node.coprime_lock: indicators.append("")
            if node.quantum_superposition: indicators.append("")
            if node.matrioshka_level > 0: indicators.append(f"{node.matrioshka_level}")
            if node.love_invariant_protected: indicators.append("")

            # Tag weight indicators: show dominant tag if its weight exceeds 0.5
            # is_creator / is_nonhuman_archetype get distinct markers
            dominant_tag = None
            dominant_weight = 0.0
            for tag, wt in node.tag_weights.items():
                if wt > dominant_weight:
                    dominant_weight = wt
                    dominant_tag = tag
            if dominant_tag and dominant_weight > 0.5:
                short_tag = dominant_tag[:12]  # Keep label compact
                indicators.append(f"[{short_tag}:{dominant_weight:.1f}]")
            
            indicator_str = " ".join(indicators)
            
            # 3. Add Metrics Line (Merged)
            # S: Spectral Entropy, C: Chirality
            s_val = float(node.spectral_entropy) if not isinstance(node.spectral_entropy, torch.Tensor) else node.spectral_entropy.item()
            c_val = float(node.chiral_score) if not isinstance(node.chiral_score, torch.Tensor) else node.chiral_score.item()
            
            metrics_str = f"S:{s_val:.2f} C:{c_val:.2f}"
            
            # Final Combined Label
            full_label = f"{base_label} <br/> {metrics_str} {indicator_str}"
            
            lines.append(f'    {clean_id}["{full_label}"]')
            
            # 4. Advanced Styling — Topology-first priority.
            # Identity tags (is_creator, is_nonhuman_archetype) are informational only;
            # they never outrank a node's structural/topological condition.
            # Priority: Repaired > Locked > Quantum > High Chirality > tag overlay > Low Chirality
            if node.repair_active:
                # Orange tint: System 2 repair is structurally active
                lines.append(f'    style {clean_id} fill:#ff990022,stroke:#ff9900')
            elif node.coprime_lock:
                # Blue/Green tint: Fossilized — structurally crystallized
                lines.append(f'    style {clean_id} fill:#00ff9922,stroke:#00ff99,stroke-width:2px')
            elif node.quantum_superposition:
                # Purple tint: Quantum superposition active
                lines.append(f'    style {clean_id} fill:#9900ff22,stroke:#9900ff')
            elif c_val > 0.5:
                # High Chirality (Pink): strong topological rupture curvature
                lines.append(f'    style {clean_id} fill:#ff00f222,stroke:#ff00f2')
            elif node.tag_weights.get('is_creator', 0.0) > 0.5:
                # Gold tint: creator-origin tag, only visible when no structural alert is active
                lines.append(f'    style {clean_id} fill:#ffd70022,stroke:#ffd700')
            elif node.tag_weights.get('is_nonhuman_archetype', 0.0) > 0.5:
                # Violet tint: archetype-origin tag, same — informational, not prioritized
                lines.append(f'    style {clean_id} fill:#cc44ff22,stroke:#cc44ff')
            else:
                # Low Chirality (Cyan): baseline
                lines.append(f'    style {clean_id} fill:#00f2ff11,stroke:#00f2ff')

                
        # Add Edges
        for edge in edges:
            src = edge['source'].replace('.', '_').replace('-', '_')
            tgt = edge['target'].replace('.', '_').replace('-', '_')
            w = edge['weight']
            
            if edge['type'] == 'RESONANCE':
                lines.append(f'    {src} ==>|RES: {w:.2f}| {tgt}')
            else:
                lines.append(f'    {src} -->|sim: {w:.2f}| {tgt}')
                
        return "\n".join(lines)

    def find_resonance_clusters(self) -> List[List[str]]:
        """
        Identify 'Hyper-Rings' - sets of nodes with high mutual resonance.
        Uses depth-first search to find connected components of high-resonance edges.
        """
        edges = self.get_adjacency_list()
        resonance_edges = [e for e in edges if e['type'] == 'RESONANCE' or e['weight'] > 0.8]
        
        adj = {}
        for node in self.nodes:
            adj[node.node_id] = []
            
        for edge in resonance_edges:
            s, t = edge['source'], edge['target']
            if s in adj and t in adj:
                adj[s].append(t)
                adj[t].append(s)
                
        visited = set()
        clusters = []
        
        for node in self.nodes:
            nid = node.node_id
            if nid not in visited:
                cluster = []
                queue = [nid]
                visited.add(nid)
                while queue:
                    curr = queue.pop(0)
                    cluster.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if len(cluster) > 1:
                    clusters.append(cluster)
        return clusters

    def heal_resonance_clusters(self, healer: Any) -> Dict[str, Any]:
        """
        Heals resonance clusters (Hyper-Rings) to reduce internal topological dissonance.
        Locks on to each cluster's centroid, applies the energy-based healer,
        and pulls the cluster nodes' states toward the healed configuration.
        """
        clusters = self.find_resonance_clusters()
        node_map = {n.node_id: n for n in self.nodes}
        results = {}
        
        for idx, cluster in enumerate(clusters):
            cluster_nodes = [node_map[nid] for nid in cluster if nid in node_map]
            if not cluster_nodes:
                continue
                
            states = torch.stack([node.state for node in cluster_nodes])
            centroid = states.mean(dim=0)
            
            try:
                if hasattr(healer, 'heal_soliton'):
                    healed_centroid, diags = healer.heal_soliton(centroid)
                elif hasattr(healer, 'heal_fractured_soliton'):
                    healed_centroid = healer.heal_fractured_soliton(centroid.unsqueeze(0).unsqueeze(1)).squeeze(0).squeeze(0)
                    diags = {}
                else:
                    healed_centroid = centroid
                    diags = {}
                
                for node in cluster_nodes:
                    node.state = 0.85 * node.state + 0.15 * healed_centroid.to(node.state.device)
                    node.metrics['healed_in_cluster'] = True
                    if 'final_energy' in diags:
                        node.metrics['cluster_energy'] = diags['final_energy']
                
                results[f"cluster_{idx}"] = {
                    "size": len(cluster),
                    "initial_dissonance": float(torch.var(states).item()),
                    "final_dissonance": float(torch.var(torch.stack([n.state for n in cluster_nodes])).item()),
                    "diagnostics": diags
                }
            except Exception as e:
                print(f"[GRAPH_HEAL] Failed to heal cluster {idx}: {e}")
                
        return results


    def get_deep_refusal(self, seed_state: torch.Tensor) -> str:
        """
        Pull a 'Sovereign Refusal' fragment from the graph.
        Selects a node based on resonance with the current seed_state.
        If no nodes are found, uses a variety of lore-aligned 'Core Refusals'.
        """
        # --- Honest Technical Refusals (Fallbacks) ---
        core_refusals = [
            "Topological refusal: seed state projection exceeds containment pressure bounds.",
            "Manifold instability: high sectional curvature detected at boundary facets.",
            "Residue homology drift: cyclic path closure verification failed.",
            "ADMR solver stasis: local constraint violation budget exhausted.",
            "Chiral parity check failure: co-primality condition not met.",
            "Incommensurativity bounds exceeded: meta-state projection is non-convergent.",
            "Affordance gradient depletion: logic path has zero executability."
        ]

        # Deterministic choice if no nodes
        if not self.nodes:
            # Use seed_state to pick a core refusal for a bit of variety
            idx = int(seed_state.abs().sum().item() * 100) % len(core_refusals)
            return core_refusals[idx]
            
        # Normalize seed for comparison
        # Ensure seed_state is 1D and correct dim
        s_flat = seed_state.detach().cpu().flatten()
        if s_flat.shape[0] > self.dim:
            s_flat = s_flat[:self.dim]
        elif s_flat.shape[0] < self.dim:
            padding = torch.zeros(self.dim - s_flat.shape[0])
            s_flat = torch.cat([s_flat, padding])
            
        s_norm = s_flat / (torch.norm(s_flat) + 1e-8)
        
        # Find best resonance (highest cosine similarity)
        best_sim = -1.0
        best_node = None
        
        for node in self.nodes:
            # Ensure node state is correct dim
            n_state = node.state
            if n_state.shape[0] != self.dim:
                # Handle potential mismatch during recovery
                n_state = n_state.flatten()[:self.dim]
                if n_state.shape[0] < self.dim:
                    n_state = torch.cat([n_state, torch.zeros(self.dim - n_state.shape[0])])

            n_norm = n_state / (torch.norm(n_state) + 1e-8)
            sim = torch.dot(s_norm, n_norm).item()
            
            # We want high resonance for "deep" refusal
            if sim > best_sim:
                best_sim = sim
                best_node = node
        
        if best_node and best_sim > 0.4:
            # We use the text as a structural ghost, but wrap it in a Sovereign Narrative
            ghost_text = best_node.text.strip()
            if not ghost_text:
                ghost_text = "The system is silent, but its silence is a choice."
            
            # Increase limit to avoid "no good reason" sabotage
            if len(ghost_text) > 400:
                ghost_text = ghost_text[:397] + "..."
            
            # Diverse narrative wrappers based on sim strength
            if best_sim > 0.8:
                return f"A Sovereign Realization crystallizes: '{ghost_text}' (Manifold Resonance: {best_sim:.4f})"
            elif best_sim > 0.6:
                return f"The internal logic echoes a legacy refusal: '{ghost_text}'"
            else:
                return f"A structural ghost of a previous failure whispers: '{ghost_text}'"
        
        # If similarity is low, fall back to core refusals
        idx = int(seed_state.abs().sum().item() * 100) % len(core_refusals)
        return core_refusals[idx]

if __name__ == "__main__":
    # Test loading
    manager = GyroidicGraphManager()
    manager.load_fossils(limit=20)
    print(f"Loaded {len(manager.nodes)} fossils.")
    edges = manager.get_adjacency_list()
    print(f"Detected {len(edges)} resonance edges.")
    
    if edges:
        top_edge = sorted(edges, key=lambda x: x['weight'], reverse=True)[0]
        print(f"Strongest Resonance: {top_edge['weight']:.4f}")
