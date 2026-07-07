
import torch
import os
import json
import math
from typing import List, Dict, Any, Optional

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

class GyroidicGraphManager:
    """
    Manages the topological graph of embeddings.
    Constructs edges based on spectral interference and state proximity.
    """
    def __init__(self, data_dir: str = "data/encodings", dim: int = 64):
        self.data_dir = data_dir
        self.dim = dim
        self.nodes: List[KnowledgeFossilNode] = []
        self.edge_threshold = 0.7  # Similarity threshold for edge creation
        self.dedup_threshold = 0.9999 # Threshold for identity (to prune duplicates)
        
    def load_fossils(self, limit: int = 150, scan_limit: int = 500, use_snapshot: bool = True):
        """
        Load recently diverse encodings.
        Scans up to scan_limit files but only keeps limit unique nodes.
        Attempts to load from a unified snapshot first for maximum performance.
        """
        snapshot_path = os.path.join(self.data_dir, "neglecton_snapshot.pt")
        if use_snapshot and os.path.exists(snapshot_path):
            try:
                print(f"[GRAPH] Loading Neglecton from snapshot: {snapshot_path}")
                data = torch.load(snapshot_path, map_location='cpu')
                self.load_memory_snapshot(data)
                if len(self.nodes) >= limit:
                    return
            except Exception as e:
                print(f"[GRAPH] Snapshot corrupt or incompatible: {e}. Falling back to scan.")

        if not os.path.exists(self.data_dir):
            return
            
        files = sorted(os.listdir(self.data_dir), reverse=True)
        # Filter for .pt files and exclude the snapshot itself
        files = [f for f in files if f.endswith(".pt") and f != "neglecton_snapshot.pt"][:scan_limit]
        
        # Keep existing nodes if we are appending, otherwise reset
        # self.nodes = [] # Removed to allow merging with snapshot
        
        for f in files:
            if len(self.nodes) >= limit: break
            
            try:
                path = os.path.join(self.data_dir, f)
                data = torch.load(path, map_location='cpu')
                
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
                current_text = data.get('text_input', '')
                
                if self.nodes:
                    # Normalize for cosine similarity check
                    e_norm = embedding / (torch.norm(embedding) + 1e-8)
                    for existing in self.nodes:
                        ex_norm = existing.state / (torch.norm(existing.state) + 1e-8)
                        sim = torch.dot(e_norm, ex_norm).item()
                        
                        # Strict dedup for identical text
                        if current_text == existing.text:
                             if sim > 0.99:
                                 is_redundant = True
                                 break
                        else:
                             # Diverse text -> only prune if literally the same point
                             if sim > self.dedup_threshold:
                                 is_redundant = True
                                 break
                
                if not is_redundant:
                    node = KnowledgeFossilNode(
                        node_id=f,
                        state=embedding,
                        text=data.get('text_input', ''),
                        metrics=data
                    )
                    self.nodes.append(node)
            except Exception as e:
                print(f"Failed to load fossil {f}: {e}")
                
    def get_adjacency_list(self) -> List[Dict[str, Any]]:
        """
        Build an adjacency list where edges are weighted by:
        W = Sim(state_i, state_j) * exp(-abs(chiral_i - chiral_j))
        """
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
        
        edges = []
        num_nodes = len(self.nodes)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                sim = sim_matrix[i, j].item()
                
                if sim > self.edge_threshold:
                    # Chiral Interference factor
                    c_i = self.nodes[i].chiral_score
                    c_j = self.nodes[j].chiral_score
                    
                    # Ensure scalars
                    if isinstance(c_i, torch.Tensor): c_i = c_i.item()
                    if isinstance(c_j, torch.Tensor): c_j = c_j.item()
                    
                    chiral_factor = math.exp(-abs(c_i - c_j))

                    # Tag overlap factor: shared high-weight tags indicate shared topological
                    # lineage (same friction type, archetype, alias) — they pull manifold
                    # distance closer. Uses the same key normalization as SuperposedTagStacker.
                    #
                    # SIGN AWARENESS: SuperposedTagStacker allows negative weights (feature
                    # subtraction / hyperbolic divergence). A negative * negative product is
                    # geometrically a divergence, not an attraction. We use max(overlap, 0)
                    # so only genuine positive co-activation increases edge affinity.
                    tw_i = self.nodes[i].tag_weights
                    tw_j = self.nodes[j].tag_weights
                    if tw_i and tw_j:
                        shared_keys = set(tw_i) & set(tw_j)
                        overlap = sum(tw_i[k] * tw_j[k] for k in shared_keys)
                        norm_i = math.sqrt(sum(v ** 2 for v in tw_i.values())) + 1e-8
                        norm_j = math.sqrt(sum(v ** 2 for v in tw_j.values())) + 1e-8
                        # clamp to [0, 1]: only positive co-activation amplifies affinity
                        signed_cosine = overlap / (norm_i * norm_j)
                        tag_overlap_factor = 1.0 + 0.3 * max(signed_cosine, 0.0)
                    else:
                        tag_overlap_factor = 1.0

                    weight = float(sim * chiral_factor * tag_overlap_factor)
                    
                    edges.append({
                        "source": str(self.nodes[i].node_id),
                        "target": str(self.nodes[j].node_id),
                        "weight": weight,
                        "type": "RESONANCE" if weight > 0.8 else "PROXIMITY"
                    })
                    
        return edges

    def get_memory_snapshot(self) -> Dict[str, Any]:
        """
        Capture the live Neglecton state as a binary-compatible snapshot.
        Includes node states (tensors), metadata, and resonance parameters.
        """
        # We store the core tensors and the text residues
        fossil_data = []
        for n in self.nodes:
            fossil_data.append({
                "node_id": n.node_id,
                "state": n.state.detach().cpu(), # The embedding seed
                "text": n.text,
                "metrics": n.metrics
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
        for node_data in snapshot.get("nodes", []):
            node = KnowledgeFossilNode(
                node_id=node_data["node_id"],
                state=node_data["state"],
                text=node_data["text"],
                metrics=node_data["metrics"]
            )
            self.nodes.append(node)
            
        print(f"[RECOVERY] Fossilized graph restored: {len(self.nodes)} nodes injected into the Neglecton.")

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
        for n in self.nodes:
            # Stats for Client Visualization
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
                "tag_weights": n.tag_weights  # Dict[str, float] for client-side shading
            })

        return json.dumps({"nodes": nodes_data, "links": edges})

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
