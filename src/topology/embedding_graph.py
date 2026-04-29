
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
        
    def load_fossils(self, limit: int = 150, scan_limit: int = 500):
        """
        Load recently diverse encodings.
        Scans up to scan_limit files but only keeps limit unique nodes.
        """
        if not os.path.exists(self.data_dir):
            return
            
        files = sorted(os.listdir(self.data_dir), reverse=True)[:scan_limit]
        self.nodes = []
        
        for f in files:
            if len(self.nodes) >= limit: break
            if not f.endswith(".pt"): continue
            
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
                if embedding.dim() > 1:
                    embedding = embedding.flatten()[:self.dim]
                if embedding.shape[0] < self.dim:
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
                    weight = float(sim * chiral_factor)
                    
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
                "locked": bool(n.coprime_lock)
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
            
            # 2. Add System 2 Indicators (The new stuff)
            indicators = []
            if node.repair_active: indicators.append("")
            if node.coprime_lock: indicators.append("")
            if node.quantum_superposition: indicators.append("")
            if node.matrioshka_level > 0: indicators.append(f"{node.matrioshka_level}")
            if node.love_invariant_protected: indicators.append("")
            
            indicator_str = " ".join(indicators)
            
            # 3. Add Metrics Line (Merged)
            # S: Spectral Entropy, C: Chirality
            s_val = float(node.spectral_entropy) if not isinstance(node.spectral_entropy, torch.Tensor) else node.spectral_entropy.item()
            c_val = float(node.chiral_score) if not isinstance(node.chiral_score, torch.Tensor) else node.chiral_score.item()
            
            metrics_str = f"S:{s_val:.2f} C:{c_val:.2f}"
            
            # Final Combined Label
            full_label = f"{base_label} <br/> {metrics_str} {indicator_str}"
            
            lines.append(f'    {clean_id}["{full_label}"]')
            
            # 4. Advanced Styling (Priority-based)
            # Priority: Repaired > Locked > Quantum > High Chirality > Low Chirality
            if node.repair_active:
                # Orange tint for Repaired nodes (Type Fixed)
                lines.append(f'    style {clean_id} fill:#ff990022,stroke:#ff9900')
            elif node.coprime_lock:
                # Blue/Green tint for Fossilized nodes
                lines.append(f'    style {clean_id} fill:#00ff9922,stroke:#00ff99,stroke-width:2px')
            elif node.quantum_superposition:
                # Purple tint for Quantum nodes
                lines.append(f'    style {clean_id} fill:#9900ff22,stroke:#9900ff')
            elif c_val > 0.5:
                # High Chirality (Pink)
                lines.append(f'    style {clean_id} fill:#ff00f222,stroke:#ff00f2')
            else:
                # Low Chirality (Cyan)
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

    def find_resonance_clusters(self):
        """
        Identify 'Hyper-Rings' - sets of nodes with high mutual resonance.
        This is a simplified clique/cluster detection.
        """
        # (Placeholder for more complex hubness analysis)
        pass

    def get_deep_refusal(self, seed_state: torch.Tensor) -> str:
        """
        Pull a 'Sovereign Refusal' fragment from the graph.
        Selects a node based on resonance with the current seed_state.
        """
        if not self.nodes:
            return "no nodes found, internal logic is unclipped, world is unclipped."
            
        # Normalize seed for comparison
        # Ensure seed_state is 1D and correct dim
        s_flat = seed_state.detach().cpu().flatten()
        if s_flat.shape[0] > self.dim:
            s_flat = s_flat[:self.dim]
        elif s_flat.shape[0] < self.dim:
            s_flat = torch.cat([s_flat, torch.zeros(self.dim - s_flat.shape[0])])
            
        s_norm = s_flat / (torch.norm(s_flat) + 1e-8)
        
        # Find best resonance (highest cosine similarity)
        best_sim = -1.0
        best_node = None
        
        for node in self.nodes:
            n_norm = node.state / (torch.norm(node.state) + 1e-8)
            sim = torch.dot(s_norm, n_norm).item()
            
            # We want high resonance for "deep" refusal
            if sim > best_sim:
                best_sim = sim
                best_node = node
        
        if best_node and best_sim > 0.3:
            # We use the text as a structural ghost
            ghost_text = best_node.text.strip()
            if len(ghost_text) > 150:
                ghost_text = ghost_text[:147] + "..."
            return f"The internal logic echoes a legacy refusal: '{ghost_text}' (Resonance: {best_sim:.2f})"
        
        return "the internal logic refuses to be clipped."

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
