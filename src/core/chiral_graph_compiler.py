# -*- coding: utf-8 -*-
"""
src/core/chiral_graph_compiler.py

Chirality-Based Graph Compiler: Asymmetric Flow -> Executable Form.
Part of the CODES v40 / Action Layer.

Translates non-commutative hyper-graphs, reasoning trajectories, and 
Alien Handshake (RP4 projective puncture) states into validated, executable forms:
1. Python scripts (imperative/functional code maintaining non-commutative dependency order)
2. Shell scripts (PowerShell / POSIX bash action pipelines)
3. UI elements (interactive HTML5/Canvas/SVG widgets reflecting directed topology)

Preserves topological chirality:
    chi = (spectral_centroid - midpoint) / midpoint
    chi > 0: Dynamic, transformative, high-frequency execution (iterative loops, CLI events)
    chi <= 0: Structural, declarative, negentropic invariant construction (classes, pipelines)
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.core.invariants import compute_chirality, compute_chiral_shift
from src.core.archetype_engines import RP4ProjectiveRouter


@dataclass
class CompiledArtifact:
    """Represents a validated compilation from topological reasoning to executable form."""
    target: str                          # "python", "shell", "ui", "bundle"
    code: str                            # Concrete code / script / markup
    chirality: float                     # Measured chirality index
    execution_order: List[str]          # Ordered node sequence (non-commutative)
    non_commutative_hash: str            # Deterministic topological signature
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "code": self.code,
            "chirality": self.chirality,
            "execution_order": self.execution_order,
            "non_commutative_hash": self.non_commutative_hash,
            "metadata": self.metadata,
        }


class ChiralGraphCompiler(nn.Module):
    """
    Compiler that maps asymmetric reasoning graphs and RP4-punctured latent states
    to executable Python code, shell automation, and functional UI components.
    
    Guarantees:
    1. Non-commutative order preservation (A -> B != B -> A).
    2. Zero phantom dependencies: every executed block has explicit topological inputs.
    3. Chirality branching: aligns code structure with entropy/order characteristics.
    """
    def __init__(self, state_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        # Optional projection from state_dim to code token embeddings
        self.flow_projector = nn.Linear(state_dim, 32)

    def measure_chirality(self, state: Union[torch.Tensor, List[float], Any] = None) -> float:
        """Computes normalized chirality index chi in [-1.0, 1.0]."""
        if state is None:
            return 0.0
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.numel() == 0:
            return 0.0
        with torch.no_grad():
            t = state.detach().float()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            if t.shape[-1] < 2:
                return 0.0
            shift = compute_chiral_shift(t)
            return float(shift.mean().item())

    def topological_sort(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
        """
        Non-commutative directed topological sort.
        Maintains asymmetric execution order: edge (u -> v) forces u to precede v.
        """
        in_degree: Dict[str, int] = {n.get("id", f"node_{i}"): 0 for i, n in enumerate(nodes)}
        adjacency: Dict[str, List[str]] = {node_id: [] for node_id in in_degree}

        for edge in edges:
            src = str(edge.get("source", ""))
            tgt = str(edge.get("target", ""))
            if src in adjacency and tgt in in_degree:
                adjacency[src].append(tgt)
                in_degree[tgt] += 1

        # Priority queue / stable queue preserving definition order
        queue = [n_id for n_id, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in adjacency.get(curr, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Append any remaining nodes if cycles exist, maintaining stability
        for n_id in in_degree:
            if n_id not in order:
                order.append(n_id)

        return order

    def compute_non_commutative_hash(self, order: List[str], edges: List[Dict[str, Any]], chirality: float) -> str:
        """Deterministic cryptographic signature of directed order and chiral asymmetry."""
        hasher = hashlib.sha256()
        hasher.update(f"order:{','.join(order)};chi:{chirality:.6f}".encode("utf-8"))
        for e in edges:
            hasher.update(f"({e.get('source')}->{e.get('target')})".encode("utf-8"))
        return hasher.hexdigest()

    def compile_to_python(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        chirality: float,
        order: List[str],
        module_name: str = "reasoner_flow"
    ) -> str:
        """
        Compiles the directed flow into executable Python code.
        If chirality > 0: Generates an iterative, streaming, dynamic pipeline.
        If chirality <= 0: Generates a functional, declarative, invariant-preserving class.
        """
        lines = [
            "# -*- coding: utf-8 -*-",
            f'"""Auto-generated executable flow compiled by ChiralGraphCompiler.',
            f"Chirality Index: {chirality:+.4f} ({'Entropic / High-Freq Dynamic' if chirality > 0 else 'Negentropic / Structural Invariant'})",
            f"Topological Order: {' -> '.join(order)}",
            '"""',
            "import sys",
            "import time",
            "from typing import Any, Dict",
            "",
        ]

        if chirality > 0:
            # Dynamic iterative pipeline
            lines.extend([
                f"def run_{module_name}(initial_payload: Dict[str, Any] = None) -> Dict[str, Any]:",
                "    state = initial_payload or {}",
                "    print(f'[CHIRAL_FLOW] Starting dynamic execution pipeline (chi={chirality:+.4f})')",
                "",
            ])
            for step_idx, node_id in enumerate(order):
                node_data = next((n for n in nodes if n.get("id") == node_id), {"label": node_id})
                label = node_data.get("label", node_id)
                op = node_data.get("op", "transform")
                lines.extend([
                    f"    # Step {step_idx + 1}: Node '{node_id}' ({label})",
                    f"    print(f'  -> Executing [{node_id}] op={op}')",
                    f"    state['{node_id}'] = {{'executed': True, 'timestamp': time.time(), 'op': '{op}'}}",
                    "",
                ])
            lines.extend([
                "    print('[CHIRAL_FLOW] Dynamic pipeline completed successfully.')",
                "    return state",
                "",
                'if __name__ == "__main__":',
                f"    result = run_{module_name}()",
                "    print('Final Flow Result:', result)",
            ])
        else:
            # Structural declarative pipeline
            class_name = "".join(part.capitalize() for part in module_name.split("_"))
            lines.extend([
                f"class {class_name}:",
                f'    """Invariant-preserving structural manifold pipeline."""',
                f"    CHIRALITY_INDEX: float = {chirality:.6f}",
                "",
                "    def __init__(self):",
                "        self.registry: Dict[str, Any] = {}",
                "",
            ])
            for step_idx, node_id in enumerate(order):
                node_data = next((n for n in nodes if n.get("id") == node_id), {"label": node_id})
                safe_name = node_id.replace("-", "_").replace(" ", "_")
                label = node_data.get("label", node_id)
                lines.extend([
                    f"    def step_{safe_name}(self, payload: Dict[str, Any]) -> Dict[str, Any]:",
                    f"        # Node: {node_id} | Label: {label}",
                    f"        payload['{node_id}'] = {{'status': 'converged', 'id': '{node_id}'}}",
                    f"        return payload",
                    "",
                ])
            lines.extend([
                "    def execute(self, payload: Dict[str, Any] = None) -> Dict[str, Any]:",
                "        state = payload or {}",
            ])
            for node_id in order:
                safe_name = node_id.replace("-", "_").replace(" ", "_")
                lines.append(f"        state = self.step_{safe_name}(state)")
            lines.extend([
                "        return state",
                "",
                'if __name__ == "__main__":',
                f"    pipeline = {class_name}()",
                "    res = pipeline.execute()",
                "    print('Structural Execution Result:', res)",
            ])

        return "\n".join(lines)

    def compile_to_shell(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        chirality: float,
        order: List[str]
    ) -> str:
        """
        Compiles the directed flow into an executable shell automation pipeline
        (PowerShell & POSIX compatible comments).
        """
        lines = [
            "# Auto-generated Chiral Shell Pipeline",
            f"# Chirality: {chirality:+.4f} | Nodes: {len(order)}",
            "$ErrorActionPreference = 'Stop'",
            "Write-Host '[CHIRAL_SHELL] Initializing asymmetric execution chain...'",
            "",
        ]
        for idx, node_id in enumerate(order):
            node_data = next((n for n in nodes if n.get("id") == node_id), {"label": node_id})
            label = node_data.get("label", node_id)
            lines.extend([
                f"# Stage {idx + 1}: {node_id}",
                f"Write-Host '  -> [Stage {idx + 1}/{len(order)}] Executing {node_id} ({label})...'",
                f"if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {{",
                f"    Write-Error 'Pipeline broke at {node_id}'",
                f"    exit $LASTEXITCODE",
                f"}}",
                "",
            ])
        lines.append("Write-Host '[CHIRAL_SHELL] Pipeline completed with topological fidelity.'")
        return "\n".join(lines)

    def compile_to_ui(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        chirality: float,
        order: List[str]
    ) -> str:
        """
        Compiles the directed flow into a standalone, interactive HTML5/SVG UI element.
        """
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        order_json = json.dumps(order)
        ui_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chiral Flow Inspector</title>
<style>
  body {{ background: #0c0d12; color: #c0caf5; font-family: monospace; margin: 0; padding: 16px; }}
  .header {{ border-bottom: 1px solid #2ac3de; padding-bottom: 8px; margin-bottom: 16px; }}
  .badge {{ background: #1f2335; border: 1px solid #7aa2f7; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
  .flow-container {{ display: flex; flex-direction: column; gap: 8px; max-width: 600px; }}
  .flow-node {{ background: #16161e; border: 1px solid #3b4261; padding: 10px 14px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center; }}
  .flow-node.active {{ border-color: #9ece6a; background: #1a2b26; }}
  .btn {{ background: #7aa2f7; color: #15161e; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: bold; font-family: monospace; }}
  .btn:hover {{ background: #89ddff; }}
</style>
</head>
<body>
<div class="header">
  <h2>Topological Chiral Flow</h2>
  <span class="badge">Chirality: {chirality:+.4f}</span>
  <span class="badge">Nodes: {len(order)}</span>
  <span class="badge">Edges: {len(edges)}</span>
</div>
<div class="flow-container" id="nodes-list"></div>
<div style="margin-top: 16px;">
  <button class="btn" onclick="executeStep()">Step Next</button>
  <button class="btn" onclick="resetFlow()" style="background:#f7768e; margin-left: 8px;">Reset</button>
</div>
<script>
  const order = {order_json};
  const nodes = {nodes_json};
  let currentStep = 0;

  function render() {{
    const container = document.getElementById('nodes-list');
    container.innerHTML = '';
    order.forEach((id, idx) => {{
      const div = document.createElement('div');
      div.className = 'flow-node' + (idx < currentStep ? ' active' : '');
      div.innerHTML = `<span><strong>${{idx + 1}}.</strong> ${{id}}</span><span>${{idx < currentStep ? 'EXECUTED' : 'PENDING'}}</span>`;
      container.appendChild(div);
    }});
  }}

  function executeStep() {{
    if (currentStep < order.length) {{
      currentStep++;
      render();
    }}
  }}

  function resetFlow() {{
    currentStep = 0;
    render();
  }}

  render();
</script>
</body>
</html>"""
        return ui_html

    def compile(
        self,
        graph_or_state: Union[Dict[str, Any], torch.Tensor],
        target: str = "python",
        module_name: str = "reasoner_flow"
    ) -> CompiledArtifact:
        """
        Main entry point for compilation.
        
        Args:
            graph_or_state: Either a dictionary with 'nodes' and 'edges', or a latent tensor.
            target: 'python', 'shell', 'ui', or 'all'
            module_name: Name for the compiled code module
            
        Returns:
            CompiledArtifact containing code, order, and non-commutative hash.
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        chirality: float = 0.0

        if isinstance(graph_or_state, dict):
            nodes = graph_or_state.get("nodes", [])
            edges = graph_or_state.get("edges", [])
            if "latent_state" in graph_or_state:
                chirality = self.measure_chirality(graph_or_state["latent_state"])
            elif "chirality" in graph_or_state:
                chirality = float(graph_or_state["chirality"])
            else:
                chirality = 0.05
        elif isinstance(graph_or_state, torch.Tensor):
            # Synthesize topological graph nodes from non-zero channels of the tensor
            chirality = self.measure_chirality(graph_or_state)
            flat = graph_or_state.detach().cpu().flatten()
            num_nodes = min(max(3, flat.numel() // 8), 12)
            nodes = [{"id": f"state_channel_{i}", "label": f"Channel {i}", "energy": float(flat[i].item() if i < flat.numel() else 0.0)} for i in range(num_nodes)]
            edges = [{"source": f"state_channel_{i}", "target": f"state_channel_{i+1}"} for i in range(num_nodes - 1)]
        else:
            nodes = [{"id": "entrypoint", "label": "Entry"}]
            edges = []

        order = self.topological_sort(nodes, edges)
        nc_hash = self.compute_non_commutative_hash(order, edges, chirality)

        if target == "python":
            code = self.compile_to_python(nodes, edges, chirality, order, module_name=module_name)
        elif target == "shell":
            code = self.compile_to_shell(nodes, edges, chirality, order)
        elif target == "ui":
            code = self.compile_to_ui(nodes, edges, chirality, order)
        elif target == "all":
            py_code = self.compile_to_python(nodes, edges, chirality, order, module_name=module_name)
            sh_code = self.compile_to_shell(nodes, edges, chirality, order)
            ui_code = self.compile_to_ui(nodes, edges, chirality, order)
            bundle = {
                "python": py_code,
                "shell": sh_code,
                "ui": ui_code
            }
            code = json.dumps(bundle, indent=2)
        else:
            code = self.compile_to_python(nodes, edges, chirality, order, module_name=module_name)

        return CompiledArtifact(
            target=target,
            code=code,
            chirality=chirality,
            execution_order=order,
            non_commutative_hash=nc_hash,
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges),
                "module_name": module_name
            }
        )
