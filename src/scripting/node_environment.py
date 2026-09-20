import threading
import logging
from typing import Dict, Any, Callable
import dearpygui.dearpygui as dpg

logger = logging.getLogger(__name__)

class PhysicalNodeEditor:
    """
    Dedicated DearPyGui Node Environment for Physical Scripting.
    Implements Rust-style object-orientedness (nodes as instances)
    and Virtual Links (borrowed state passing).
    """
    def __init__(self, patch_state=None):
        self.patch_state = patch_state
        self.running = False
        self.thread = None
        
        # Virtual Links and Sidechain parameters
        self.virtual_links = []
        self.node_registry = {}
        self.eval_thread = None
        self.last_latent_dim = None
        self.last_links_count = None

    def _link_callback(self, sender, app_data):
        """Virtual link creation between nodes."""
        dpg.add_node_link(app_data[0], app_data[1], parent=sender)
        self.virtual_links.append((app_data[0], app_data[1]))
        logger.info(f"[Node Editor] Virtual link established: {app_data[0]} -> {app_data[1]}")

    def _delink_callback(self, sender, app_data):
        """Virtual link deletion."""
        dpg.delete_item(app_data)
        self.virtual_links = [l for l in self.virtual_links if l != app_data]

    def _setup_nodes(self):
        with dpg.window(label="Gyroidic Node Scripting", width=800, height=600):
            with dpg.node_editor(callback=self._link_callback, delink_callback=self._delink_callback, id="node_editor"):
                # Rust-style BSpline Node (Generator)
                with dpg.node(label="BSpline Mod Generator", tag="node_bspline"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_bspline_out"):
                        dpg.add_text("BSpline Tensor Out")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Latent Dim", default_value=3, min_value=1, max_value=10, tag="slider_latent_dim")
                        dpg.add_slider_int(label="Resolution", default_value=20, min_value=5, max_value=100, tag="slider_resolution")
                        
                # Dark Matter Attractor Node
                with dpg.node(label="Dark Matter Attractor", tag="node_dark_matter"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_dark_matter_out"):
                        dpg.add_text("Dark Matter Fossil Out")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_int(label="Iterations", default_value=500, min_value=100, max_value=2000, tag="slider_iterations")
                        dpg.add_slider_float(label="Time Step", default_value=0.05, min_value=0.01, max_value=0.2, tag="slider_dt")
                
                # Physical Parameters Node
                with dpg.node(label="Global Physical Properties", tag="node_physics"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="Mass Cost Mod", default_value=1.0, min_value=0.1, max_value=5.0, tag="slider_mass_cost")
                        dpg.add_slider_float(label="Topological Persistence", default_value=0.5, min_value=0.1, max_value=1.0, tag="slider_topo_persist")
                        dpg.add_slider_float(label="Resonance Frequency (Hz)", default_value=432.0, min_value=1.0, max_value=1000.0, tag="slider_resonance")
                        dpg.add_slider_float(label="Quantum Tunnel Prob", default_value=0.05, min_value=0.0, max_value=1.0, tag="slider_quantum_tunnel")

                # Texture DSP Node (Music Filtering style)
                with dpg.node(label="Texture DSP (Spectral)", tag="node_texture_dsp"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="attr_texture_in"):
                        dpg.add_text("Texture In")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                        dpg.add_slider_float(label="Cutoff Freq", default_value=0.5, min_value=0.01, max_value=1.0, tag="slider_cutoff")
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, tag="attr_texture_out"):
                        dpg.add_text("Filtered Texture Out")
                        
                # Voxelboxter Sink Node
                with dpg.node(label="Voxelboxter Graph Sink", tag="node_sink"):
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, tag="attr_sink_in"):
                        dpg.add_text("Compiled Mod In")

    def _evaluation_worker(self):
        """
        D-Wave Collective Computing Pool
        Asynchronously evaluates the graph to prevent DSP processing from
        blocking the UI thread. Educated by left-of-field SOMA mechanics.
        """
        import time
        import torch
        from src.core.structural_blueprints import MangostienBSplineMod, DarkMatterAttractorLayer
        from src.core.honest_jitter import harvest_honest_jitter
        from src.core.hardware_monitor import has_headroom
        from src.p2p.bonfire_consensus import BonfireNomadicRing
        from src.p2p.freenet_ws_client import FreenetClient
        from src.p2p.zk_aggregator import ZKAggregator
        from src.surrogates.calm_predictor import CALM

        # Instantiate the Bonfire P2P Node
        try:
            dummy_freenet = FreenetClient(ws_url="ws://localhost:8080/bonfire")
            bonfire = BonfireNomadicRing(freenet_client=dummy_freenet)
        except Exception:
            bonfire = None
            
        zk_agg = ZKAggregator()
        calm = CALM(dim=8) # 8 dimensions corresponding to our 8 slider parameters
        
        # Buffer for CALM trajectory tracking
        history_buffer = torch.zeros(1, 8, 8) 
        
        while self.running:
            # Replaced arbitrary time clock with Freenet/Consensus polling
            # We wait until hardware headroom is available for the next phase
            if not has_headroom():
                time.sleep(0.5)
                continue
                
            time.sleep(0.5) 
            
            try:
                latent_dim = dpg.get_value("slider_latent_dim")
                resolution = dpg.get_value("slider_resolution")
                iterations = dpg.get_value("slider_iterations")
                dt = dpg.get_value("slider_dt")
                mass_cost = dpg.get_value("slider_mass_cost")
                topo_persist = dpg.get_value("slider_topo_persist")
                resonance = dpg.get_value("slider_resonance")
                quantum_tunnel = dpg.get_value("slider_quantum_tunnel")
                current_links = list(self.virtual_links)
            except Exception:
                continue
                
            state_tuple = (latent_dim, resolution, iterations, dt, mass_cost, topo_persist, resonance, quantum_tunnel, len(current_links))
            
            if not hasattr(self, 'last_state_tuple') or self.last_state_tuple != state_tuple:
                self.last_state_tuple = state_tuple
                
                # Ouroboros Meditation / CALM Veto logic
                current_state_tensor = torch.tensor([[latent_dim, resolution, iterations, dt, mass_cost, topo_persist, resonance, quantum_tunnel]])
                history_buffer = calm.update_buffer(history_buffer, current_state_tensor)
                
                # Check for "Play" vs "Panic"
                abort_score, _, _, _, _, _ = calm(history_buffer, h_mischief=0.9, dt=dt)
                
                if abort_score.item() > 0.8 and not getattr(calm, 'meditation_active', False):
                    logger.warning(f"[CALM] Trajectory vetoed (score: {abort_score.item():.2f}). Entropic collapse detected. Aborting compile.")
                    continue
                
                # D-Wave Consensus: Fetch Kelly fraction from P2P Bonfire ring
                # Use it to modulate the mass cost. If consensus is low, mass cost increases to hedge risk.
                consensus_kelly = 1.0
                if bonfire:
                    consensus_kelly = bonfire.compute_egalitarian_consensus()
                effective_mass_cost = mass_cost * (2.0 - consensus_kelly)
                
                connected_bspline = False
                connected_dark_matter = False
                for link in current_links:
                    if link[0] == "attr_bspline_out" and link[1] == "attr_sink_in":
                        connected_bspline = True
                    elif link[0] == "attr_dark_matter_out" and link[1] == "attr_sink_in":
                        connected_dark_matter = True
                        
                if self.patch_state:
                    with self.patch_state.lock:
                        # Clear existing node mods
                        self.patch_state.routine.layers = [l for l in self.patch_state.routine.layers if not getattr(l, 'name', '').startswith("NodeCompiledMod_")]
                        
                        # Generate ID via hardware jitter, not mathematically empty PRNG
                        jitter_id = str(harvest_honest_jitter(torch.Size([1])).item())
                        
                        if connected_bspline:
                            # Prove Chern-Simons invariant before injection
                            dummy_gauge = torch.zeros(1)
                            proof = zk_agg.prove_chern_simons_invariant(current_state_tensor, dummy_gauge)
                            if not zk_agg.verify_proof("chern_simons", proof):
                                logger.error("[ZKAggregator] Topological leak detected. Refusing BSpline injection.")
                                continue
                                
                            logger.info(f"[D-Wave] Compiling BSpline layer (Dim: {latent_dim}, Res: {resolution})...")
                            new_layer = MangostienBSplineMod(f"NodeCompiledMod_BSpline_{jitter_id}", latent_dim=int(latent_dim), resolution=int(resolution))
                            new_layer.settings.mass_cost_modifier = effective_mass_cost
                            new_layer.settings.topological_persistence = topo_persist
                            new_layer.settings.resonance_frequency = resonance
                            new_layer.settings.quantum_tunnel_prob = quantum_tunnel
                            self.patch_state.routine.layers.append(new_layer)
                            self.patch_state.graph.dirty = True

                        if connected_dark_matter:
                            proof = zk_agg.prove_chern_simons_invariant(current_state_tensor, torch.zeros(1))
                            if not zk_agg.verify_proof("chern_simons", proof):
                                logger.error("[ZKAggregator] Topological leak detected. Refusing Dark Matter injection.")
                                continue
                                
                            logger.info(f"[D-Wave] Compiling Dark Matter Attractor (Iter: {iterations}, dt: {dt})...")
                            new_layer = DarkMatterAttractorLayer(f"NodeCompiledMod_DarkMatter_{jitter_id}", iterations=int(iterations), dt=dt)
                            new_layer.settings.mass_cost_modifier = effective_mass_cost
                            new_layer.settings.topological_persistence = topo_persist
                            new_layer.settings.resonance_frequency = resonance
                            new_layer.settings.quantum_tunnel_prob = quantum_tunnel
                            self.patch_state.routine.layers.append(new_layer)
                            self.patch_state.graph.dirty = True
                        
    def _run_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title='Voxelboxter - Physical Scripting Console', width=800, height=600)
        dpg.setup_dearpygui()
        
        self._setup_nodes()
        
        dpg.show_viewport()
        
        while dpg.is_dearpygui_running() and self.running:
            dpg.render_dearpygui_frame()
            # Sidechain hook: sync parameters back to Voxelboxter state here
            
        dpg.destroy_context()
        self.running = False

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_dpg, daemon=True)
            self.thread.start()
            
            # Start D-Wave Worker
            self.eval_thread = threading.Thread(target=self._evaluation_worker, daemon=True)
            self.eval_thread.start()
            
            logger.info("[Physical Scripting] DearPyGui separate window and D-Wave Evaluation Pool activated.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.eval_thread:
            self.eval_thread.join(timeout=2.0)
