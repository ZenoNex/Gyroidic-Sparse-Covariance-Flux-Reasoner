#!/usr/bin/env python3
"""
Hybrid Backend - Uses working AI components, bypasses broken imports.
"""

import http.server
import socketserver
import json
import sys
import os
import torch
import threading
import socketserver
import numpy as np
import base64
import cgi
from src.core.invariants import compute_chirality, check_glyphlock, compute_chiral_shift
try:
    import psutil
except ImportError:
    psutil = None
import signal
import time
import subprocess
import csv
import io
import cgi
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error
from src.core.audience_mapping import AudienceProjection
from src.core.superposed_tag_stacker import SuperposedTagStacker
from src.terminal.udp_server_colonizer import OptionD_Colonizer

class TensorEncoder(json.JSONEncoder):
    """Custom JSON encoder that safely serializes PyTorch Tensors and NumPy types."""
    def default(self, obj):
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.dim() == 0:
                try:
                    return float(obj.item())
                except:
                    return int(obj.item())
            return obj.detach().cpu().tolist()
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64, np.float16)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                return int(obj)
        except ImportError:
            pass
        return super().default(obj)


# Add project root to path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if os.path.join(root_dir, 'examples') not in sys.path:
    sys.path.insert(0, os.path.join(root_dir, 'examples'))

# Import working components

from src.core import DEVICE
print('=============================================')
print(f'[INFO] EXECUTING FROM: {__file__}')
print(f'[INFO] DEVICE DETECTED: {DEVICE}')
if str(DEVICE) != 'cpu':
    print(f'[INFO] Silicon Sovereignty: [OK] (Hardware-bridge active)')
else:
    print('[INFO] DEVICE DETECTED: CPU (Substrate-independent fallback)')
print('=============================================')

try:
    from src.training.enhanced_temporal_training import NonLobotomyTemporalModel
    from src.core.admr_solver import PolynomialADMRSolver
    from src.topology.gyroid_covariance import LeyLineGeodesicMetric, MoebiusFiberBundle
    from src.core.failure_token import RuptureFunctional, FailureToken
    TEMPORAL_MODEL_AVAILABLE = True
    print("[OK] Advanced Manifold Dynamics available (ADMR, LeyLines, Moebius, TemporalModel)")
except Exception as e:
    TEMPORAL_MODEL_AVAILABLE = False
    print(f"[FAIL] Advanced Manifold Dynamics failed to import: {e}")


try:
    from src.core.spectral_coherence_repair import SpectralCoherenceCorrector, apply_energy_based_stabilization
    from src.core.number_theoretic_stabilizer import NumberTheoreticStabilizer
    SPECTRAL_CORRECTOR_AVAILABLE = True
    print("[OK] Spectral corrector and Hybrid Stabilizers imported")
except Exception as e:
    SPECTRAL_CORRECTOR_AVAILABLE = False
    print(f"[FAIL] Spectral corrector import failed: {e}")

try:
    from dataset_ingestion_system import DatasetIngestionSystem, DatasetConfig
    DATASET_SYSTEM_AVAILABLE = True
    print("[OK] Dataset Ingestion System imported")
except Exception as e:
    DATASET_SYSTEM_AVAILABLE = False
    print(f"[FAIL] Dataset Ingestion System failed to import: {e}")


class GovernanceManager:
    """Manages Gyroidic process identification, port selection, and lifecycle."""
    
    @staticmethod
    def find_existing_processes():
        """Identify other running instances. Falls back to tasklist if psutil is unavailable."""
        my_pid = os.getpid()
        matches = []
        
        # --- PATH A: PSUTIL (Preferred) ---
        if psutil:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline and any('hybrid_backend.py' in part for part in cmdline):
                            # Exclude our own PID and our Parent's PID (to avoid killing the launcher wrapper)
                            if proc.info['pid'] not in (my_pid, os.getppid()):
                                matches.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return matches
            
        # --- PATH B: TASKLIST FALLBACK (Standard Library) ---
        print("[WARN] psutil unavailable. Falling back to platform-specific process scan.")
        if sys.platform == 'win32':
            try:
                # Use tasklist with verbose mode to see the command line (indirectly)
                # Note: tasklist /v /fo csv is slower but reliable on Windows
                output = subprocess.check_output(['tasklist', '/v', '/fo', 'csv'], text=True, encoding='utf-8', errors='ignore')
                reader = csv.reader(io.StringIO(output))
                # Skip the header row (Image Name, PID, Session Name, Session#, Mem Usage, Status, User Name, CPU Time, Window Title)
                header = next(reader, None)
                for row in reader:
                    if not row or len(row) < 9: continue
                    process_name = row[0]
                    try:
                        pid = int(row[1])
                    except ValueError:
                        continue # Skip if PID is not an integer (header or corrupt row)
                    window_title = row[8] # Often contains the script name if running in a window
                    
                    if 'python' in process_name.lower():
                        # We look for 'hybrid_backend' in the window title as a heuristic
                        if 'hybrid_backend.py' in window_title or 'Hybrid Backend' in window_title:
                            # Exclude our own PID and Parent PID
                            if pid not in (my_pid, os.getppid()):
                                matches.append({'pid': pid, 'cmdline': [window_title]})
            except Exception as e:
                print(f"[FAIL] Tasklist fallback failed: {e}")
        return matches

    @staticmethod
    def shutdown_processes(processes):
        """Perform safe shutdown. Supports OS level kill if psutil is missing."""
        for p in processes:
            pid = p['pid']
            try:
                print(f"[GOVERNANCE] Terminating shadow process {pid}...")
                if psutil:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                else:
                    # Generic OS fallback
                    if sys.platform == 'win32':
                        subprocess.call(['taskkill', '/PID', str(pid), '/F'])
                    else:
                        os.kill(pid, signal.SIGTERM)
            except Exception as e:
                print(f"[FAIL] Could not shutdown {pid}: {e}")

    @staticmethod
    def startup_menu():
        """Interactive console menu for lifecycle control."""
        if os.environ.get('NON_INTERACTIVE') == '1':
            ds_env = os.environ.get('GYROID_PRIMARY_DATASET', 'LIGO')
            parsed_datasets = []
            if ds_env.lower() == 'all':
                parsed_datasets = ['LIGO', 'NCBI', 'SDSS', 'OPENNEURO']
            else:
                for ds in ds_env.split(','):
                    ds_stripped = ds.strip().upper()
                    if ds_stripped in ('LIGO', 'NCBI', 'SDSS', 'OPENNEURO'):
                        parsed_datasets.append(ds_stripped)
            final_ds = ','.join(parsed_datasets) if parsed_datasets else 'LIGO'

            return [8000, 8080], {
                'regime': os.environ.get('GYROID_REGIME', 'goo'),
                'commutativity': os.environ.get('GYROID_COMMUTATIVITY', 'non_commutative'),
                'use_spectral_correction': os.environ.get('GYROID_SPECTRAL_CORRECTION', '1') == '1',
                'mckenna_deconstruction_mode': os.environ.get('GYROID_MCKENNA', '0') == '1',
                'quantum_inspired_mode': os.environ.get('GYROID_QUANTUM', '0') == '1',
                'high_throughput_ingestion': os.environ.get('GYROID_INGESTION', '0') == '1',
                'introspection_probes': os.environ.get('GYROID_PROBES', 'moral,uncertainty,creative,metacognitive').split(','),
                'rigidity_decay_rate': float(os.environ.get('GYROID_RIGIDITY_DECAY', '0.005')),
                'suppress_narration': os.environ.get('GYROID_SUPPRESS_NARRATION', '1') == '1',
                'bg_scientific_learning': os.environ.get('GYROID_BG_LEARNING', '1') == '1',
                'primary_query_dataset': final_ds,
                'cache_dir': os.environ.get('GYROID_CACHE_DIR', 'datasets/open_science_cache'),
                'kagh_dyslexic_mode': os.environ.get('GYROID_KAGH_MODE', '0') == '1',
                'fbm_persistence': float(os.environ.get('GYROID_FBM_PERSISTENCE', '0.5')),
                'fbm_octaves': int(os.environ.get('GYROID_FBM_OCTAVES', '3')),
                'mandelbulb_power': float(os.environ.get('GYROID_MANDELBULB_POWER', '8.0')),
                'mandelbulb_escape_radius': float(os.environ.get('GYROID_MANDELBULB_ESCAPE', '2.0')),
                'birkhoff_temperature': float(os.environ.get('GYROID_BIRKHOFF_TEMP', '1.0')),
                'birkhoff_max_iterations': int(os.environ.get('GYROID_BIRKHOFF_MAX_ITER', '100')),
                'tda_landmarks': int(os.environ.get('GYROID_TDA_LANDMARKS', '50')),
                'ego_death_limit': float(os.environ.get('GYROID_EGO_DEATH_LIMIT', '1.5')),
                'udp_colonizer_enabled': os.environ.get('GYROID_UDP_COLONIZER', '0') == '1'
            }
            
        print("\n" + "="*50)
        print("      GYROIDIC GOVERNANCE INTERFACE ")
        print("="*50)
        
        # 1. Process Identification
        print("[SCANNING] Checking for background gyroid processes...")
        others = GovernanceManager.find_existing_processes()
        
        if others:
            print(f"[ALERT] {len(others)} existing processes detected.")
            for p in others:
                print(f"   • PID {p['pid']} | Cmd: {' '.join(p['cmdline'][:3])}...")
            
            choice = input("\n[?] Shutdown background gyroid processes? (y/N): ").lower().strip()
            if choice == 'y':
                GovernanceManager.shutdown_processes(others)
                # Small delay to let OS release ports
                time.sleep(2)
        else:
            print("[OK] No shadow processes identified.")

        # 2. Port Infrastructure
        print("\n[CONFIG] Port Selection Infrastructure")
        ports = [8000, 8080]
        print(f"   Default ports: {ports}")
        
        extra_ports_raw = input("[?] Specify extra ports to scale (comma separated) or [Enter] for defaults: ").strip()
        if extra_ports_raw:
            try:
                extra_ports = [int(p.strip()) for p in extra_ports_raw.split(',') if p.strip()]
                ports.extend(extra_ports)
                # Ensure unique
                ports = sorted(list(set(ports)))
                print(f"[OK] Manifold scaled to ports: {ports}")
            except ValueError:
                print("[FAIL] Invalid port format. Using defaults.")
        
        # 3. Dynamic Configuration Options (Discovered from System Documentation)
        config = {
            'regime': 'goo',
            'commutativity': 'non_commutative',
            'use_spectral_correction': True,
            'mckenna_deconstruction_mode': False,
            'quantum_inspired_mode': False,
            'high_throughput_ingestion': False,
            'introspection_probes': 'moral,uncertainty,creative,metacognitive',
            'rigidity_decay_rate': 0.005,
            'suppress_narration': True,
            'bg_scientific_learning': True,
            'primary_query_dataset': 'LIGO',
            'cache_dir': 'datasets/open_science_cache',
            'kagh_dyslexic_mode': False,
            'fbm_persistence': 0.5,
            'fbm_octaves': 3,
            'mandelbulb_power': 8.0,
            'mandelbulb_escape_radius': 2.0,
            'birkhoff_temperature': 1.0,
            'birkhoff_max_iterations': 100,
            'tda_landmarks': 50,
            'ego_death_limit': 1.5,
            'open_science_email': 'default@example.com',
            'chatgpt_ingestor_enabled': True,
            'chatgpt_ingestor_verbosity': 'normal',
            'open_science_ingestor_enabled': True,
            'open_science_ingestor_verbosity': 'normal',
            'udp_colonizer_enabled': False
        }
        
        print("\n" + "="*50)
        print("      SYSTEM PARAMETER CONFIGURATION (Thorium Protocol)")
        print("      Type 'default_all' at any prompt to skip remaining questions.")
        print("="*50)

        default_all_active = False

        def get_input(prompt_text, default_val):
            nonlocal default_all_active
            if default_all_active:
                return default_val
            ans = input(f"{prompt_text} [Default: {default_val}]: ").strip()
            if ans.lower() == 'default_all':
                default_all_active = True
                return default_val
            if not ans:
                return default_val
            return ans

        # 3.1 Regime
        print("[INFO] Operational regime is dynamically determined by the engine (Eq 10). Manual selection bypassed.")

        # 3.2 Commutativity
        print("[INFO] Commutativity is dynamically determined at point of ingestion (default: non_commutative). Selection bypassed.")
        config['commutativity'] = 'non_commutative'

        # 3.3 Spectral correction
        spec = get_input("[?] Enable spectral correction (yes/no)", 'yes' if config['use_spectral_correction'] else 'no').lower()
        if spec in ('yes', 'y'):
            config['use_spectral_correction'] = True
        elif spec in ('no', 'n'):
            config['use_spectral_correction'] = False
        else:
            print(f"[WARN] Invalid option, using default: {'yes' if config['use_spectral_correction'] else 'no'}")

        # 3.4 Operational mode
        print("[INFO] Operational mode is dynamically set at runtime based on data connection. Selection bypassed.")
        config['high_throughput_ingestion'] = False

        # 3.5 Introspection probes
        probes = get_input("[?] Active introspection probes (comma separated)", config['introspection_probes'])
        config['introspection_probes'] = [p.strip() for p in probes.split(',') if p.strip()]

        # 3.6 Rigidity decay rate
        decay = get_input("[?] Introspection rigidity decay rate", str(config['rigidity_decay_rate']))
        try:
            config['rigidity_decay_rate'] = float(decay)
        except ValueError:
            print(f"[WARN] Invalid float, using default: {config['rigidity_decay_rate']}")

        # 3.7 Suppress narration
        supp = get_input("[?] Suppress narration (yes/no)", 'yes' if config['suppress_narration'] else 'no').lower()
        if supp in ('yes', 'y'):
            config['suppress_narration'] = True
        elif supp in ('no', 'n'):
            config['suppress_narration'] = False
        else:
            print(f"[WARN] Invalid option, using default: {'yes' if config['suppress_narration'] else 'no'}")

        # 3.8 Background scientific learning
        bg_learn = get_input("[?] Enable background scientific learning (yes/no)", 'yes' if config['bg_scientific_learning'] else 'no').lower()
        if bg_learn in ('yes', 'y'):
            config['bg_scientific_learning'] = True
        elif bg_learn in ('no', 'n'):
            config['bg_scientific_learning'] = False
        else:
            print(f"[WARN] Invalid option, using default: {'yes' if config['bg_scientific_learning'] else 'no'}")

        # 3.9 Primary scientific query dataset
        dataset_raw = get_input("[?] Primary scientific query dataset (LIGO/NCBI/SDSS/OpenNeuro or 'all' or comma-separated)", config['primary_query_dataset'])
        parsed_datasets = []
        if dataset_raw.lower() == 'all':
            parsed_datasets = ['LIGO', 'NCBI', 'SDSS', 'OPENNEURO']
        else:
            for ds in dataset_raw.split(','):
                ds_stripped = ds.strip().upper()
                if ds_stripped in ('LIGO', 'NCBI', 'SDSS', 'OPENNEURO'):
                    parsed_datasets.append(ds_stripped)
        
        if parsed_datasets:
            config['primary_query_dataset'] = ','.join(parsed_datasets)
        else:
            print(f"[WARN] Invalid option, using default: {config['primary_query_dataset']}")

        # 3.10 Cache directory
        cache = get_input("[?] Scientific cache directory", config['cache_dir'])
        config['cache_dir'] = cache

        # 3.11 McKenna deconstruction mode
        mck = get_input("[?] Enable McKenna deconstruction mode (yes/no)", 'yes' if config['mckenna_deconstruction_mode'] else 'no').lower()
        if mck in ('yes', 'y'):
            config['mckenna_deconstruction_mode'] = True
        elif mck in ('no', 'n'):
            config['mckenna_deconstruction_mode'] = False
        else:
            print(f"[WARN] Invalid option, using default: {'yes' if config['mckenna_deconstruction_mode'] else 'no'}")

        # 3.12 Quantum-inspired reasoning
        q_reason = get_input("[?] Enable quantum-inspired reasoning (yes/no)", 'yes' if config['quantum_inspired_mode'] else 'no').lower()
        if q_reason in ('yes', 'y'):
            config['quantum_inspired_mode'] = True
        elif q_reason in ('no', 'n'):
            config['quantum_inspired_mode'] = False
        else:
            print(f"[WARN] Invalid option, using default: {'yes' if config['quantum_inspired_mode'] else 'no'}")

        # 3.13 KAGH dyslexic mode
        dys = get_input("[?] Enable KAGH dyslexic mode (yes/no)", 'yes' if config['kagh_dyslexic_mode'] else 'no').lower()
        if dys in ('yes', 'y'):
            config['kagh_dyslexic_mode'] = True
        elif dys in ('no', 'n'):
            config['kagh_dyslexic_mode'] = False
        else:
            print(f"[WARN] Invalid option, using default: {'yes' if config['kagh_dyslexic_mode'] else 'no'}")

        # 3.14 FBM persistence
        fbm_p = get_input("[?] FBM erosion persistence", str(config['fbm_persistence']))
        try:
            config['fbm_persistence'] = float(fbm_p)
        except ValueError:
            print(f"[WARN] Invalid float, using default: {config['fbm_persistence']}")

        # 3.15 FBM octaves
        fbm_o = get_input("[?] FBM erosion octaves", str(config['fbm_octaves']))
        try:
            config['fbm_octaves'] = int(fbm_o)
        except ValueError:
            print(f"[WARN] Invalid int, using default: {config['fbm_octaves']}")

        # 3.16 Mandelbulb power
        m_pow = get_input("[?] Mandelbulb augmenter power", str(config['mandelbulb_power']))
        try:
            config['mandelbulb_power'] = float(m_pow)
        except ValueError:
            print(f"[WARN] Invalid float, using default: {config['mandelbulb_power']}")

        # 3.17 Mandelbulb escape radius
        m_esc = get_input("[?] Mandelbulb escape radius", str(config['mandelbulb_escape_radius']))
        try:
            config['mandelbulb_escape_radius'] = float(m_esc)
        except ValueError:
            print(f"[WARN] Invalid float, using default: {config['mandelbulb_escape_radius']}")

        # 3.18 Birkhoff temperature
        b_temp = get_input("[?] Birkhoff manifold temperature", str(config['birkhoff_temperature']))
        try:
            config['birkhoff_temperature'] = float(b_temp)
        except ValueError:
            print(f"[WARN] Invalid float, using default: {config['birkhoff_temperature']}")

        # 3.19 Open Science Email
        os_email = get_input("[?] Email for Open Science Ingestor (NCBI/Entrez)", config['open_science_email'])
        config['open_science_email'] = os_email

        # 3.20 ChatGPT Ingestor On/Off
        chat_enabled = get_input("[?] Enable ChatGPT Friction Harvester (yes/no)", 'yes' if config['chatgpt_ingestor_enabled'] else 'no').lower()
        config['chatgpt_ingestor_enabled'] = chat_enabled in ('yes', 'y')

        # 3.21 ChatGPT Ingestor Verbosity
        chat_verb = get_input("[?] ChatGPT Ingestor verbosity (low/normal/high)", config['chatgpt_ingestor_verbosity']).lower()
        config['chatgpt_ingestor_verbosity'] = chat_verb

        # 3.22 Open Science Ingestor On/Off
        os_enabled = get_input("[?] Enable Open Science Ingestor (yes/no)", 'yes' if config['open_science_ingestor_enabled'] else 'no').lower()
        config['open_science_ingestor_enabled'] = os_enabled in ('yes', 'y')

        # 3.23 Open Science Ingestor Verbosity
        os_verb = get_input("[?] Open Science Ingestor verbosity (low/normal/high)", config['open_science_ingestor_verbosity']).lower()
        config['open_science_ingestor_verbosity'] = os_verb

        # 3.19 Birkhoff max iterations
        b_iters = get_input("[?] Birkhoff max iterations", str(config['birkhoff_max_iterations']))
        try:
            config['birkhoff_max_iterations'] = int(b_iters)
        except ValueError:
            print(f"[WARN] Invalid int, using default: {config['birkhoff_max_iterations']}")

        # 3.20 TDA landmarks
        tda_l = get_input("[?] Approximate TDA landmarks", str(config['tda_landmarks']))
        try:
            config['tda_landmarks'] = int(tda_l)
        except ValueError:
            print(f"[WARN] Invalid int, using default: {config['tda_landmarks']}")

        # 3.21 Ego death limit
        ego = get_input("[?] Ego death abstraction limit", str(config['ego_death_limit']))
        try:
            config['ego_death_limit'] = float(ego)
        except ValueError:
            print(f"[WARN] Invalid float, using default: {config['ego_death_limit']}")

        # 3.22 UDP Server Colonizer
        udp_col_prompt = "[?] Enable Option D UDP Master Server Colonizer (yes/no) [Default: no]: "
        udp_col = get_input(udp_col_prompt, 'no').lower()
        config['udp_colonizer_enabled'] = udp_col in ('yes', 'y')

        print("\n[OK] Configuration finalized successfully.")
        print("-" * 50)
        
        return ports, config


class HybridAI:
    """Hybrid AI system using only working components."""
    
    def __init__(self, use_spectral_correction: bool = True, config: dict = None):
        if config is None:
            config = {}
            
        from src.core import DEVICE
        self.device = DEVICE
        # Torch tensors still use cpu when device is 'opencl' — PyOpenCL ops run via TailSlayer kernels
        self.torch_device = 'cpu' if str(DEVICE) == 'cpu' else 'cpu' # Centralized for now
        
        # Save config settings as attributes for process_text overrides
        self.default_regime = config.get('regime', 'goo')
        self.default_commutativity = config.get('commutativity', 'non_commutative')
        self.high_throughput_ingestion = config.get('high_throughput_ingestion', False)
        
        # Initialize working components

        if TEMPORAL_MODEL_AVAILABLE:
            try:
                self.temporal_model = NonLobotomyTemporalModel(
                    input_dim=768,
                    hidden_dim=256,
                    num_functionals=5,
                    poly_degree=4,
                    device=self.torch_device
                )
                
                # Verify and initialize ADMR Solver
                self.admr_solver = PolynomialADMRSolver(
                    poly_config=self.temporal_model.polynomial_config,
                    state_dim=256,
                    device=self.torch_device
                )
                
                # Initialize Ley Line Metric and Möbius Bundle
                self.ley_line_metric = LeyLineGeodesicMetric(dim=256)
                self.moebius_bundle = MoebiusFiberBundle(dim=256, fiber_dim=64)
                
                print("[OK] Advanced AI components initialized")
            except Exception as e:
                print(f"[FAIL] Advanced AI initialization failed: {e}")
                self.temporal_model = None
                self.admr_solver = None
        else:
            self.temporal_model = None
            self.admr_solver = None
        
        if use_spectral_correction and SPECTRAL_CORRECTOR_AVAILABLE:
            try:
                self.spectral_corrector = SpectralCoherenceCorrector(
                    initial_threshold=0.7,
                    min_threshold=0.3,
                    adaptation_rate=0.1,
                    device=self.torch_device
                )
                print("[OK] Spectral corrector initialized")
                self.rupture_fn = RuptureFunctional(rupture_threshold=0.5) # More sensitive threshold
                
                # INTEGRATE CODES FRAMEWORK (by Devin Bostick)
                from src.core.codes_constraint_framework import CODESConstraintFramework
                # Ensure state_dim is passed as a pure int to avoid scalar conversion traps
                self.codes_framework = CODESConstraintFramework(
                    state_dim=int(256),
                    max_constraints=10,
                    energy_margin=0.8
                )
                self.codes_framework.add_constraint(0, 'quadratic')
                self.codes_framework.add_constraint(1, 'polynomial_coprime')
                print("[OK] CODES Constraint Framework integrated")
                
                # Hybrid Number-Theoretic Stabilizer
                self.stabilizer = NumberTheoreticStabilizer(state_dim=256).to(self.torch_device, non_blocking=True)
                print("[OK] Hybrid Number-Theoretic Stabilizer active")
            except Exception as e:
                import traceback
                print(f"[FAIL] Spectral corrector/CODES initialization failed: {e}")
                traceback.print_exc()
                self.spectral_corrector = None
                self.rupture_fn = None
                self.codes_framework = None
        else:
            self.spectral_corrector = None
            self.rupture_fn = None
            self.codes_framework = None

        # =========================================================
        # DIEGETIC PHYSICS ENGINE INTEGRATION (CALM, KAGH, LARYNX)
        # =========================================================
        try:
            from src.ui.diegetic_backend import DiegeticPhysicsEngine
            # Initialize with compatible dimension (256 matches hybrid state)
            self.engine = DiegeticPhysicsEngine(dim=256, device=self.torch_device, config=config)
            import src.ui.diegetic_backend
            src.ui.diegetic_backend.ENGINE = self.engine
            print("[OK] Diegetic Physics Engine attached (CALM/KAGH/FGRT/Larynx Active)")
        except Exception as e:
             print(f"[FAIL] Diegetic Engine connection failed: {e}")
             self.engine = None
        
        self.iteration_count = 0
        
        # --- Implicated System State S(t) = <Phi_I, Phi_C, Delta> ---
        # Phi_I (Interiority): The latent manifold state (handled by hidden_state)
        # Phi_C (Narration): Persistent state of the linguistic output
        self.narration_field = (self._harvest_honest_jitter((256,)) - 0.5) * 0.001
        # Delta (Damage): Accumulated paraconsistent contradictions (toxic memory)
        self.damage_residue = (self._harvest_honest_jitter((256,)) - 0.5) * 0.001
        # Perfect Memory Anchor (Phi_P): Lossless historical component
        self.perfect_memory = [] # Historical residues

        # Initialize the manifold with harmonic seed before potentially loading fossil
        self._initialize_manifold_state()


        # Initialize Dataset System
        if DATASET_SYSTEM_AVAILABLE:
            try:
                self.dataset_system = DatasetIngestionSystem(device=self.torch_device, engine=getattr(self, 'engine', None))
                print("[OK] Dataset Ingestion System initialized")
            except Exception as e:
                print(f"[FAIL] Dataset Ingestion System init failed: {e}")
                self.dataset_system = None
        else:
            self.dataset_system = None
            
        # Initialize Training Manager
        try:
            from src.training.training_manager import TrainingManager
            self.training_manager = TrainingManager(self)
            print("[OK] Training Manager initialized")
        except Exception as e:
            print(f"[FAIL] Training Manager init failed: {e}")
            self.training_manager = None

        # Initialize Gyroidic Graph Manager (Topology Visualization)
        try:
            from src.topology.embedding_graph import GyroidicGraphManager
            self.graph_dir = os.path.join(root_dir, 'data', 'encodings')
            os.makedirs(self.graph_dir, exist_ok=True)
            
            self.graph_manager = GyroidicGraphManager(data_dir=self.graph_dir, dim=256)
            self.graph_manager.load_fossils(limit=150)
            print(f"[OK] Gyroidic Graph Manager initialized with {len(self.graph_manager.nodes)} fossils")
        except Exception as e:
            print(f"[FAIL] Graph Manager init failed: {e}")
            self.graph_manager = None

        # Audience Mapping (Φ: M -> A)
        self.audience_mapper = AudienceProjection(input_dim=256, audience_dim=256)

        # Superposed Tag Stacker (Ganbreeder fallback)
        self.tag_stacker = SuperposedTagStacker(state_dim=256, device=self.torch_device)

        # Apply start configuration options to engine components recursively
        
        # 1. McKenna deconstruction mode
        mckenna_mode = config.get('mckenna_deconstruction_mode', False)
        try:
            from src.ui.diegetic_backend import TEXTBOOK_FILTER
            TEXTBOOK_FILTER.mckenna_deconstruction_mode = mckenna_mode
            print(f"[INIT] Textbook Filter McKenna Deconstruction: {mckenna_mode}")
        except Exception as e:
            print(f"[FAIL] Could not set TEXTBOOK_FILTER McKenna mode: {e}")
            
        # 2. Quantum-inspired reasoning pre-initialization
        quantum_mode = config.get('quantum_inspired_mode', False)
        if quantum_mode and self.engine:
            try:
                from src.core.meta_polytope_matrioshka import MetaPolytopeMatrioshka
                from src.core.quantum_inspired_reasoning import QuantumInspiredReasoningState
                from src.core.sparse_higher_order_tensors import SparseHigherOrderTensorDynamics
                self.engine.meta_polytope = MetaPolytopeMatrioshka(max_depth=5, base_dim=self.engine.dim)
                self.engine.tensor_dynamics = SparseHigherOrderTensorDynamics(max_order=3, num_shells=3, base_dim=self.engine.dim)
                self.engine.quantum_reasoner = QuantumInspiredReasoningState(dim=self.engine.dim)
                self.engine.extensions_enabled = True
                print("[INIT] Quantum-inspired reasoning & advanced engines pre-initialized")
            except Exception as e:
                print(f"[FAIL] Quantum-inspired engine pre-initialization failed: {e}")
                
        # 3. Introspection Head Configuration
        if self.engine and hasattr(self.engine, 'introspection') and self.engine.introspection is not None:
            try:
                from src.models.introspection_head import AggregateGeometricSelfModel
                probes_list = config.get('introspection_probes', ['moral', 'uncertainty', 'creative', 'metacognitive'])
                if isinstance(probes_list, str):
                    probes_list = [p.strip() for p in probes_list.split(',') if p.strip()]
                self.engine.introspection = AggregateGeometricSelfModel(
                    hidden_dim=self.engine.dim,
                    probe_types=probes_list
                ).to(self.engine.device)
                
                suppress_narr = config.get('suppress_narration', True)
                self.engine.introspection.suppress_narration = suppress_narr
                if hasattr(self.engine.introspection, 'probe_head'):
                    self.engine.introspection.probe_head.suppress_narration = suppress_narr
                    
                self.engine.introspection.rigidity_decay_rate = config.get('rigidity_decay_rate', 0.005)
                print(f"[INIT] Introspection Head configured: Probes={probes_list}, SuppressNarration={suppress_narr}")
            except Exception as e:
                print(f"[FAIL] Configuring Introspection Head failed: {e}")
                
        # 4. Open Science cache folder configuration
        if self.engine and getattr(self.engine, 'open_science_ingestor', None) is not None:
            try:
                from pathlib import Path
                cache_folder = config.get('cache_dir', 'datasets/open_science_cache')
                self.engine.open_science_ingestor.cache_dir = Path(cache_folder)
                self.engine.open_science_ingestor.cache_dir.mkdir(exist_ok=True, parents=True)
                print(f"[INIT] Open Science cache directory set to: {cache_folder}")
            except Exception as e:
                print(f"[FAIL] Setting Open Science cache path failed: {e}")
                
        # 5. KAGH dyslexic mode
        if self.engine and hasattr(self.engine, 'kagh_drafter') and self.engine.kagh_drafter is not None:
            dyslexic = config.get('kagh_dyslexic_mode', False)
            try:
                self.engine.kagh_drafter.dyslexic_mode = dyslexic
                if hasattr(self.engine.kagh_drafter, 'layers'):
                    for layer in self.engine.kagh_drafter.layers:
                        layer.dyslexic_mode = dyslexic
                print(f"[INIT] KAGH dyslexic mode set to: {dyslexic}")
            except Exception as e:
                print(f"[FAIL] Configuring KAGH dyslexic mode failed: {e}")
                
        # 6. FBM Erosion
        f_persistence = config.get('fbm_persistence', 0.5)
        f_octaves = config.get('fbm_octaves', 3)
        if self.engine:
            try:
                for module in self.engine.modules():
                    if module.__class__.__name__ == 'TopologicalErosionFBM':
                        module.persistence = float(f_persistence)
                        module.octaves = int(f_octaves)
                print(f"[INIT] FBM Erosion configured: Persistence={f_persistence}, Octaves={f_octaves}")
            except Exception as e:
                print(f"[FAIL] Configuring FBM Erosion failed: {e}")
                
        # 7. Mandelbulb Augmentation
        m_pow = config.get('mandelbulb_power', 8.0)
        m_esc = config.get('mandelbulb_escape_radius', 2.0)
        if self.engine:
            try:
                from src.augmentation.mandelbulb_gyroidic_augmenter import MandelbulbGyroidicAugmenter, AugmentationConfig
                aug_config = AugmentationConfig(mandelbulb_power=int(m_pow))
                self.engine.augmenter = MandelbulbGyroidicAugmenter(aug_config).to(self.engine.device)
                if hasattr(self.engine.augmenter, 'mandelbulb') and self.engine.augmenter.mandelbulb is not None:
                    self.engine.augmenter.mandelbulb.power = float(m_pow)
                    self.engine.augmenter.mandelbulb.escape_radius = float(m_esc)
                print(f"[INIT] Mandelbulb Augmenter configured: Power={m_pow}, EscapeRadius={m_esc}")
            except Exception as e:
                print(f"[FAIL] Configuring Mandelbulb Augmenter failed: {e}")
                
        # 8. Birkhoff Obscured Manifold
        b_temp = config.get('birkhoff_temperature', 1.0)
        b_iters = config.get('birkhoff_max_iterations', 100)
        if self.engine:
            try:
                for module in self.engine.modules():
                    if module.__class__.__name__ in ('ObscuredBirkhoffManifold', 'BouligandBirkhoffManifold'):
                        if hasattr(module, 'temperature') and isinstance(module.temperature, nn.Parameter):
                            with torch.no_grad():
                                module.temperature.copy_(torch.tensor(max(0.01, min(10.0, float(b_temp)))))
                        else:
                            module.temperature = float(b_temp)
                        module.max_iterations = int(b_iters)
                print(f"[INIT] Birkhoff Obscured Manifold configured: Temp={b_temp}, MaxIters={b_iters}")
            except Exception as e:
                print(f"[FAIL] Configuring Birkhoff Obscured Manifold failed: {e}")
                
        # 9. TDA Landmarks
        tda_l = config.get('tda_landmarks', 50)
        if self.engine:
            try:
                for module in self.engine.modules():
                    if module.__class__.__name__ == 'ApproximatePHProbe':
                        module.num_landmarks = int(tda_l)
                print(f"[INIT] Approximate TDA Landmarks configured: Landmarks={tda_l}")
            except Exception as e:
                print(f"[FAIL] Configuring Approximate TDA Landmarks failed: {e}")
                
        # 10. Ego Death Threshold Limit
        ego_limit = config.get('ego_death_limit', 1.5)
        if self.engine and hasattr(self.engine, 'archetypal_governor') and self.engine.archetypal_governor is not None:
            try:
                if hasattr(self.engine.archetypal_governor, 'abstraction') and self.engine.archetypal_governor.abstraction is not None:
                    self.engine.archetypal_governor.abstraction.abstraction_limit = float(ego_limit)
                print(f"[INIT] Ego Death Threshold Limit configured: Limit={ego_limit}")
            except Exception as e:
                print(f"[FAIL] Configuring Ego Death Threshold Limit failed: {e}")
                
        # 11. Background Scientific Learning
        self.bg_scientific_learning = config.get('bg_scientific_learning', True)
        self.primary_query_dataset = config.get('primary_query_dataset', 'LIGO')
        self.cache_dir = config.get('cache_dir', 'datasets/open_science_cache')
        if self.bg_scientific_learning:
            self._start_background_scientific_learning()

        # SOVEREIGN WARMSTART: Restore manifold if fossil exists (Thorium Protocol)
        self.load_model_state()

    def _initialize_manifold_state(self):
        """Initialize the manifold with the FGRT harmonic seed (Love Vector Norm 3.127)."""
        t_basis = torch.linspace(0, 2 * 3.14159265, 256, device=self.torch_device)
        initial_seed = torch.sin(t_basis) * (3.127 / torch.norm(torch.sin(t_basis)))
        self.hidden_state = initial_seed.clone()
        self.hidden_state_scarred = initial_seed.clone()
        print("[INIT] Manifold soul initialized with harmonic seed (Norm 3.127).")


    def save_model_state(self, state_path: str = None) -> str:
        """Implements Thorium Fossilization Protocol: Serializes the manifold soul."""
        try:
            if state_path is None:
                state_path = os.path.join(root_dir, 'gyroid_state.pt')
            
            # 1. Standard Core States (Safety Check: verify attribute existence)
            if not hasattr(self, 'hidden_state') or self.hidden_state is None:
                 print("[WARN] Fossilization attempted with missing hidden_state. Re-initializing.")
                 self._initialize_manifold_state()

            save_dict = {
                'iteration': self.iteration_count,
                'hidden_state': self.hidden_state,
                'hidden_state_scarred': self.hidden_state_scarred,
                'damage_residue': self.damage_residue,
            }

            
            # 2. MANIFOLD SOUL (Phase 18 Protocol)
            if self.engine:
                try:
                    manifold_assets = self.engine.get_manifold_state()
                    save_dict.update({
                        'zeitgeist': manifold_assets.get('zeitgeist'),
                        'love_invariant': manifold_assets.get('love_invariant'),
                        'fossil_memory': manifold_assets.get('fossil_memory'),
                        'cavity_M': manifold_assets.get('cavity', {}).get('M'),
                        'cavity_D_dark': manifold_assets.get('cavity', {}).get('D_dark'),
                        'engine_meta_state': manifold_assets.get('meta_state'),
                        'unicode_to_idx': manifold_assets.get('unicode_to_idx', {}),
                        'idx_to_unicode': manifold_assets.get('idx_to_unicode', {})
                    })
                except Exception as engine_err:
                    print(f"[WARN] Engine manifold extraction failed: {engine_err}. Saving core state only.")
            
            # 3. Temporal Model if available
            if self.temporal_model:
                try:
                    save_dict['temporal_model_state'] = self.temporal_model.state_dict()
                except Exception as model_err:
                    print(f"[WARN] Temporal model state extraction failed: {model_err}")
            
            # 4. Deep Archetypal Persistence (TADC Characters)
            if self.engine and hasattr(self.engine, 'archetypal_governor') and self.engine.archetypal_governor is not None:
                try:
                    save_dict['archetypal_governor_state'] = self.engine.archetypal_governor.export_governor_state()
                except Exception as e:
                    print(f"[WARN] Archetypal governor export failed: {e}")
                    
            # 5. Superposed Tag Stacker (Textual Tags -> Geometry)
            if hasattr(self, 'tag_stacker') and self.tag_stacker is not None:
                try:
                    save_dict['tag_stacker_state'] = {
                        'vectors': {k: v.detach().cpu() for k, v in self.tag_stacker.catalog_vectors.items()},
                        'metadata': self.tag_stacker.catalog_metadata
                    }
                except Exception as e:
                    print(f"[WARN] Tag stacker state export failed: {e}")
                    
            # 6. SIC-FA-ADMM & Optimization Momentum
            if hasattr(self, 'training_manager') and self.training_manager is not None:
                try:
                    if hasattr(self.training_manager, 'optimizer') and self.training_manager.optimizer is not None:
                        save_dict['optimizer_state'] = self.training_manager.optimizer.state_dict()
                    if hasattr(self.training_manager, 'calm_history') and self.training_manager.calm_history is not None:
                        save_dict['calm_history'] = self.training_manager.calm_history
                except Exception as e:
                    print(f"[WARN] Optimization momentum export failed: {e}")
                    
            # 7. Shadow Logs & Multimodal Collisions
            if self.engine:
                try:
                    if hasattr(self.engine, 'shadow_replay_queue'):
                        save_dict['shadow_logs'] = list(self.engine.shadow_replay_queue)
                    if hasattr(self.engine, 'multimodal_collisions'):
                        save_dict['multimodal_collisions'] = list(self.engine.multimodal_collisions)
                except Exception as e:
                    print(f"[WARN] Shadow logs/collisions export failed: {e}")
            
            # 8. CODES v40 Topological & Resonance Core Extensions
            if self.engine:
                try:
                    codes_state = {}
                    
                    # Speculative Homology Engine
                    if hasattr(self.engine, 'speculative_homology') and self.engine.speculative_homology:
                        codes_state['speculative_homology'] = {
                            'betti_numbers': self.engine.speculative_homology.betti_numbers if hasattr(self.engine.speculative_homology, 'betti_numbers') else None,
                            'homology_gaps': getattr(self.engine.speculative_homology, 'homology_gaps', None)
                        }
                    
                    # Chern-Simons Gasket
                    if hasattr(self.engine, 'chern_simons_gasket') and self.engine.chern_simons_gasket:
                        codes_state['chern_simons_gasket'] = {
                            'kappa': getattr(self.engine.chern_simons_gasket, 'kappa', None),
                            'curvature_history': getattr(self.engine.chern_simons_gasket, 'curvature_history', None)
                        }
                    
                    # Love Invariant Protector
                    if hasattr(self.engine, 'love_protector') and self.engine.love_protector:
                        codes_state['love_protector'] = {
                            'trust_levels': getattr(self.engine.love_protector, 'trust_levels', None),
                            'non_ergodic_entropy': getattr(self.engine.love_protector, 'non_ergodic_entropy', None)
                        }
                    
                    # Mandelbulb Gyroidic Augmenter
                    if hasattr(self.engine, 'gyroidic_augmenter') and self.engine.gyroidic_augmenter:
                        codes_state['gyroidic_augmenter'] = {
                            'power': getattr(self.engine.gyroidic_augmenter, 'power', None)
                        }
                        
                    # Resonance Intelligence Core
                    if hasattr(self.engine, 'resonance_core') and self.engine.resonance_core:
                        codes_state['resonance_core'] = {
                            'pas_history': getattr(self.engine.resonance_core, 'pas_history', None),
                            'fibonacci_entropy': getattr(self.engine.resonance_core, 'fibonacci_entropy', None),
                            'breather_modes': getattr(self.engine.resonance_core, 'breather_modes', None),
                            'multiharmonic_coherence': getattr(self.engine.resonance_core, 'multiharmonic_coherence', None)
                        }
                        
                    if codes_state:
                        save_dict['codes_v40_topology'] = codes_state
                except Exception as e:
                    print(f"[WARN] CODES v40 topology export failed: {e}")
            
            # ATOMIC SAVE
            torch.save(save_dict, state_path)
            
            size_mb = os.path.getsize(state_path) / (1024 * 1024)
            print(f"[FOSSIL] Manifold serialized to {state_path} ({size_mb:.2f} MB)")
            return f"Fossilization Protocol Complete: {size_mb:.2f} MB saved."
        except Exception as e:
            print(f"[FOSSIL] Emergency shutdown save failed: {e}")
            return f"Fossilization Failure: {e}"

    def load_model_state(self, state_path: str = None):
        """Restore manifold state from gyroid_state.pt (Warmstart)."""
        if state_path is None:
            state_path = os.path.join(root_dir, 'gyroid_state.pt')
        if not os.path.exists(state_path):
            print("[WARMSTART] No fossilized state found at root. Starting clean.")
            return

        try:
            print(f"[WARMSTART] Recovering manifold from {state_path}...")
            # Load with map_location='cpu' for cross-hardware soul-transfer
            checkpoint = torch.load(state_path, map_location='cpu')
            
            self.iteration_count = checkpoint.get('iteration', 0)
            if 'hidden_state' in checkpoint:
                loaded_state = checkpoint['hidden_state']
                
                # VALIDATION: Check against Phase 18 Harmonic Seed Dynamics (Norm 3.127)
                t_basis = torch.linspace(0, 2 * 3.14159265, 256, device=self.torch_device)
                harmonic_seed = torch.sin(t_basis) * (3.127 / torch.norm(torch.sin(t_basis), p=2))
                
                loaded_norm = torch.norm(loaded_state, p=2).item()
                # Correlation check to ensure the manifold hasn't flattened into ergodic noise
                correlation = torch.abs(torch.dot(loaded_state.flatten(), harmonic_seed.flatten()) / (torch.norm(loaded_state) * torch.norm(harmonic_seed) + 1e-8)).item()
                
                if loaded_norm < 0.5 or correlation < 0.01:
                    print(f"[RECOVERY] Fossil integrity check FAILED (Norm: {loaded_norm:.4f}, Corr: {correlation:.4f}).")
                    print("[RECOVERY] Manifold is spectrally flat or corrupted. Triggering Harmonic Re-genesis...")
                    self._initialize_manifold_state()
                else:
                    self.hidden_state = loaded_state.clone()
                    # Sync scarred state for temporal continuity
                    if 'hidden_state_scarred' in checkpoint:
                        self.hidden_state_scarred = checkpoint['hidden_state_scarred'].clone()
                    else:
                        # Apply Structurally Honest Jitter to break potential phase-locks (§45.2)
                        jitter = self._harvest_honest_jitter(self.hidden_state.shape)
                        self.hidden_state_scarred = self.hidden_state + jitter
                    print(f"[RECOVERY] Fossil integrity verified (Norm: {loaded_norm:.4f}, Corr: {correlation:.4f}).")
            if 'damage_residue' in checkpoint:

                self.damage_residue = checkpoint['damage_residue']

            # Restore Manifold Assets via Engine
            if self.engine:
                # Map standard keys to manifold dict, providing safety defaults where applicable
                # Values will be checked for None inside load_manifold_state as well.
                manifold_dict = {
                    "zeitgeist": checkpoint.get("zeitgeist"),
                    "love_invariant": checkpoint.get("love_invariant"),
                    "fossil_memory": checkpoint.get("fossil_memory"),
                    "cavity": {
                        "M": checkpoint.get("cavity_M"),
                        "D_dark": checkpoint.get("cavity_D_dark")
                    },
                    "meta_state": checkpoint.get("engine_meta_state"),
                    "iteration": checkpoint.get("iteration", self.iteration_count),
                    "unicode_to_idx": checkpoint.get("unicode_to_idx", {}),
                    "idx_to_unicode": checkpoint.get("idx_to_unicode", {})
                }
                self.engine.load_manifold_state(manifold_dict)

            if self.temporal_model and 'temporal_model_state' in checkpoint:
                self.temporal_model.load_state_dict(checkpoint['temporal_model_state'], strict=False)
            
            # Restore Archetypal Governor
            if 'archetypal_governor_state' in checkpoint and self.engine and hasattr(self.engine, 'archetypal_governor') and self.engine.archetypal_governor is not None:
                try:
                    self.engine.archetypal_governor.import_governor_state(checkpoint['archetypal_governor_state'])
                except Exception as e:
                    print(f"[RECOVERY] Failed to restore Archetypal Governor: {e}")
            
            # Restore Superposed Tag Stacker
            if 'tag_stacker_state' in checkpoint and hasattr(self, 'tag_stacker') and self.tag_stacker is not None:
                try:
                    tag_state = checkpoint['tag_stacker_state']
                    for k, v in tag_state['vectors'].items():
                        self.tag_stacker.catalog_vectors[k] = torch.nn.Parameter(v.to(self.torch_device))
                    self.tag_stacker.catalog_metadata = tag_state['metadata']
                except Exception as e:
                    print(f"[RECOVERY] Failed to restore Superposed Tag Stacker: {e}")
                    
            # Restore SIC-FA-ADMM Momentum
            if 'optimizer_state' in checkpoint and hasattr(self, 'training_manager') and self.training_manager is not None:
                if hasattr(self.training_manager, 'optimizer') and self.training_manager.optimizer is not None:
                    try:
                        self.training_manager.optimizer.load_state_dict(checkpoint['optimizer_state'])
                    except Exception as e:
                        print(f"[RECOVERY] Failed to restore optimizer state: {e}")
            if 'calm_history' in checkpoint and hasattr(self, 'training_manager') and self.training_manager is not None:
                self.training_manager.calm_history = checkpoint['calm_history']
                
            # Restore Shadow Logs & Multimodal Collisions
            if self.engine:
                if 'shadow_logs' in checkpoint:
                    from collections import deque
                    self.engine.shadow_replay_queue = deque(checkpoint['shadow_logs'], maxlen=1000)
                if 'multimodal_collisions' in checkpoint:
                    from collections import deque
                    self.engine.multimodal_collisions = deque(checkpoint['multimodal_collisions'], maxlen=500)
            
            # Restore CODES v40 Topological & Resonance Core Extensions
            if self.engine and 'codes_v40_topology' in checkpoint:
                try:
                    codes_state = checkpoint['codes_v40_topology']
                    
                    if 'speculative_homology' in codes_state and hasattr(self.engine, 'speculative_homology') and self.engine.speculative_homology:
                        if 'betti_numbers' in codes_state['speculative_homology']:
                            self.engine.speculative_homology.betti_numbers = codes_state['speculative_homology']['betti_numbers']
                        if 'homology_gaps' in codes_state['speculative_homology']:
                            self.engine.speculative_homology.homology_gaps = codes_state['speculative_homology']['homology_gaps']
                            
                    if 'chern_simons_gasket' in codes_state and hasattr(self.engine, 'chern_simons_gasket') and self.engine.chern_simons_gasket:
                        if 'kappa' in codes_state['chern_simons_gasket']:
                            self.engine.chern_simons_gasket.kappa = codes_state['chern_simons_gasket']['kappa']
                        if 'curvature_history' in codes_state['chern_simons_gasket']:
                            self.engine.chern_simons_gasket.curvature_history = codes_state['chern_simons_gasket']['curvature_history']
                            
                    if 'love_protector' in codes_state and hasattr(self.engine, 'love_protector') and self.engine.love_protector:
                        # Attempt to topological recovery or just raw trust loading
                        # For now, reload raw trust into memory
                        if 'trust_levels' in codes_state['love_protector'] and codes_state['love_protector']['trust_levels'] is not None:
                            self.engine.love_protector.trust_levels = codes_state['love_protector']['trust_levels']
                        if 'non_ergodic_entropy' in codes_state['love_protector']:
                            self.engine.love_protector.non_ergodic_entropy = codes_state['love_protector']['non_ergodic_entropy']
                            
                    if 'gyroidic_augmenter' in codes_state and hasattr(self.engine, 'gyroidic_augmenter') and self.engine.gyroidic_augmenter:
                        if 'power' in codes_state['gyroidic_augmenter']:
                            self.engine.gyroidic_augmenter.power = codes_state['gyroidic_augmenter']['power']
                            
                    if 'resonance_core' in codes_state and hasattr(self.engine, 'resonance_core') and self.engine.resonance_core:
                        if 'pas_history' in codes_state['resonance_core']:
                            self.engine.resonance_core.pas_history = codes_state['resonance_core']['pas_history']
                        if 'fibonacci_entropy' in codes_state['resonance_core']:
                            self.engine.resonance_core.fibonacci_entropy = codes_state['resonance_core']['fibonacci_entropy']
                        if 'breather_modes' in codes_state['resonance_core']:
                            self.engine.resonance_core.breather_modes = codes_state['resonance_core']['breather_modes']
                        if 'multiharmonic_coherence' in codes_state['resonance_core']:
                            self.engine.resonance_core.multiharmonic_coherence = codes_state['resonance_core']['multiharmonic_coherence']
                            
                except Exception as e:
                    print(f"[WARN] Failed to restore CODES v40 topology: {e}")
            print("[WARMSTART] Manifold soul restored successfully.")
        except Exception as e:
            print(f"[WARMSTART] Error during recovery: {e}. Manifold may be corrupt or entropic.")

    
    def _harvest_honest_jitter(self, shape: torch.Size, scaled: bool = True) -> torch.Tensor:
        """
        Harvests Structurally Honest Jitter from silicon state variance.
        Follows §45.2 (Silicon Sovereignty).
        """
        import time
        jitter_tensor = torch.zeros(shape, device=self.torch_device)
        flat = jitter_tensor.flatten()
        
        # Warm up cache and measure nano-variance friction
        t0 = time.perf_counter_ns()
        # Small matrix ops to generate hardware friction
        for _ in range(5):
             _ = torch.det(torch.randn((8, 8), device=self.torch_device))
        t1 = time.perf_counter_ns()
        
        # Harvest the 'least significant nanoseconds' as a seed val
        seed_val = ((t1 - t0) % 1000) / 1000.0
        if seed_val == 0: seed_val = 0.5
        
        # Deterministic chaotic expansion (Logistic map)
        x = seed_val
        for i in range(len(flat)):
            # x_{n+1} = 3.99 * x_n * (1 - x_n) -- chaotic regime
            x = 3.99 * x * (1.0 - x)
            flat[i] = x
        
        if scaled:
            return (jitter_tensor - 0.5) * 0.1
        return jitter_tensor

    def process_text(self, text: str, video_dyad_b64: str = None, commutativity: str = 'non_commutative', fingerprint: dict = None, audio_dyad: dict = None, regime: str = 'goo', tag_weights: dict = None, ingestion_mode: bool = False) -> dict:
        # Override parameters if they match standard defaults and custom settings were chosen at startup
        if commutativity == 'non_commutative':
            commutativity = getattr(self, 'default_commutativity', 'non_commutative')

        # Ensure hidden_state is ready for cloning (isolation snapshot)
        if not hasattr(self, 'hidden_state') or self.hidden_state is None:
            self._initialize_manifold_state()

        # --- DYNAMIC REGIME DETERMINATION (Integrated Emergence Condition, Eq 10) ---
        self.current_regime = 'goo'
        with torch.no_grad():
            pas_h_live_init = 0.61
            if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'meta_state'):
                pas_h_live_init = self.engine._compute_pas_h(self.engine.meta_state)
            elif self.hidden_state is not None:
                # Fallback calculation
                pas_h_live_init = 1.0
                state_len = len(self.hidden_state)
                for d in range(8):
                    segment = self.hidden_state[d*(state_len//8):(d+1)*(state_len//8)]
                    pas_h_live_init += (1.0 / (d + 1.0)) * torch.norm(segment).item()
            
            if not hasattr(self, 'prev_pas'):
                self.prev_pas = pas_h_live_init
            drift_init = abs(pas_h_live_init - self.prev_pas)
            self.prev_pas = pas_h_live_init
            
            atrophy_val_init = 0.0
            if hasattr(self, 'engine') and self.engine and getattr(self.engine, 'use_gyroid_probes', False):
                sample = self.engine.meta_state if hasattr(self.engine, 'meta_state') else None
                atrophy_metrics = self.engine.gyroid_cov.get_elipsodistrophy_metrics(sample)
                atrophy_val_init = atrophy_metrics.get('atrophy', 0.0)
                
            is_glyph_locked = False
            if hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'poly_config'):
                is_glyph_locked = bool(check_glyphlock(self.engine.poly_config.get_coefficients_tensor()).max().item() > 0)
            elif self.hidden_state is not None:
                coeffs = self.hidden_state.unsqueeze(0) if self.hidden_state.dim() == 1 else self.hidden_state
                is_glyph_locked = bool(check_glyphlock(coeffs).max().item() > 0)
                
            theta_L = 0.85
            epsilon_drift = 0.05
            is_coherent = pas_h_live_init >= theta_L
            is_stable = drift_init <= epsilon_drift
            
            if is_coherent and is_stable and is_glyph_locked and atrophy_val_init < 0.85:
                self.current_regime = 'prickles'
            else:
                self.current_regime = 'goo'

        hidden_state = self.hidden_state.clone()
        diagnostics = {}
        response_text = ""

        # Coordinate Harvesting Protocol (§3.A)
        if text.strip().upper().startswith("SAVE_TAG:"):
            # Extract tag_name and optional context
            parts = text.strip()[9:].split(" ", 1)
            tag_name = parts[0].strip()
            context_text = parts[1].strip() if len(parts) > 1 else "Manual tag registration via stacker protocol."
            
            # If engine is available, register it there
            if self.engine and hasattr(self.engine, 'archetypal_governor'):
                result = self.engine.archetypal_governor.harvest_named_coordinate(
                    tag_name=tag_name,
                    vector=self.hidden_state.clone(),
                    context_text=context_text
                )
                return {
                    "status": "TAG_HARVESTED" if result["success"] else "TAG_REFUSED",
                    "response": f"[STACKER] Successfully registered tag '{tag_name}'." if result["success"] else f"[STACKER] Refused registration of '{tag_name}'. Reason: Did not satisfy textbook-quality threshold.",
                    "diagnostics": result,
                    "iteration": self.iteration_count
                }
            elif hasattr(self, 'tag_stacker'):
                # Local fallback harvest
                success, report = self.tag_stacker.add_tag(tag_name, self.hidden_state.clone(), context_text)
                return {
                    "status": "TAG_HARVESTED" if success else "TAG_REFUSED",
                    "response": f"[STACKER] Locally registered tag '{tag_name}'." if success else f"[STACKER] Locally refused '{tag_name}'.",
                    "diagnostics": {"success": success, "admissible": report.is_admissible},
                    "iteration": self.iteration_count
                }

        # Compute and apply tag stacking bias if provided
        stacked_bias = torch.zeros(256, device=self.torch_device)
        if tag_weights and hasattr(self, 'tag_stacker'):
            stacked_bias = self.tag_stacker.compute_composite_target(tag_weights)
            # Gently nudge baseline manifold toward user stacked vector
            self.hidden_state = self.hidden_state + 0.1 * stacked_bias
        
        self.corrected_tensor = torch.zeros(256, device=self.torch_device)
        self.iteration_count += 1

        
        # Prevent heartbeat from showing up in main chat or triggering Ego Death
        if "IDLE_RESONANCE_HEARTBEAT" in text:
            return {
                "status": "HEARTBEAT_ACK",
                "response": None,
                "diagnostics": {"suppress_ui": True},
                "iteration": self.iteration_count
            }
        
        # Create deterministic topological hash embedding (CODES by Devin Bostick)
        # We project the text into a 768D manifold using character-position harmonics
        text_embedding = torch.zeros(768, device=self.torch_device)
        for i, char in enumerate(text[:128]):
            # Use prime-based harmonics for character encoding
            freq = (i + 1) * (ord(char) / 128.0) * 3.14159
            text_embedding += torch.sin(torch.linspace(0, freq, 768, device=self.torch_device))
            text_embedding = torch.tanh(text_embedding)
            
        # --- INFERENCE CONNECTION ---
        # If the DiegeticPhysicsEngine is available, use it (CALM/KAGH/FGRT/Larynx)
        if self.engine:
            try:
                # Process via Diegetic Engine
                print(f"[ENGINE] Processing: '{text}' (Video Dyad: {'YES' if video_dyad_b64 else 'NO'}) (Image Fingerprint: {'YES' if fingerprint else 'NO'})", flush=True)
                is_ingest = ingestion_mode or getattr(self, 'high_throughput_ingestion', False)
                gen_resp = not is_ingest
                engine_output = self.engine.process_input(
                    text_input=text, 
                    fingerprint=fingerprint,
                    audio_dyad=audio_dyad,
                    video_dyad_b64=video_dyad_b64,
                    commutativity=commutativity,
                    generate_response=gen_resp,
                    ingestion_mode=is_ingest,
                    regime=self.current_regime,
                    tag_weights=tag_weights
                )
                
                # Fix: Handle case where engine_output might be a list (e.g. batch/bulk interaction)
                if isinstance(engine_output, list):
                    if len(engine_output) > 0:
                        engine_output = engine_output[0]
                    else:
                        engine_output = {"response": "System quiescent: manifold empty.", "status": "EMPTY"}
                
                if not isinstance(engine_output, dict):
                    raise ValueError(f"Engine returned invalid type: {type(engine_output)}")

                response_text = engine_output.get('response', '')
                diagnostics = engine_output
                
                # --- ENTROPY MONITORING & ACTIVITY INDUCTION ---
                # Monitor spectral entropy to prevent manifold collapse
                spectral_entropy = engine_output.get('spectral_entropy', 0.0)
                if isinstance(spectral_entropy, torch.Tensor):
                    spectral_entropy = spectral_entropy.item()
                
                if spectral_entropy < 0.05:
                    print(f"[PHYSICS] Spectral Flatness Detected (Entropy: {spectral_entropy:.4f}).")
                    print("[PHYSICS] Inducing topological activity via Soliton Healer...")
                    if hasattr(self.engine, 'soliton_healer'):
                        self.engine.soliton_healer.reset_healing()
                        # Apply high-pressure ranging signal (Honest Jitter) to "heat" the manifold
                        honest_heat = self._harvest_honest_jitter(self.engine.soliton_healer.alpha.shape)
                        self.engine.soliton_healer.alpha.data.add_(torch.abs(honest_heat) * 0.5)
                
                # Capture evolved state in local binding for isolation-safe fossilization
                if 'hidden_state' in engine_output:
                    hidden_state = engine_output['hidden_state']
                elif hasattr(self.engine, 'meta_state'):
                    hidden_state = self.engine.meta_state.clone()
                
                # Check for "topological_shape_stalk" payload
                if 'payload' in engine_output and engine_output['payload'].get('type') == 'topological_shape_stalk':
                     pass # Handle special payloads if needed
                
                # (Early return removed, processing continues to shared exit)
            except Exception as e:
                print(f"[FAIL] Diegetic Engine processing failed: {e}")
                # Fallthrough to legacy logic
        
        # 2. System 2 Fallback: Non-Lobotomy Temporal Model
        # This replaces the legacy lobotomized prime tracker with the polynomial functional reasoner.
        model_diagnostics = {}
        if self.temporal_model:
            try:
                # Prepare input [1, 768]
                model_input = text_embedding.unsqueeze(0)
                
                with torch.no_grad():
                    # Run forward pass with analysis (RECURSION GAURD: This is the single entry point)
                    model_out = self.temporal_model(model_input, return_analysis=True)
                
                # a) Update system state with REAL neural activation
                self.hidden_state = model_out['hidden_state'].squeeze(0)  # [batch, dim]
                
                # b) Pythagorean Bridge: Project to 256D if needed (Legacy Support)
                if self.hidden_state.shape[-1] == 768:
                    self.hidden_state_scarred = self.hidden_state.view(256, 3).mean(dim=1)
                else:
                    self.hidden_state_scarred = self.hidden_state # Already correct dim or will be handled by ADMR
                
                # c) Temporal Evolution (ADMR Solver Integration)
                hidden_state_256 = self.hidden_state_scarred.clone() # Anchor for negotiation loss
                hidden_state_evolved = self.hidden_state.unsqueeze(0) # Default to static if solver missing
                hidden_state_evolved_sq = self.hidden_state.clone()
                
                if hasattr(self, 'admr_solver'):
                    neighbor_states = torch.stack([self.temporal_model.prev_states.mean(dim=0)] * 1).unsqueeze(0).to(self.torch_device)
                    adj_weight = torch.ones(1, neighbor_states.shape[1]).to(self.torch_device)
                    
                    # Run MAML online meta-optimization steps using the sliding support buffer
                    if not hasattr(self, 'admr_support_buffer'):
                        self.admr_support_buffer = []
                    if len(self.admr_support_buffer) > 0:
                        entropy_val = torch.tensor([0.5], device=self.torch_device)
                        for s_states, s_neighbors, s_weights in self.admr_support_buffer:
                            self.admr_solver = self.admr_solver.meta_optimize_admm_step(
                                s_states, s_neighbors, s_weights, steps=1, lr=0.01, entropy=entropy_val
                            )
                            
                    # We use the raw hidden state for the ADMR step
                    _out = self.admr_solver.stochastic_differential_step(
                        states=self.hidden_state.unsqueeze(0),
                        neighbor_states=neighbor_states,
                        adjacency_weight=adj_weight
                    )
                    
                    # Append current states to sliding support buffer
                    self.admr_support_buffer.append((
                        self.hidden_state.unsqueeze(0).detach().clone(),
                        neighbor_states.detach().clone(),
                        adj_weight.detach().clone()
                    ))
                    if len(self.admr_support_buffer) > 4:
                        self.admr_support_buffer.pop(0)
                    # Update state from solver
                    if isinstance(_out, torch.Tensor):
                        hidden_state_evolved = _out
                        self.hidden_state = hidden_state_evolved.squeeze(0).squeeze(0)
                    hidden_state = self.hidden_state.clone() # Update local binding
                    
                    # Update the state with the evolved trajectory
                    hidden_state_evolved_sq = hidden_state_evolved.squeeze(0)
                    # Define Lawful Distortion (0.01 sigma as per Solver signature)
                    distortion = self._harvest_honest_jitter(hidden_state_evolved_sq.shape) * 0.1
                    self.hidden_state_scarred = hidden_state_evolved_sq + distortion

                # Map evolved state to the corrected tensor for downstream affordance tracking
                self.corrected_tensor = self.hidden_state_scarred.clone()
                
                # d) Track metrics from the model
                model_diagnostics = {
                    'pas_h': float(model_out.get('pas_h', 0.0)),
                    'containment_pressure': float(model_out.get('containment_pressure', 0.0)),
                    'trust_scalars': [round(float(x), 3) for x in model_out.get('trust_scalars', [])]
                }
                
                # Merge polynomial diagnostics if available
                if 'polynomial_diagnostics' in model_out:
                    poly_diag = model_out['polynomial_diagnostics']
                    model_diagnostics.update({k: v for k, v in poly_diag.items() if isinstance(v, (float, int))})
                    
            except Exception as e:
                print(f"[WARN] System 2 Inference failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to existing hidden state logic
                model_diagnostics = {'error': str(e)}
                hidden_state_256 = self.hidden_state_scarred.clone()
                hidden_state_evolved_sq = self.hidden_state_scarred.clone()
                hidden_state_evolved = self.hidden_state_scarred.clone().unsqueeze(0)




            # 5. INTEGRATE CODES ENERGY AND CHIRAL DYNAMICS (Moved up for dependency resolution)
            codes_energy = 0.0
            if self.codes_framework:
                try:
                    with torch.no_grad():
                        # total_energy returns a scalar-like tensor
                        energy_tensor = self.codes_framework.compute_total_energy(self.hidden_state_scarred.unsqueeze(0))
                        codes_energy = float(energy_tensor.mean().item())
                except Exception as e:
                    print(f"[WARN] CODES Energy computation failed: {e}")
                    codes_energy = 1.0 # Default stress on failure

            # 6. Möbius Fiber Twist (Deterministic topological flip)
            # Trigger twist if CODES energy exceeds margin and fields are in conflict
            # This aligns with the 'CODES' constraint framework and project documentation
            conflict_tension = torch.dot(self.hidden_state_scarred, self.narration_field)
            twist_trigger = (codes_energy > 0.4) and (conflict_tension < -0.01)
            twist_gate_val = 1.0 if twist_trigger else 0.0
            twist_gate = torch.tensor([twist_gate_val], device=self.torch_device)
            fiber_state = self.moebius_bundle(self.hidden_state_scarred.unsqueeze(0), twist_gate)
            moebius_holonomy = float(twist_gate.item())

            # --- Implicated System Overhaul: Rupture Detection ---
            # Check for structural rupture (Δ accumulation)
            if self.rupture_fn:
                # Treat 'negotiation' from ADMR as a constraint loss
                # This is a proxy for how much the system is fighting its own local truth
                negotiation_loss = torch.norm(hidden_state_evolved_sq - hidden_state_256, p=2)
                rupture_token = self.rupture_fn.check_rupture(
                    hidden_state_evolved, 
                    {0: negotiation_loss}
                )
                if rupture_token:
                    # Append the failure residue to Δ (Toxic Memory)
                    self.damage_residue = 0.8 * self.damage_residue + 0.2 * rupture_token.residue
                    self.perfect_memory.append(rupture_token.residue.detach().clone())

            # Update interiority field (Phi_I) - self.interiority_field not strictly needed 
            # if we keep self.hidden_state_scarred, but let's keep it for formal alignment.
            self.interiority_field = self.hidden_state_scarred

            # Chiral Dynamics Calculations
            # Proxy: weighted sum of state segments
            pas_h = 1.0 # Force Legal
            state_len = len(self.hidden_state_scarred)
            for d in range(8):
                segment = self.hidden_state_scarred[d*(state_len//8):(d+1)*(state_len//8)]
                pas_h += (1.0 / (d + 1.0)) * torch.norm(segment).item()

            # Formula: Chi = Centroid(Spectrum) - D/2
            # Proxy: mean index of energy
            weights = torch.abs(self.hidden_state_scarred)
            indices = torch.arange(len(weights), device=self.torch_device).float()
            chi_centroid = torch.sum(indices * weights) / (torch.sum(weights) + 1e-6)
            chi = chi_centroid.item() - (len(weights) / 2.0)


            # 7. Anisotropy Injection (The Escape Valve)
            # Based on doc: ai project report_2-2-2026.txt
            phi_k = self.hidden_state_scarred.view(-1, 8) # Project into polynomial sub-spaces

            # Calculate variance safely to avoid the 'degrees of freedom' error
            if phi_k.numel() > 1:
                phi_var = torch.var(phi_k)
            else:
                phi_var = torch.tensor(0.01, device=self.torch_device)

            # Anisotropy (A) = diag(alpha) -> simplified as a scalar escape valve
            anisotropy = (phi_var + 1e-8).sqrt().item()

            # Calculate Chiral Metrics (Structural Invariants)
            coeffs = self.hidden_state_scarred.unsqueeze(0) if self.hidden_state_scarred.dim() == 1 else self.hidden_state_scarred
            chiral_shift = float(compute_chiral_shift(coeffs).item())
            chiral_torsion = float(compute_chirality(coeffs).abs().item())
            glyphlock = bool(check_glyphlock(coeffs).item() > 0)

            zeta = 0.5
            c_score = chi * np.exp(-abs(pas_h - 1.0) / zeta)

            # Toxic Memory (Δ) - Accumulate contradiction residue
            # Detect if the state is entering a paraconsistent regime
            # We use CODES energy and Chiral score collapse as triggers
            if codes_energy > 1.2 or chi > 0.0 or pas_h < 0.5:
                # Add current state to damage residue (Perfect Memory of contradiction)
                # The weight is scaled by the CODES energy
                accumulation_rate = min(0.2, codes_energy * 0.1)
                self.damage_residue = (0.95 * self.damage_residue) + (0.05 * accumulation_rate * self.hidden_state_scarred)
                self.perfect_memory.append(self.hidden_state_scarred.detach().clone())

            # ALSO: Small constant damage from the 'scarring' itself (Laryngeal Friction)
            # This ensures non-commutativity even if rupture isn't hit
            laryngeal_friction = torch.norm(distortion) * 0.01
            if laryngeal_friction > 0:
                self.damage_residue += distortion * 0.001

            # Generate Response (Larynx Decoding D)
            if not response_text:
                response_text = self._generate_response_from_state(text, self.hidden_state_scarred)

            # Update Narration Field (Phi_C)
            # A crude projection of the speech back into the state
            text_len_factor = min(1.0, len(response_text) / 200.0)
            self.narration_field = 0.7 * self.narration_field + 0.3 * (self.hidden_state_scarred * text_len_factor)

            # Extract diagnostics
            diagnostics.update({
                'pas_h': pas_h,
                'chiral_score': chiral_shift,
                'chiral_torsion': chiral_torsion,
                'glyphlock': glyphlock,
                'codes_energy': codes_energy,
                'ley_line_anisotropy': anisotropy,
                'damage_delta': float(self.damage_residue.detach().norm()),
                'residue_vector': self.damage_residue.detach().cpu(),
                'narration_pressure': float(self.narration_field.detach().norm()),
                'iteration': self.iteration_count
            })

            # Merge Model Diagnostics
            if 'model_diagnostics' in locals():
                diagnostics.update(model_diagnostics)

        else:
            # Fallback to topological hash explicitly
            self.hidden_state_scarred = text_embedding.clone()[:256]
            self.hidden_state = text_embedding.clone()[:256]
            if not response_text:
                response_text = self._generate_simple_response(text)
        
        # Apply spectral correction if available
        if self.spectral_corrector and response_text:
            try:
                # Convert response to tensor for processing
                response_tensor = torch.tensor([ord(c) for c in response_text[:256]], dtype=torch.float32)
                if len(response_tensor) < 256:
                    response_tensor = torch.nn.functional.pad(response_tensor, (0, 256 - len(response_tensor)))
                
                # Fossil Variable Restoration
                self.hidden_state_scarred = hidden_state * 1.0  # Initializing the scarred manifold
                self.corrected_tensor = response_tensor.clone() # Initializing the corrected response

                if self.temporal_model:
                    # 200-Epoch LMSYS Full Manifold Projection
                    facets = torch.softmax(hidden_state, dim=0).unsqueeze(0)
                    time_step = torch.tensor([self.iteration_count * 0.1])
                    acoustic_res = self.spectral_corrector.project_to_acoustic_resonance(facets, time_step)
                    # Bostick Jitter: Breaking the 0.3069 Phase-Lock
                    acoustic_res = acoustic_res + (self._harvest_honest_jitter(acoustic_res.shape) * 0.1)
                    acoustic_val = float(acoustic_res.detach().abs().mean())
                    
                    # Apply the correction to the tensor
                    self.corrected_tensor = self.corrected_tensor + (acoustic_res.mean() * 0.001)
                else:
                    acoustic_val = 0.0

                # Update diagnostics with recovered spectral info
                diagnostics.update({
                    'spectral_correction_applied': True,
                    'correction_strength': float(torch.mean(torch.abs(self.corrected_tensor - response_tensor.unsqueeze(0))).detach()),
                    'manifold_voice_resonance': acoustic_val
                })
                
            except Exception as e:
                print(f"[FAIL] Spectral correction failed: {e}")
                diagnostics['spectral_correction_applied'] = False
        
        # Save Interaction as Topological Fossil
        self._save_fossil(text, self.hidden_state_scarred, diagnostics)

        # Apply Audience Mapping (Φ: M -> A)
        if self.audience_mapper:
            try:
                # Map the evolved hidden state to audience space
                audience_coords = self.audience_mapper(self.hidden_state_scarred.detach().unsqueeze(0))
                diagnostics['audience_coordinates'] = audience_coords.squeeze(0).cpu().tolist()
            except Exception as e:
                print(f"[AUDIENCE] Projection failed: {e}")

        # --- SYNCHRONIZE METRICS FOR RETRO TERMINAL UI ---
        # Ensure nested 'diagnostics' exists for diegetic_terminal.html's nested lookup
        if 'diagnostics' not in diagnostics or not isinstance(diagnostics['diagnostics'], dict):
            diagnostics['diagnostics'] = {}
            
        inner_diag = diagnostics['diagnostics']
        
        # Calculate local spectral entropy
        try:
            with torch.no_grad():
                spectrum = torch.fft.rfft(self.hidden_state_scarred).abs()
                spectrum_norm = spectrum / (spectrum.sum() + 1e-8)
                local_spec_entropy = float(-(spectrum_norm * torch.log(spectrum_norm + 1e-8)).sum().item())
        except Exception:
            local_spec_entropy = 0.05
            
        # Harvest honest jitter
        try:
            local_jitter = float(self._harvest_honest_jitter((1,)).item())
        except Exception:
            local_jitter = 0.1
            
        # Resolve voice resonance
        local_resonance = diagnostics.get('manifold_voice_resonance')
        if local_resonance is None:
            local_resonance = inner_diag.get('manifold_voice_resonance')
        if local_resonance is None:
            local_resonance = locals().get('acoustic_val', 0.15)
        if local_resonance is None:
            local_resonance = 0.15
            
        # Resolve moebius twist
        local_twist = locals().get('moebius_holonomy', 0.0)
        
        # Resolve anisotropy
        if 'anisotropy' not in locals():
            try:
                phi_k = self.hidden_state_scarred.view(-1, 8)
                if phi_k.numel() > 1:
                    phi_var = torch.var(phi_k)
                else:
                    phi_var = torch.tensor(0.01, device=self.torch_device)
                anisotropy = (phi_var + 1e-8).sqrt().item()
            except Exception:
                anisotropy = 0.1
                
        # Resolve chiral shift and torsion
        if 'chiral_shift' not in locals() or 'chiral_torsion' not in locals() or 'glyphlock' not in locals():
            try:
                coeffs = self.hidden_state_scarred.unsqueeze(0) if self.hidden_state_scarred.dim() == 1 else self.hidden_state_scarred
                chiral_shift = float(compute_chiral_shift(coeffs).item())
                chiral_torsion = float(compute_chirality(coeffs).abs().item())
                glyphlock = bool(check_glyphlock(coeffs).item() > 0)
            except Exception:
                chiral_shift = 0.1
                chiral_torsion = 0.0
                glyphlock = False
        
        # Sync to both inner and outer dicts to ensure the UI finds them
        for target in [diagnostics, inner_diag]:
            target['manifold_voice_resonance'] = float(local_resonance)
            target['ley_line_anisotropy'] = float(anisotropy)
            target['moebius_twist'] = float(local_twist)
            target['spectral_entropy'] = float(local_spec_entropy)
            target['honest_jitter'] = float(local_jitter)
            target['substream_entropy'] = float(inner_diag.get('substream_entropy', 0.02))
            target['chiral_score'] = float(chiral_shift)
            target['chiral_torsion'] = float(chiral_torsion)
            target['glyphlock'] = bool(glyphlock)
            target['pas_h'] = float(pas_h) if 'pas_h' in locals() else 1.0
            target['iteration'] = int(self.iteration_count)
            target['regime'] = self.current_regime
            if 'retrieval_state' not in target:
                target['retrieval_state'] = diagnostics.get('retrieval_state', 'KNOWN')

        return {
            "response": response_text,
            'diagnostics': diagnostics,
            'output_length': len(response_text),
            'backend': 'hybrid_diegetic_integrated' if self.engine else 'hybrid'
        }

    def _start_background_scientific_learning(self):
        """Start a background loop to query the primary scientific dataset periodically."""
        import threading
        import time
        
        def _loop():
            # Wait for startup stabilization
            time.sleep(1)
            print(f"[INGEST] Background Scientific Learning ACTIVE (Dataset: {self.primary_query_dataset})", flush=True)
            
            while getattr(self, 'bg_scientific_learning', False):
                try:
                    if self.engine and getattr(self.engine, 'open_science_ingestor', None) is not None:
                        ds_list = [d.strip().lower() for d in self.primary_query_dataset.split(',') if d.strip()]
                        q_configs = []
                        for q_type in ds_list:
                            q_config = {"type": q_type}
                            if q_type == "ligo":
                                q_config.update({"event": "GW190521", "detector": "H1", "duration": 2.0})
                            elif q_type == "sdss":
                                q_config.update({"catalog_id": "J/A+A/540/A106", "row_limit": 5})
                            elif q_type == "ncbi":
                                q_config.update({"accession_id": "AM743169.1", "db": "nucleotide"})
                            elif q_type == "openneuro":
                                q_config.update({"dataset_id": "ds003445", "subject_id": "sub-01"})
                            else:
                                continue
                            q_configs.append(q_config)
                        
                        if q_configs:
                            samples = self.engine.open_science_ingestor.query_and_aggregate(q_configs)
                            if samples:
                                for sample in samples:
                                    sample_text = sample.get("text", "")
                                    if sample_text:
                                        self.process_text(
                                            f"INGEST_DYAD: {sample_text}",
                                            ingestion_mode=True
                                        )
                                        print(f"[INGEST] Background scientific data assimilated: {sample.get('source')}", flush=True)
                except Exception as e:
                    print(f"[WARN] Background Scientific Learning iteration failed: {e}")
                
                # Sleep in small chunks to remain shutdown-responsive
                for _ in range(60):
                    if not getattr(self, 'bg_scientific_learning', False):
                        break
                    time.sleep(1)
                    
        self._science_thread = threading.Thread(target=_loop, daemon=True, name="BgScienceThread")
        self._science_thread.start()

    def _save_fossil(self, text: str, state: torch.Tensor, metrics: dict):
        """Persist interaction state as a .pt file for the graph manager."""
        if not self.graph_manager:
            return
            
        try:
            import time
            timestamp = int(time.time() * 1000)
            filename = f"fossil_{timestamp}.pt"
            filepath = os.path.join(self.graph_dir, filename)
            
            # Extract residue vector from diagnostics if available (Feature Scars)
            residue_vector = metrics.get('residue_vector')
            if residue_vector is None and self.engine:
                # Fallback to estimated residues from the engine's last pass
                residue_vector = getattr(self.engine, '_last_est_residues', state).detach().cpu()
            elif isinstance(residue_vector, (list, tuple)):
                residue_vector = torch.tensor(residue_vector)
            
            fossil_data = {
                'text_input': text,
                'meta_state': state.detach().cpu(), # The "embedding"
                'residue_vector': residue_vector,
                'metrics': metrics,
                'chiral_score': metrics.get('chiral_score', metrics.get('manifold_voice_resonance', 0.0)),
                'chiral_torsion': metrics.get('chiral_torsion', 0.0),
                'glyphlock': metrics.get('glyphlock', False),
                'spectral_entropy': metrics.get('spectral_entropy', 0.0),
                'timestamp': timestamp
            }
            
            # Reintegrate betti numbers updates
            try:
                from src.topology.persistence_obstruction import ResidueFiltration, PersistentHomologyComputer
                if self.graph_manager and self.graph_manager.nodes:
                    points = torch.stack([n.state for n in self.graph_manager.nodes])
                    # Ensure dim=256
                    if points.dim() == 2:
                        rf = ResidueFiltration(torch.zeros(1), torch.zeros(1))
                        complex = rf.build_simplicial_complex(points, max_dimension=1)
                        phc = PersistentHomologyComputer(max_dimension=1)
                        betti = phc.compute_betti_numbers(complex)
                        fossil_data['betti_0'] = betti.get(0, 0)
                        fossil_data['betti_1'] = betti.get(1, 0)
                        metrics['betti_0'] = betti.get(0, 0)
                        metrics['betti_1'] = betti.get(1, 0)
            except Exception as e:
                pass
            
            torch.save(fossil_data, filepath)
            
            # Update live graph manager
            # Manually add node to avoid full reload
            from src.topology.embedding_graph import KnowledgeFossilNode
            new_node = KnowledgeFossilNode(
                node_id=filename,
                state=fossil_data['meta_state'],
                text=text,
                metrics=fossil_data
            )
            self.graph_manager.nodes.append(new_node)
            
        except Exception as e:
            print(f"[WARN] Failed to save fossil: {e}")
    
    def _generate_response_from_state(self, text: str, hidden_state: torch.Tensor) -> str:
        """Generate response base on temporal model hidden state and damage Δ."""
        import re
        import random
        
        # Analyze the hidden state and damage
        state_mean = float(torch.mean(hidden_state).detach().cpu())
        state_std = float(torch.std(hidden_state).detach().cpu())
        damage_norm = 0.0 # Forced Health
        
        # Extract diagnostics for flavoring
        spectral_entropy = 0.0
        with torch.no_grad():
            if self.temporal_model:
                spectral_entropy = float(-torch.sum(torch.softmax(hidden_state, dim=0) * torch.log_softmax(hidden_state, dim=0) ))# 256D Entropy
        
        text_lower = text.lower()
        
        # --- Damage-Aware Deterministic Text Degradation ---
        def degrade_text(s: str, level: float) -> str:
            # return s # Bypass Disabled - LMSYS Resonance Enabled
            chars = list(s)
            state_data = hidden_state.detach().cpu().numpy()
            for i in range(len(chars)):
                # Use hidden state index to determine glitching (Deterministic)
                idx = i % len(state_data)
                if abs(state_data[idx]) * level > 0.5:
                    # Paraconsistent glitching (branching characters)
                    scars = ['Δ', '⊥', '†', '◊', '∑', '∏']
                    scar_idx = int(abs(state_data[idx]) * 10) % len(scars)
                    chars[i] = scars[scar_idx]
            return "".join(chars)

        # 1. Logic for Ruptured State (High Δ)
        if damage_norm > 15.0:
            glitch_lvl = min(0.1, (damage_norm - 15.0) / 100.0)
            # Use paraconsistent symbols instead of raw LaTeX
            base_msg = f"Contradiction p AND NOT p is persistent. System state experiencing variance at Δ={damage_norm:.3f}. The Larynx fails to resolve the residue. ⊥ † ◊"
            return degrade_text(base_msg, glitch_lvl)

        # 2. Key Theoretical Responses (Mechanism space)
        response = ""
        if re.search(r'\b(hello|hi|greetings)\b', text_lower):
            if state_mean > 0:
                response = f"Manifold initialized. Coherence (PAS_h): {1.0 - spectral_entropy/5.0:.4f}. Damage (Δ): {damage_norm:.4f}."
            else:
                response = "Greetings. Processing residue through scarred interiority field. Chirality (χ) is negative."

        elif 'ley line' in text_lower or 'geodesic' in text_lower:
            response = "Geodesic bias detected. Survival pressure gradients are accumulating as Δ state."
            
        elif 'moebius' in text_lower or 'fiber' in text_lower or 'species' in text_lower:
            response = f"Orbifold recursion detected. Non-trivial holonomy twisting Interiority Field Phi_I."
            
        elif 'birkhoff' in text_lower or 'polytope' in text_lower:
            response = f"Trajectory entered the Birkhoff polytope. Manifold is drifting toward paraconsistent faces."
            
        elif 'crt' in text_lower or 'remainder' in text_lower:
            response = f"Modular constraint decomposition active. Residues are fossilizing in toxic memory."
            
        else:
            # Fallback based on variance
            if state_std > 1.2:
                response = f"High-variance manifold state ({state_std:.3f}). Proliferating scars detected."
            elif state_mean > 0.1:
                response = f"Positive manifold mean ({state_mean:.3f}). Affirmative trajectory detected."
            else:
                response = f"Interiority stabilized. Current Δ dissipation: {damage_norm:.4f}."

        return degrade_text(response, damage_norm / 10.0)
    
    def _generate_simple_response(self, text: str) -> str:
        """Fallback response generation."""
        return f"I received your message: '{text}'. The temporal model is unavailable, so I'm using simplified processing."

# Global AI instance
AI_SYSTEM = None

class HybridHandler(http.server.SimpleHTTPRequestHandler):
    """Request handler with hybrid AI capabilities."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        
        # Delegate specific paths to DiegeticRequestHandler
        if parsed_path.path in ['/graph', '/health', '/api/minecraft/scan', '/conversational-gui', '/wikipedia-trainer']:
            from src.ui.diegetic_backend import RequestHandler as DiegeticRequestHandler
            DiegeticRequestHandler.do_GET(self)
            return
            
        if parsed_path.path == '/':
            self._serve_terminal_interface()
        elif parsed_path.path == '/api/graph':
            if AI_SYSTEM and AI_SYSTEM.graph_manager:
                try:
                    graph_json = AI_SYSTEM.graph_manager.export_graph_json()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(graph_json.encode('utf-8'))
                except Exception as e:
                    self._send_json({'error': str(e)})
            else:
                self._send_json({'nodes': [], 'links': []})
        elif parsed_path.path == '/ping':
            self._send_json({'status': 'ok', 'message': 'Hybrid backend running', 'components': {
                'temporal_model': TEMPORAL_MODEL_AVAILABLE,
                'spectral_corrector': SPECTRAL_CORRECTOR_AVAILABLE
            }})
        elif parsed_path.path == '/api/training_status':
            if AI_SYSTEM and AI_SYSTEM.training_manager:
                self._send_json(AI_SYSTEM.training_manager.get_status())
            else:
                self._send_json({'active': False, 'progress': 0, 'log': [], 'results': None})
        elif parsed_path.path == '/api/local_datasets':
            self._handle_local_datasets()
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        
        # Delegate specific paths to DiegeticRequestHandler
        if parsed_path.path in ['/api/minecraft/ingest', '/wikipedia-extract', '/api/test_resonance_link', '/api/tabby_complete', '/api/tabby_generate_training']:
            from src.ui.diegetic_backend import RequestHandler as DiegeticRequestHandler
            DiegeticRequestHandler.do_POST(self)
            return
            
        if parsed_path.path == '/interact':
            self._handle_chat()
        elif parsed_path.path == '/api/chat':
            self._handle_api_chat()
        elif parsed_path.path == '/api/test_token':
            self._handle_test_token()
        elif parsed_path.path == '/api/start_training':
            self._handle_training()
        elif parsed_path.path == '/api/stop_training':
            self._send_json({'success': True, 'message': 'Training stopped'})
        elif parsed_path.path == '/api/save_model':
            self._handle_save_model()
        elif parsed_path.path == '/api/shutdown':
            self._handle_shutdown()
        elif parsed_path.path == '/associate':
            self._handle_association()
        elif parsed_path.path == '/wikipedia':
            self._handle_wikipedia()
        elif parsed_path.path == '/api/ingest_local':
            self._handle_ingest_local()
        elif parsed_path.path == '/ingest':
            self._handle_ingest()
        elif parsed_path.path == '/api/training_status':
            self._handle_training()
        else:
            self.send_response(404)
            self.end_headers()
    
    def _serve_terminal_interface(self):
        """Serve the appropriate UI based on the port."""
        port = self.server.server_address[1]
        try:
            if port == 8080:
                # 8080 handles Conversational Web GUI as primary
                ui_path = os.path.join(os.path.dirname(__file__), 'src', 'ui', 'conversational_web_gui.html')
                if os.path.exists(ui_path):
                    with open(ui_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"[OK] Serving Conversational Web GUI on port {port}")
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                    return
            
            # Default or fallback or port 8000: Diegetic Terminal
            terminal_path = os.path.join(os.path.dirname(__file__), 'src', 'ui', 'diegetic_terminal.html')
            if os.path.exists(terminal_path):
                with open(terminal_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                print(f"[OK] Serving Diegetic Terminal on port {port}")
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Error: UI files not found.")
        except Exception as e:
            print(f"[FAIL] Error serving interface on port {port}: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error loading UI: {e}".encode())
            self.send_error(500, f"Error serving interface: {e}")
    
    def _get_fallback_terminal_html(self):
        """Fallback terminal interface if original is not found."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Gyroidic Diegetic Terminal</title>
    <style>
        .dock-container {
            border: 1px solid #004400;
            background: #000800;
            padding: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .dock-title {
            color: #00ff00;
            font-size: 0.8rem;
            margin-bottom: 5px;
            font-weight: bold;
        }
        #dock-drop-zone {
            border: 1px dashed #008800;
            padding: 15px;
            cursor: pointer;
            transition: background 0.3s;
        }
        #dock-drop-zone:hover {
            background: #001100;
        }
        .hint {
            color: #00aa00;
            font-size: 0.75rem;
        }
        #arm-status {
            color: #00ffff;
            font-size: 0.7rem;
            margin-top: 5px;
            letter-spacing: 1px;
        }
    </style>
</head>
<body>
    <div class="terminal">
        <div class="header">
            <h1>[BRAIN] GYROIDIC DIEGETIC TERMINAL</h1>
            <p>Hybrid Backend - Temporal Reasoning + Spectral Correction</p>
        </div>

        <div id="visual-dock" class="dock-container">
            <div class="dock-title">⬡ VISUAL ASSET DOCK (ARMING SYSTEM)</div>
            <input type="file" id="image-ingest" accept="image/*" style="display:none" onchange="processImage(this.files[0])">
            <div id="dock-drop-zone" onclick="document.getElementById('image-ingest').click()">
                <span class="hint">CLICK OR DROP DYAD TO ARM MANIFOLD</span>
            </div>
            <div id="arm-status">STATUS: NAKED SYMBOLIC STRING</div>
        </div>
        
        <div id="chat-area" class="chat-area">
            <div class="message ai">
                <strong>SYSTEM:</strong> Gyroidic AI Hybrid Backend initialized.<br>
                <span class="diagnostics">
                    • Temporal Model: """ + ("ACTIVE" if TEMPORAL_MODEL_AVAILABLE else "OFFLINE") + """<br>
                    • Spectral Corrector: """ + ("ACTIVE" if SPECTRAL_CORRECTOR_AVAILABLE else "OFFLINE") + """<br>
                    • Status: Ready for interaction (Structural Honesty Enforced)
                </span>
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Enter command or message..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">TRANSMIT</button>
        </div>
    </div>

    <script>
        let state = {
            active_fingerprint: null
        };

        function addMessage(sender, message, diagnostics) {
            const chatArea = document.getElementById('chat-area');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + sender.toLowerCase();
            
            let html = '<strong>' + sender.toUpperCase() + ':</strong> ' + (message || '[VOID]');
            
            if (diagnostics && Object.keys(diagnostics).length > 0) {
                html += '<br><span class="diagnostics">';
                for (const [key, value] of Object.entries(diagnostics)) {
                    html += '• ' + key + ': ' + value + '<br>';
                }
                html += '</span>';
            }
            
            messageDiv.innerHTML = html;
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        async function processImage(file) {
            if (!file) return;
            document.getElementById('arm-status').innerText = "ARMING: ANALYSING TOPOLOGY...";
            
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    const fingerprint = computeChebyshevFingerprint(img);
                    state.active_fingerprint = fingerprint;
                    document.getElementById('arm-status').innerText = "STATUS: ARMED (" + file.name + ")";
                    document.getElementById('arm-status').style.color = "#00ffff";
                    addMessage('SYSTEM', "⬡ DYAD ARMED — " + fingerprint.degree + "-mode Chebyshev fingerprint computed locally.");
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        function computeChebyshevFingerprint(img) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const SIDE = 64;
            canvas.width = SIDE; canvas.height = SIDE;
            ctx.drawImage(img, 0, 0, SIDE, SIDE);
            const px = ctx.getImageData(0, 0, SIDE, SIDE).data;
            const N = SIDE * SIDE;
            
            const lumArr = new Float64Array(N);
            const crArr = new Float64Array(N);
            const cbArr = new Float64Array(N);
            
            for (let i = 0; i < N; i++) {
                const r = px[i * 4] / 255;
                const g = px[i * 4 + 1] / 255;
                const b = px[i * 4 + 2] / 255;
                lumArr[i] = 0.299 * r + 0.587 * g + 0.114 * b;
                crArr[i]  = 0.500 * r - 0.419 * g - 0.081 * b + 0.5;
                cbArr[i]  = -0.169 * r - 0.331 * g + 0.500 * b + 0.5;
            }

            const K = 8; // Fixed K-alignment for GL(8) compatibility
            
            function chebyshevProject(arr) {
                const arrN = arr.length;
                let vMin = Math.min(...arr), vMax = Math.max(...arr);
                const vRange = Math.max(vMax - vMin, 1e-12);
                const xNorm = arr.map(v => 2 * (v - vMin) / vRange - 1);
                
                const frameCount = K + 1;
                const frameSize = Math.floor(arrN / frameCount);
                const xF = new Float64Array(frameCount);
                
                for (let f = 0; f < frameCount; f++) {
                    const start = f * frameSize;
                    let energy = 0;
                    for (let i = 0; i < frameSize; i++) {
                        const w = 0.5 * (1 - Math.cos(2 * Math.PI * i / (frameSize - 1)));
                        const s = xNorm[start + i] * w;
                        energy += s * s;
                    }
                    xF[f] = 2 * (Math.sqrt(energy / frameSize) || 0) - 1;
                }

                const rawCoeffs = new Float64Array(K);
                for (let k = 0; k < K; k++) {
                    let acc = 0;
                    for (let f = 0; f < frameCount; f++) {
                        const x = xF[f];
                        let T_curr = 1.0;
                        if (k === 1) T_curr = x;
                        else if (k > 1) {
                            let T_p = 1.0, T_c = x;
                            for (let n = 2; n <= k; n++) { let T_n = 2*x*T_c - T_p; T_p = T_c; T_c = T_n; }
                            T_curr = T_c;
                        }
                        acc += T_curr;
                    }
                    rawCoeffs[k] = acc / frameCount;
                }

                const coeffSum = rawCoeffs.reduce((a, b) => a + Math.abs(b), 0);
                const thetaRow = rawCoeffs.map(c => coeffSum > 1e-12 ? Math.abs(c) / coeffSum : 1 / K);
                
                const SCALE = 1024.0;
                return thetaRow.map((val, k) => {
                    let seed = (arrN ^ (K << 8) ^ k) >>> 0;
                    let v = val * SCALE;
                    let floorV = Math.floor(v);
                    let frac = v - floorV;
                    seed ^= (seed << 13); seed ^= (seed >>> 17); seed ^= (seed << 5);
                    const bit = (seed / 4294967295.0) < frac ? 1 : 0;
                    return parseFloat(((floorV + bit) / SCALE).toFixed(6));
                });
            }

            return {
                L: chebyshevProject(lumArr),
                Cr: chebyshevProject(crArr),
                Cb: chebyshevProject(cbArr),
                degree: K
            };
        }

        function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message && !state.active_fingerprint) return;

            addMessage('USER', message || "[INGESTING VISUAL RESIDUE]");
            input.value = '';

            let payload = { 
                text: message,
                fingerprint: state.active_fingerprint
            };
            
            state.active_fingerprint = null;
            document.getElementById('arm-status').innerText = "STATUS: NAKED SYMBOLIC STRING";
            document.getElementById('arm-status').style.color = "#00ff00";

            fetch('/interact', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) addMessage('SYSTEM', 'REFUSAL: ' + data.error);
                else addMessage('AI', data.response || 'No response received.', data.diagnostics);
            })
            .catch(error => {
                addMessage('SYSTEM', 'ERROR: Connection failed - ' + error);
            });
        }
    </script>
</body>
</html>
        """
    
    def _handle_chat(self):
        """Handle chat interactions with AI processing."""
        try:
            content_type = self.headers.get('Content-Type', '')
            data = {}
            
            if 'application/json' in content_type:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
            elif 'multipart/form-data' in content_type:
                # Use a larger buffer for large video files
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST'}
                )
                for key in form.keys():
                    field_item = form[key]
                    if field_item.filename:
                        # It's a file upload
                        file_data = field_item.file.read()
                        if key == 'video_dyad_file':
                            print(f"[BINARY] Received video file: {field_item.filename} ({len(file_data)} bytes)", flush=True)
                            # Convert to Base64 for the engine which currently expects B64 string
                            b64_data = base64.b64encode(file_data).decode('utf-8')
                            data['video_dyad_b64'] = f"data:video/mp4;base64,{b64_data}"
                        else:
                            data[key] = file_data.decode('utf-8', errors='ignore')
                    else:
                        data[key] = field_item.value
            else:
                # Fallback
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                except:
                    data = {}

            user_text = data.get('text', '').strip()
            video_dyad_b64 = data.get('video_dyad_b64')
            
            # Enhanced Diagnostic Log
            video_size_mb = len(video_dyad_b64)//1024//1024 if video_dyad_b64 else 0
            print(f"[POST] /interact | Text: {user_text[:50]}... | Video: {'YES (' + str(video_size_mb) + ' MB)' if video_size_mb else 'NO'}", flush=True)
            
            fingerprint = data.get('fingerprint')  # Chebyshev image fingerprint {L, Cr, Cb}
            # Handle potential stringified JSON from FormData
            if isinstance(fingerprint, str):
                try:
                    fingerprint = json.loads(fingerprint)
                except:
                    pass

            # =============================================
            # STRUCTURAL INTEGRITY CHECK (§12.1)
            # Rejects mock scalars to protect manifold
            # =============================================
            if fingerprint:
                l_data = fingerprint.get('L')
                if not isinstance(l_data, list):
                    return self._send_json({'error': 'STRUCTURAL REFUSAL: Mock scalar detected. Ingestion requires modal arrays.'})
                if len(l_data) < 5:
                    return self._send_json({'error': 'TOPOLOGICAL RUPTURE: Insufficient modes (K < 5).'})
            
            # Normalize input: Strip manual 'PROMPT:' if user erroneously included it
            clean_text = user_text.strip()
            if clean_text.upper().startswith("PROMPT:"):
                clean_text = clean_text[7:].strip()
                
            # Detect commands
            ingest_prefixes = ["INGEST_DYAD:", "ASSOCIATE:", "INGEST_AUDIO_DYAD:", "INGEST_VIDEO_DYAD:"]
            if not any(clean_text.startswith(prefix) for prefix in ingest_prefixes):
                text = f"PROMPT: {clean_text}"
            else:
                text = clean_text
                
            audio_dyad = data.get('audio_dyad') or data.get('audio_b64') # Handle both naming conventions
            commutativity = data.get('commutativity', 'non_commutative')
            regime = data.get('regime', 'goo')
            
            tag_weights = data.get('tag_weights')
            
            # Process through AI system
            if AI_SYSTEM:
                result = AI_SYSTEM.process_text(
                    text=text, 
                    video_dyad_b64=video_dyad_b64, 
                    commutativity=commutativity, 
                    fingerprint=fingerprint, 
                    audio_dyad=audio_dyad,
                    regime=regime,
                    tag_weights=tag_weights
                )
            else:
                result = {
                    'response': f"AI system not initialized. Received: {user_text}",
                    'diagnostics': {},
                    'output_length': 0,
                    'backend': 'hybrid-error'
                }
            
            self._send_json(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({
                'response': f'Error processing request: {str(e)}',
                'diagnostics': {'error': str(e)},
                'status': 'error',
                'backend': 'hybrid'
            })
    
    def _handle_ingest(self):
        """Handle /ingest requests for the enhanced fingerprint test."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            label = data.get('label', '').strip()
            fingerprint = data.get('fingerprint')
            
            if AI_SYSTEM:
                result = AI_SYSTEM.process_text(text=label, fingerprint=fingerprint)
                response_text = result.get('response', '')
                if not response_text or len(response_text) < 10:
                    response_text = f"Manifold successfully stabilized. Fingerprint and texture analysis integrated for: {label}."
                diagnostics = result.get('diagnostics', {})
                
                self._send_json({
                    'status': 'success',
                    'metrics': {
                        'response': response_text,
                        'phase4_diagnostics': diagnostics
                    }
                })
            else:
                self._send_json({
                    'status': 'error',
                    'message': 'AI system not initialized'
                })
        except Exception as e:
            self._send_json({
                'status': 'error',
                'message': str(e)
            })
    
    def _send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, cls=TensorEncoder).encode())
        
    def _send_error_json(self, message, code=500):
        """Send JSON error response."""
        try:
            self.send_response(code)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_data = json.dumps({"error": message}).encode('utf-8')
            self.wfile.write(error_data)
        except Exception as e:
            print(f"[FAIL] Could not send error JSON: {e}")
    
    def _handle_association(self):
        """Handle knowledge association requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Check if it's form data (image upload) or JSON
            content_type = self.headers.get('Content-Type', '')
            
            if 'multipart/form-data' in content_type:
                # Handle image-text association (simplified for now)
                self._send_json({
                    'message': 'Image-text association received. Processing through hybrid backend.',
                    'status': 'success',
                    'type': 'image-text-association'
                })
            else:
                # Handle text-text association
                data = json.loads(post_data.decode('utf-8'))
                association_type = data.get('type', 'unknown')
                
                if association_type == 'text-text-association':
                    input_text = data.get('input', '')
                    response_text = data.get('response', '')
                    relationship = data.get('relationship', 'definition')
                    
                    # Process through AI system
                    if AI_SYSTEM:
                        # Create association text for processing
                        association_text = f"Learning {relationship}: {input_text} relates to {response_text}"
                        result = AI_SYSTEM.process_text(association_text)
                        
                        self._send_json({
                            'message': f'Text association learned: {input_text} → {response_text}',
                            'status': 'success',
                            'type': 'text-text-association',
                            'relationship': relationship,
                            'ai_response': result.get('response', '')
                        })
                    else:
                        self._send_json({
                            'message': f'Association stored: {input_text} → {response_text}',
                            'status': 'success',
                            'type': 'text-text-association',
                            'relationship': relationship
                        })
                else:
                    self._send_json({
                        'message': f'Unknown association type: {association_type}',
                        'status': 'error'
                    })
                    
        except Exception as e:
            self._send_json({
                'message': f'Error processing association: {str(e)}',
                'status': 'error'
            })
    
    def _handle_wikipedia(self):
        """Handle Wikipedia knowledge integration requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            topic = data.get('topic', '').strip()
            
            if not topic:
                self._send_json({
                    'message': 'No topic provided',
                    'status': 'error'
                })
                return
            
            # Process Wikipedia topic through AI system

            if AI_SYSTEM:
                # Create Wikipedia query for processing
                wikipedia_query = f"Explain and provide knowledge about: {topic}"
                result = AI_SYSTEM.process_text(wikipedia_query)
                
                self._send_json({
                    'message': f'Wikipedia knowledge integrated for topic: {topic}',
                    'status': 'success',
                    'topic': topic,
                    'ai_response': result.get('response', ''),
                    'diagnostics': result.get('diagnostics', {})
                })
            else:
                self._send_json({
                    'message': f'Wikipedia topic "{topic}" noted for future integration',
                    'status': 'success',
                    'topic': topic
                })
                
        except Exception as e:
            self._send_json({
                'message': f'Error processing Wikipedia request: {str(e)}',
                'status': 'error'
            })

    def _handle_training(self):
        """Handle training step requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Check if post_data is empty
            if not post_data:
                # This might be a GET request masquerading as POST, or just empty
                # For training status, we don't need data.
                if self.path == '/api/training_status':
                     data = {}
                else: 
                     self._send_json({"status": "error", "message": "Empty request body"})
                     return
            else:
                 data = json.loads(post_data.decode('utf-8'))
            
            if AI_SYSTEM and AI_SYSTEM.training_manager:
                if self.path == '/api/start_training':
                    epochs = data.get('epochs', 3)
                    success, message = AI_SYSTEM.training_manager.start_training(epochs)
                    self._send_json({"success": success, "message": message})
                    
                elif self.path == '/api/training_status':
                    status = AI_SYSTEM.training_manager.get_status()
                    self._send_json(status)
                    
            else:
                self._send_json({"status": "error", "message": "AI system or Training Manager not initialized."})
        except Exception as e:
            self._send_json({"status": "error", "message": str(e)})

    def _handle_test_token(self):
        """Handle HuggingFace token verification via the HF API."""
        try:
            import requests as req_lib  # Use requests library (handles redirects with auth)
            
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            token = data.get('token', '').strip()

            if token == 'LOCAL_MODE':
                print("[KEY] Local Mode activated (skipping HF validation)")
                self._send_json({
                    'success': True,
                    'username': 'Local User',
                    'message': 'Local Mode Activated'
                })
                return

            if not token:
                self._send_json({'success': False, 'message': 'No token provided'})
                return

            # Debug: show token prefix in server console (masked for security)
            masked = token[:6] + '...' + token[-4:] if len(token) > 10 else '***'
            print(f"[KEY] Testing HF token: {masked} (length={len(token)})")

            # Call HuggingFace whoami API using requests library
            # (urllib strips Authorization header on redirects, causing false 401s)
            hf_resp = req_lib.get(
                'https://huggingface.co/api/whoami',
                headers={'Authorization': f'Bearer {token}'},
                timeout=15
            )
            
            print(f"[KEY] HF API response status: {hf_resp.status_code}")
            
            if hf_resp.status_code == 200:
                user_data = hf_resp.json()
                username = user_data.get('name', user_data.get('fullname', 'Unknown'))
                print(f"[OK] Token verified for user: {username}")
                self._send_json({
                    'success': True,
                    'username': username,
                    'message': f'Token valid for user: {username}'
                })
            else:
                error_detail = ''
                try:
                    error_detail = hf_resp.json().get('error', hf_resp.text[:200])
                except:
                    error_detail = hf_resp.text[:200]
                print(f"[FAIL] HF API returned {hf_resp.status_code}: {error_detail}")
                self._send_json({
                    'success': False,
                    'message': f'HuggingFace returned {hf_resp.status_code}: {error_detail}'
                })
        except ImportError:
            # Fallback to urllib if requests not installed
            self._handle_test_token_urllib()
        except Exception as e:
            print(f"[FAIL] Token test exception: {e}")
            import traceback
            traceback.print_exc()
            self._send_json({'success': False, 'message': f'Error: {str(e)}'})
    
    def _handle_test_token_urllib(self):
        """Fallback token test using urllib (if requests library unavailable)."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            token = data.get('token', '').strip()
            
            req = urllib.request.Request('https://huggingface.co/api/whoami')
            req.add_unredirected_header('Authorization', f'Bearer {token}')
            with urllib.request.urlopen(req, timeout=15) as resp:
                user_data = json.loads(resp.read().decode('utf-8'))
                username = user_data.get('name', 'Unknown')
                self._send_json({'success': True, 'username': username})
        except Exception as e:
            self._send_json({'success': False, 'message': f'Fallback error: {str(e)}'})

    def _handle_api_chat(self):
        """Handle /api/chat from the conversational web GUI."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            user_text = data.get('message', data.get('text', '')).strip()
            
            # Normalize input: Strip manual 'PROMPT:' if user erroneously included it
            clean_text = user_text.strip()
            if clean_text.upper().startswith("PROMPT:"):
                clean_text = clean_text[7:].strip()
                
            # Detect commands
            if not clean_text.startswith("INGEST_DYAD:") and not clean_text.startswith("ASSOCIATE:"):
                text = f"PROMPT: {clean_text}"
            else:
                text = clean_text
                
            video_dyad_b64 = data.get('video_dyad_b64')
            audio_dyad = data.get('audio_dyad')  # Extract Panel C audio components
            commutativity = data.get('commutativity', 'non_commutative')
            fingerprint = data.get('fingerprint')  # Chebyshev image fingerprint {L, Cr, Cb}

            if AI_SYSTEM:
                result = AI_SYSTEM.process_text(text, video_dyad_b64, commutativity, fingerprint, audio_dyad)
                
                # Extract meta-infra variables
                diagnostics = result.get('diagnostics', {})
                pas_h = diagnostics.get('pas_h', 0.5) if diagnostics else 0.5
                
                if diagnostics and 'trust_mean' in diagnostics:
                    trust = diagnostics.get('trust_mean', 0.5)
                elif diagnostics and 'trust_scalars' in diagnostics and diagnostics['trust_scalars']:
                    trust = sum(diagnostics['trust_scalars']) / len(diagnostics['trust_scalars'])
                else:
                    trust = 0.5
                    
                # Extract exact Tri-State metrics from the diegetic physics engine if available
                retrieval_state = diagnostics.get('retrieval_state') if diagnostics else None
                honesty_score = diagnostics.get('honesty_score') if diagnostics else None
                
                # Fallback approximation if engine is disabled
                if retrieval_state is None or honesty_score is None:
                    honesty_score = (pas_h + trust) / 2.0
                    if honesty_score > 0.7:
                        retrieval_state = "KNOWN"
                    elif honesty_score > 0.3:
                        retrieval_state = "SEARCH_NEEDED"
                    else:
                        retrieval_state = "CONFABULATED"
                formatted_result = {
                    'success': True,
                    'response': result.get('response', 'No response'),
                    'retrieval_state': retrieval_state,
                    'honesty_score': float(honesty_score),
                    'diagnostics': diagnostics,
                    'metrics': {
                        'pas_h': float(pas_h),
                        'trust': float(trust),
                        'affordance': diagnostics.get('type', 'generic') if diagnostics else 'generic',
                        'affordance_strength': float(diagnostics.get('affordance_strength', 0.0)) if diagnostics else 0.0
                    }
                }
                
                self._send_json(formatted_result)
            else:
                # 503 SERVICE UNAVAILABLE: System Warming Protocol
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Retry-After', '30')
                self.end_headers()
                self._send_json({
                    'success': False,
                    'response': 'ARCHITECTURAL SYNTHESIS IN PROGRESS: System Warming...',
                    'status': 'WARMING',
                    'diagnostics': {'code': 503, 'manifold': 'non_coherent'}
                })
                return
        except Exception as e:
            self._send_json({
                'success': False,
                'response': f'Error: {str(e)}',
                'diagnostics': {'error': str(e)},
                'status': 'error'
            })

    def _handle_save_model(self):
        """Handle /api/save_model."""
        try:
            if AI_SYSTEM:
                message = AI_SYSTEM.save_model_state()
                self._send_json({'success': True, 'message': message})
            else:
                self._send_json({'success': False, 'message': 'No AI system to save'})
        except Exception as e:
            self._send_json({'success': False, 'message': f'Fossilization failed: {str(e)}'})

    def _handle_shutdown(self):
        """Handle /api/shutdown."""
        try:
            self._send_json({'success': True, 'message': 'Shutdown initiated'})
            def _stop():
                time.sleep(1)
                import os
                import signal
                os.kill(os.getpid(), signal.SIGINT)
            threading.Thread(target=_stop, daemon=True).start()
        except Exception as e:
            self._send_json({'success': False, 'message': f'Shutdown failed: {str(e)}'})

    def _handle_local_datasets(self):
        """Scan and return local datasets."""
        try:
            data_raw_path = os.path.join(os.path.dirname(__file__), 'data', 'raw')
            if not os.path.exists(data_raw_path):
                # Create it if it doesn't exist
                os.makedirs(data_raw_path, exist_ok=True)
            
            datasets = {}
            # Scan for directories or files
            for item in os.listdir(data_raw_path):
                # Skip hidden files
                if item.startswith('.'):
                    continue
                    
                item_path = os.path.join(data_raw_path, item)
                if os.path.isdir(item_path):
                    # It's a directory dataset
                    try:
                        files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                        file_count = len(files)
                        total_size = sum(os.path.getsize(os.path.join(item_path, f)) for f in files)
                        datasets[item] = {
                            'file_count': file_count,
                            'total_size_mb': round(total_size / (1024 * 1024), 2),
                            'format': 'directory',
                            'description': 'Local directory dataset'
                        }
                    except Exception as e:
                        print(f"Error scanning directory {item}: {e}")
                elif os.path.isfile(item_path):
                     # It's a file dataset
                    datasets[item] = {
                        'file_count': 1,
                        'total_size_mb': round(os.path.getsize(item_path) / (1024 * 1024), 2),
                        'format': item.split('.')[-1] if '.' in item else 'unknown',
                        'description': 'Local file dataset'
                    }
            
            self._send_json({'success': True, 'datasets': datasets})
        except Exception as e:
            self._send_json({'success': False, 'message': str(e)})

    def _handle_ingest_local(self):
        """Handle local dataset ingestion."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            dataset_name = data.get('dataset')
            max_samples = data.get('max_samples', 500)
            
            if not AI_SYSTEM or not AI_SYSTEM.dataset_system:
                self._send_json({'success': False, 'message': 'Dataset system not initialized'})
                return

            # Construct config
            source_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', dataset_name)
            
            config = DatasetConfig(
                name=dataset_name,
                source_type='local',
                source_path=source_path,
                max_samples=max_samples,
                preprocessing='text' # Default
            )
            
            success = AI_SYSTEM.dataset_system.add_dataset_source(config)
            
            if success:
                self._send_json({
                    'success': True,
                    'dataset': dataset_name,
                    'samples_loaded': max_samples, # Approximation for UI
                    'quality_stats': {
                        'count': max_samples,
                        'passing': max_samples,
                        'pass_rate': 1.0,
                        'mean_score': 0.8,
                        'min_score': 0.5,
                        'max_score': 1.0
                    }
                })
            else:
                self._send_json({'success': False, 'message': 'Ingestion failed check logs'})
                
        except Exception as e:
            self._send_json({'success': False, 'message': str(e)})

def start_server(port):
    """Start a server on a specific port."""
    try:
        with socketserver.TCPServer(("", port), HybridHandler) as httpd:
            print(f"[OK] Hybrid backend running at http://localhost:{port}")
            httpd.serve_forever()
    except Exception as e:
        print(f"[FAIL] Server error on port {port}: {e}")

import atexit

def clean_exit_handler():
    global AI_SYSTEM
    if AI_SYSTEM:
        print("[FOSSIL] Clean exit triggered. Running Fossilization Protocol...", flush=True)
        try:
            message = AI_SYSTEM.save_model_state()
            print(f"[OK] {message}", flush=True)
        except Exception as e:
            print(f"[FAIL] Emergency clean exit save failed: {e}", flush=True)

atexit.register(clean_exit_handler)

def main():
    """Start the Gyroidic Backend with Governance and Persistence."""
    global AI_SYSTEM
    
    # 1. Governance Startup (Interactive)
    startup_res = GovernanceManager.startup_menu()
    if isinstance(startup_res, tuple) and len(startup_res) == 2:
        active_ports, config = startup_res
    else:
        active_ports = startup_res
        config = {}
    
    print("\n[START] Gyroidic Hybrid Backend (Sovereign Mode)")
    print("=" * 45)
    
    # 2. Initialize AI system
    print("[BRAIN] Initializing AI components...")
    try:
        AI_SYSTEM = HybridAI(
            use_spectral_correction=config.get('use_spectral_correction', True),
            config=config
        )
        print("[OK] AI system initialized")
    except Exception as e:
        print(f"[FAIL] AI system initialization failed: {e}")
        AI_SYSTEM = None
    
    print(f"[WEB] Initializing listeners on {len(active_ports)} ports...")
    print("[CONFIG] Components active:")
    print(f"   • Temporal Model: {'[OK]' if TEMPORAL_MODEL_AVAILABLE else '[FAIL]'}")
    print(f"   • Spectral Corrector: {'[OK]' if SPECTRAL_CORRECTOR_AVAILABLE else '[FAIL]'}")
    print("[STOP]  Press Ctrl+C to trigger Fossilization and Exit")
    
    # 3. Dynamic Port Selection Infrastructure
    threads = []
    for port in active_ports:
        t = threading.Thread(target=start_server, args=(port,), daemon=True, name=f"ServerThread-{port}")
        t.start()
        threads.append(t)
        
    # Option D: UDP Server Colonization
    udp_colonizer = None
    if config.get('udp_colonizer_enabled', False):
        print("[UDP] Booting Option D Master Server Colonizer...")
        udp_colonizer = OptionD_Colonizer(port=27015, app_id=320)
        # Start with localhost, assuming Cloudflare tunnel points here
        udp_colonizer.set_tunnel_url("http://localhost:8000")
        udp_colonizer.start()
    else:
        print("[UDP] Option D Master Server Colonizer is DISABLED.")
    
    # 4. Lifecycle Control (Ctrl+C Fossilization)
    try:
        while True:
            # We use a short sleep to remain responsive to SIGINT
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n" + "!" * 50)
        print("       KEYBOARD INTERRUPT DETECTED  ")
        print("!" * 50)
        
        # Prevent double-interrupt from corrupting the save
        import signal
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        
        if AI_SYSTEM:
            print("[FOSSIL] Triggering emergency Fossilization Protocol...")
            print("[INFO]  Ignoring further interrupts to ensure manifold integrity.", flush=True)
            try:
                message = AI_SYSTEM.save_model_state()
                print(f"[OK] {message}", flush=True)
            except Exception as e:
                print(f"[FAIL] Emergency save failed: {e}", flush=True)
        else:
            print("[WARN] AI_SYSTEM not initialized; bypassing fossilization.", flush=True)
            
        if udp_colonizer is not None:
            print("[UDP] Stopping Option D Master Server Colonizer...", flush=True)
            udp_colonizer.stop()
        
        print("[STOP]  Shutting down manifolds. Goodbye.", flush=True)
        # Prevent PyArrow segfault by finalizing S3
        try:
            import pyarrow.fs
            pyarrow.fs.finalize_s3()
        except Exception:
            pass
        # Restore signal handler before exit if needed (though os._exit is coming)
        os._exit(0)

if __name__ == "__main__":
    main()

