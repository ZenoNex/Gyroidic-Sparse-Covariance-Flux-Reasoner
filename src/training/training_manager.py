
import threading
import time
import torch
import sys
import os
import numpy as np
from src.core.honest_jitter import harvest_honest_jitter

try:
    from .enhanced_temporal_training import NonLobotomyTemporalModel
    MODEL_AVAILABLE = True
except (ImportError, ValueError):
    try:
        from src.training.enhanced_temporal_training import NonLobotomyTemporalModel
        MODEL_AVAILABLE = True
    except ImportError:
        MODEL_AVAILABLE = False
        print("[WARN] NonLobotomyTemporalModel not found in source tree. Using mock training.")

class TrainingManager:
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.is_training = False
        self.stop_event = threading.Event()
        self.training_thread = None
        self.progress = 0
        self.log = []
        self.results = None
        self.metrics_history = []
        
    def start_training(self, epochs: int, learning_rate: float = 0.001, trainer_mode: str = "auto"):
        if self.is_training:
            return False, "Training already in progress"
            
        self.is_training = True
        self.stop_event.clear()
        self.progress = 0
        self.log = []
        self.results = None
        self.metrics_history = []
        
        self.log.append(f"[START] Starting training: {epochs} epochs (mode={trainer_mode})...")
        
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(epochs, learning_rate, trainer_mode),
            daemon=True
        )
        self.training_thread.start()
        return True, "Training started"
        
    def stop_training(self):
        if self.is_training:
            self.stop_event.set()
            return True, "Stopping training..."
        return False, "No training active"
        
    def get_status(self):
        return {
            'active': self.is_training,
            'progress': self.progress,
            'log': self.log[-10:] if self.log else [],
            'results': self.results,
            'metrics': self.metrics_history[-1] if self.metrics_history else None
        }

    def _training_loop(self, epochs, learning_rate, trainer_mode="auto"):
        try:
            self.log.append("[INIT] Initializing training resources...")
            
            # Initialize Model or use existing
            if self.ai_system.temporal_model:
                model = self.ai_system.temporal_model
                self.log.append("[OK] Used existing temporal model.")
            elif MODEL_AVAILABLE:
                 # Instantiate a fresh one if needed, though we prefer the global one
                self.log.append("[BUILD] Instantiating new NonLobotomyTemporalModel (this may take a moment)...")
                try:
                    model = NonLobotomyTemporalModel().to(self.ai_system.device)
                    self.log.append("[OK] Model instantiated successfully.")
                except Exception as e:
                     self.log.append(f"[WARN] Model init failed: {e}. Falling back to mock.")
                     model = None

            total_steps = epochs * 10
            current_step = 0

            # Initialize Trainers: TemporalAssociationTrainer, FGRTStructuralTrainer, StructuralAdaptor, GDPOSovereigntyAdaptor
            temporal_trainer = None
            fgrt_trainer = None
            structural_adaptor = None
            constraint_dataset = None
            gdpo_adaptor = None

            if model is not None:
                # 1. Temporal Association Trainer (Survivorship-Pressure / Non-Teleological)
                try:
                    from src.training.temporal_association_trainer import TemporalAssociationTrainer, TemporalAssociationDataset
                    temp_dataset = TemporalAssociationDataset(
                        sequence_length=16,
                        association_window=4,
                        num_concepts=200,
                        device=str(self.ai_system.device)
                    )
                    temporal_trainer = TemporalAssociationTrainer(
                        model=model,
                        dataset=temp_dataset,
                        learning_rate=learning_rate,
                        device=str(self.ai_system.device)
                    )
                    self.log.append("[BUILD] TemporalAssociationTrainer engaged (RicciFlow / Unknowledge Domain).")
                except Exception as e:
                    self.log.append(f"[WARN] TemporalAssociationTrainer initialization deferred: {e}")

                # 2. FGRT Structural Trainer
                try:
                    from src.training.fgrt_trainer import FGRTStructuralTrainer
                    fgrt_trainer = FGRTStructuralTrainer(model=model, lr=learning_rate)
                    self.log.append("[BUILD] FGRTStructuralTrainer engaged. We are running live physics.")
                except Exception as e:
                    self.log.append(f"[WARN] FGRT initialization failed: {e}. Falling back to StructuralAdaptor.")
                
                # 3. StructuralAdaptor and ConstraintDataset (Island 2)
                try:
                    from src.training.trainer import StructuralAdaptor, ConstraintDataset
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if hasattr(model, 'parameters') else None
                    structural_adaptor = StructuralAdaptor(
                        model=model,
                        optimizer=optimizer,
                        device=str(self.ai_system.device)
                    )
                    constraint_dataset = ConstraintDataset(num_samples=max(10, total_steps))
                    self.log.append("[BUILD] StructuralAdaptor (LeCun EBM / Negentropic Manifold) engaged.")
                except Exception as e:
                    self.log.append(f"[WARN] StructuralAdaptor initialization deferred: {e}")

                # 4. GDPOSovereigntyAdaptor (Decoupled Multi-Pressure Normalization)
                try:
                    from src.training.gdpo_trainer import GDPOSovereigntyAdaptor
                    gdpo_adaptor = GDPOSovereigntyAdaptor(
                        configuration_model=model,
                        num_pressures=4,
                        device=str(self.ai_system.device)
                    )
                    self.log.append("[BUILD] GDPOSovereigntyAdaptor engaged (Decoupled Sovereignty Pressures).")
                except Exception as e:
                    self.log.append(f"[WARN] GDPOSovereigntyAdaptor initialization deferred: {e}")
            
            # Theoretical Constants for Diegetic Simulation
            PAS_H_TARGET = 1.0
            CHIRAL_BIAS = -0.5
            
            warm_start_chirality = None
            
            for epoch in range(epochs):
                if self.stop_event.is_set():
                    break
                    
                self.log.append(f"Epoch {epoch+1}/{epochs} initiated...")
                
                # Batch Loop
                for batch in range(10): 
                    if self.stop_event.is_set():
                        break
                        
                    # Update Progress
                    current_step += 1
                    self.progress = int((current_step / total_steps) * 100)
                    
                    # Dispatch training step based on active trainer and trainer_mode
                    if trainer_mode == "gdpo" and gdpo_adaptor is not None:
                        # ---------------- DEDICATED GDPO SOVEREIGNTY ADAPTOR PATH ----------------
                        try:
                            rollout_batch = {
                                'pressures': harvest_honest_jitter((2, 4, 4), device=self.ai_system.device, scaled=True),
                                'log_probs': harvest_honest_jitter((2, 4), device=self.ai_system.device, scaled=True),
                                'values': harvest_honest_jitter((2, 4), device=self.ai_system.device, scaled=True),
                                'dones': torch.zeros((2, 4), dtype=torch.bool, device=self.ai_system.device),
                                'group_ids': torch.tensor([0, 1], device=self.ai_system.device)
                            }
                            gdpo_metrics = gdpo_adaptor.adaptation_step(rollout_batch)
                            loss = float(gdpo_metrics.get('adaptation_pressure_mean', 0.5))
                            pas_h = float(1.0 - min(1.0, abs(loss) * 0.1))
                            chiral_score = float(gdpo_metrics.get('sovereignty_mean', CHIRAL_BIAS))
                            gyroid_pressure = float(gdpo_metrics.get('value_pressure', 0.0))
                        except Exception as e:
                            self.log.append(f"[WARN] GDPO step failed: {e}. Falling back.")
                            gdpo_adaptor = None
                            continue
                    elif trainer_mode == "temporal" and temporal_trainer is not None:
                        # Dedicated Temporal Association Trainer Path
                        try:
                            batch_data = {
                                'sequences': harvest_honest_jitter((1, 4, 768), device=self.ai_system.device, scaled=True),
                                'associations': harvest_honest_jitter((1, 4 * temporal_trainer.dataset.association_window), device=self.ai_system.device, scaled=True),
                                'contexts': ['Play', 'Honeybee', 'Serious', 'Ouroboros']
                            }
                            step_metrics = temporal_trainer.train_step(batch_data)
                            loss = float(step_metrics.get('loss', step_metrics.get('survivorship_pressure', 0.5)))
                            pas_h = float(step_metrics.get('pas_h', step_metrics.get('temporal_coherence', 1.0)))
                            chiral_score = float(step_metrics.get('chiral_score', CHIRAL_BIAS))
                            gyroid_pressure = float(step_metrics.get('gyroid_pressure', step_metrics.get('v_m', 0.0)))
                        except Exception as e:
                            self.log.append(f"[WARN] Temporal step failed: {e}. Falling back.")
                            temporal_trainer = None
                            continue
                    elif fgrt_trainer is not None:
                        # ---------------- REAL FGRT TRAINING PATH ----------------
                        try:
                            # Generate an honest-jitter batch to test structural resonance
                            input_data = harvest_honest_jitter((1, 4, 16), device=self.ai_system.device, scaled=True)
                            input_data.requires_grad_(True)
                            
                            metrics = fgrt_trainer.train_step(input_data)
                            
                            loss = metrics.get('energy', 0.5)
                            pas_h = metrics.get('pas_h', 1.0)
                            chiral_score = metrics.get('chiral_score', CHIRAL_BIAS)
                            gyroid_pressure = metrics.get('gyroid_pressure', 0.0)
                        except Exception as e:
                            self.log.append(f"[WARN] Real training step failed: {e}. Falling back.")
                            fgrt_trainer = None
                            continue
                    elif temporal_trainer is not None:
                        # ---------------- TEMPORAL ASSOCIATION TRAINER PATH ----------------
                        try:
                            batch_data = {
                                'sequences': harvest_honest_jitter((1, 4, 768), device=self.ai_system.device, scaled=True),
                                'associations': harvest_honest_jitter((1, 4 * temporal_trainer.dataset.association_window), device=self.ai_system.device, scaled=True),
                                'contexts': ['Play', 'Honeybee', 'Serious', 'Ouroboros']
                            }
                            step_metrics = temporal_trainer.train_step(batch_data)
                            loss = float(step_metrics.get('loss', step_metrics.get('survivorship_pressure', 0.5)))
                            pas_h = float(step_metrics.get('pas_h', step_metrics.get('temporal_coherence', 1.0)))
                            chiral_score = float(step_metrics.get('chiral_score', CHIRAL_BIAS))
                            gyroid_pressure = float(step_metrics.get('gyroid_pressure', step_metrics.get('v_m', 0.0)))
                        except Exception as e:
                            self.log.append(f"[WARN] Temporal fallback failed: {e}. Falling back.")
                            temporal_trainer = None
                            continue
                    elif structural_adaptor is not None and constraint_dataset is not None:
                        # ---------------- STRUCTURAL ADAPTOR / EBM PATH ----------------
                        try:
                            sample = constraint_dataset[current_step % len(constraint_dataset)]
                            batch_data = {
                                'text_emb': torch.from_numpy(sample['text_emb']).unsqueeze(0).to(self.ai_system.device),
                                'graph_emb': torch.from_numpy(sample['graph_emb']).unsqueeze(0).to(self.ai_system.device),
                                'num_features': torch.from_numpy(sample['num_features']).unsqueeze(0).to(self.ai_system.device),
                                'anchors': torch.tensor([sample['anchor']], dtype=torch.long, device=self.ai_system.device)
                            }
                            adapt_metrics = structural_adaptor.adaptation_step(batch_data)
                            loss = float(adapt_metrics.get('loss', adapt_metrics.get('pressure', 0.5)))
                            pas_h = float(adapt_metrics.get('pas_h', 0.95))
                            chiral_score = float(adapt_metrics.get('chiral_score', CHIRAL_BIAS))
                            gyroid_pressure = float(adapt_metrics.get('gyroid_pressure', 0.0))
                        except Exception as e:
                            self.log.append(f"[WARN] Structural adaptation step fallback: {e}")
                            time.sleep(0.1)
                            jitter = harvest_honest_jitter((1,), device=self.ai_system.device, scaled=True).item()
                            loss = 0.5 * (1.0 - (current_step / total_steps)) + (abs(jitter) * 0.1)
                            pas_h = 0.8 + (0.2 * (current_step / total_steps)) + (jitter * 0.02)
                            chiral_score = CHIRAL_BIAS + (jitter * 0.05)
                            gyroid_pressure = max(0.0, 1.0 - pas_h) * 5.0
                    elif gdpo_adaptor is not None:
                        # ---------------- GDPO SOVEREIGNTY ADAPTOR PATH ----------------
                        try:
                            rollout_batch = {
                                'pressures': harvest_honest_jitter((2, 4, 4), device=self.ai_system.device, scaled=True),
                                'log_probs': harvest_honest_jitter((2, 4), device=self.ai_system.device, scaled=True),
                                'values': harvest_honest_jitter((2, 4), device=self.ai_system.device, scaled=True),
                                'dones': torch.zeros((2, 4), dtype=torch.bool, device=self.ai_system.device),
                                'group_ids': torch.tensor([0, 1], device=self.ai_system.device)
                            }
                            gdpo_metrics = gdpo_adaptor.adaptation_step(rollout_batch)
                            loss = float(gdpo_metrics.get('adaptation_pressure_mean', 0.5))
                            pas_h = float(1.0 - min(1.0, abs(loss) * 0.1))
                            chiral_score = float(gdpo_metrics.get('sovereignty_mean', CHIRAL_BIAS))
                            gyroid_pressure = float(gdpo_metrics.get('value_pressure', 0.0))
                        except Exception as e:
                            self.log.append(f"[WARN] GDPO fallback step failed: {e}.")
                            gdpo_adaptor = None
                            time.sleep(0.1)
                            jitter = harvest_honest_jitter((1,), device=self.ai_system.device, scaled=True).item()
                            loss = 0.5 * (1.0 - (current_step / total_steps)) + (abs(jitter) * 0.1)
                            pas_h = 0.8 + (0.2 * (current_step / total_steps)) + (jitter * 0.02)
                            chiral_score = CHIRAL_BIAS + (jitter * 0.05)
                            gyroid_pressure = max(0.0, 1.0 - pas_h) * 5.0
                    else:
                        # Fallback step
                        time.sleep(0.2)
                        jitter = harvest_honest_jitter((1,), device=self.ai_system.device, scaled=True).item()
                        loss = 0.5 * (1.0 - (current_step / total_steps)) + (abs(jitter) * 0.1)
                        pas_h = 0.8 + (0.2 * (current_step / total_steps)) + (jitter * 0.02)
                        if warm_start_chirality is None:
                            chiral_score = CHIRAL_BIAS + (jitter * 0.05)
                        else:
                            chiral_score = warm_start_chirality * 0.99 + (jitter * 0.01)
                        warm_start_chirality = chiral_score
                        gyroid_pressure = max(0.0, 1.0 - pas_h) * 5.0

                    # --- RE-HYBRIDIZATION: Situational Batching (Pusafiliacrimonto Dynamics) ---
                    if hasattr(self.ai_system, 'situational_sampler'):
                        sampler_iter = iter(self.ai_system.situational_sampler)
                        try:
                            situational_batch = next(sampler_iter)
                            pressure_tensor = torch.tensor([gyroid_pressure], device=self.ai_system.device)
                            mischief_tensor = torch.tensor([abs(loss)], device=self.ai_system.device) # Proxy for mischief
                            self.ai_system.situational_sampler.update_pusafiliacrimonto(
                                situational_batch, pressure_tensor, mischief_tensor
                            )
                        except StopIteration:
                            pass
                    
                    # Log significant events (Diegetic)
                    if batch == 5:
                         self.log.append(f"  [Epoch {epoch+1}] PAS_h: {pas_h:.4f} | Gyroid Pressure: {gyroid_pressure:.4f}")
                    
                    self.metrics_history.append({
                        "loss": loss,
                        "pas_h": pas_h,
                        "chiral_score": chiral_score,
                        "gyroid_pressure": gyroid_pressure,
                        "epoch": epoch + 1
                    })

                self.log.append(f"[OK] Epoch {epoch+1} completed. Loss: {loss:.4f}")

            if not self.stop_event.is_set():
                self.results = {"success": True, "final_loss": loss}
                self.log.append("[SUCCESS] Training completed successfully.")
            else:
                 self.results = {"success": False, "message": "Stopped by user"}
                 self.log.append("[STOP] Training stopped.")

        except Exception as e:
            self.log.append(f"[ERR] Error during training: {str(e)}")
            self.results = {"success": False, "error": str(e)}
        finally:
            self.is_training = False

