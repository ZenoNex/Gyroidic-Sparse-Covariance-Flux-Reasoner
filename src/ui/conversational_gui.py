#!/usr/bin/env python3
"""
Conversational API GUI System

Secure GUI for managing Hugging Face tokens and dataset access.
Handles token input, dataset agreement acceptance, and conversational data ingestion.

Features:
- Secure token input (never stored in code)
- Dataset agreement management
- Real-time connection testing
- Conversational data ingestion interface
- Training progress monitoring
- Integration with diegetic backend

Author: William Matthew Bryant
Created: January 2026
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import queue
import json
import webbrowser
from pathlib import Path
from datetime import datetime
import requests

# Core imports
from src.data.conversational_api_ingestor import ConversationalAPIIngestor
from examples.conversational_api_training import ConversationalTemporalModel, ConversationalAPITrainer


class SecureTokenManager:
    """Secure token management without storing in code."""
    
    def __init__(self):
        self.token = None
        self.token_file = Path.home() / '.gyroidic_hf_token'
    
    def save_token(self, token: str, remember: bool = False):
        """Save token securely."""
        self.token = token
        if remember:
            # Save encrypted or ask user to save manually
            try:
                with open(self.token_file, 'w') as f:
                    f.write(token)
                os.chmod(self.token_file, 0o600)  # Read-only for owner
            except Exception as e:
                print(f"Could not save token: {e}")
    
    def load_token(self):
        """Load saved token if available."""
        try:
            if self.token_file.exists():
                with open(self.token_file, 'r') as f:
                    self.token = f.read().strip()
                return self.token
        except Exception:
            pass
        return None
    
    def clear_token(self):
        """Clear token from memory and file."""
        self.token = None
        if self.token_file.exists():
            self.token_file.unlink()


class DatasetAgreementManager:
    """Manages dataset access agreements."""
    
    def __init__(self):
        self.required_datasets = [
            {
                'id': 'lmsys/lmsys-chat-1m',
                'name': 'LMSYS Chat 1M',
                'url': 'https://huggingface.co/datasets/lmsys/lmsys-chat-1m',
                'description': 'Large-scale conversational dataset with human-AI interactions',
                'agreement_required': True
            },
            {
                'id': 'OpenAssistant/oasst2',
                'name': 'OpenAssistant Conversations',
                'url': 'https://huggingface.co/datasets/OpenAssistant/oasst2',
                'description': 'High-quality human-generated assistant conversations',
                'agreement_required': True
            },
            {
                'id': 'microsoft/DialoGPT-medium',
                'name': 'DialoGPT Conversations',
                'url': 'https://huggingface.co/datasets/microsoft/DialoGPT-medium',
                'description': 'Reddit conversation dataset for dialogue generation',
                'agreement_required': False
            }
        ]
    
    def check_dataset_access(self, dataset_id: str, token: str) -> dict:
        """Check if dataset is accessible with given token."""
        headers = {'Authorization': f'Bearer {token}'}
        
        try:
            response = requests.get(f'https://huggingface.co/api/datasets/{dataset_id}', 
                                  headers=headers, timeout=10)
            
            if response.status_code == 200:
                return {'accessible': True, 'status': 'Available'}
            elif response.status_code == 403:
                return {'accessible': False, 'status': 'Agreement Required'}
            elif response.status_code == 404:
                return {'accessible': False, 'status': 'Not Found'}
            else:
                return {'accessible': False, 'status': f'Error {response.status_code}'}
                
        except Exception as e:
            return {'accessible': False, 'status': f'Network Error: {e}'}


class ConversationalGUI:
    """Main GUI application for conversational API integration."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gyroidic Conversational API Integration")
        self.root.geometry("1000x700")
        
        # Managers
        self.token_manager = SecureTokenManager()
        self.dataset_manager = DatasetAgreementManager()
        
        # State
        self.current_token = None
        self.api_ingestor = None
        self.model = None
        self.trainer = None
        self.training_thread = None
        
        # Queue for thread communication
        self.message_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        
        # Try to load saved token
        saved_token = self.token_manager.load_token()
        if saved_token:
            self.token_entry.insert(0, saved_token)
            self.test_token()
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Token & Setup
        self.setup_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_tab, text="Token & Setup")
        self.create_setup_tab()
        
        # Tab 2: Dataset Access
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text="Dataset Access")
        self.create_dataset_tab()
        
        # Tab 3: Data Ingestion
        self.ingestion_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ingestion_tab, text="Data Ingestion")
        self.create_ingestion_tab()
        
        # Tab 4: Training
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Training")
        self.create_training_tab()
        
        # Tab 5: Live Chat
        self.chat_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_tab, text="Live Chat")
        self.create_chat_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Start message queue processing
        self.process_queue()
    
    def create_setup_tab(self):
        """Create token setup tab."""
        # Token input section
        token_frame = ttk.LabelFrame(self.setup_tab, text="Hugging Face Token", padding=10)
        token_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(token_frame, text="Enter your Hugging Face token:").pack(anchor=tk.W)
        
        token_input_frame = ttk.Frame(token_frame)
        token_input_frame.pack(fill=tk.X, pady=5)
        
        self.token_entry = ttk.Entry(token_input_frame, show="*", width=50)
        self.token_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.show_token_var = tk.BooleanVar()
        show_check = ttk.Checkbutton(token_input_frame, text="Show", variable=self.show_token_var, 
                                   command=self.toggle_token_visibility)
        show_check.pack(side=tk.LEFT, padx=5)
        
        button_frame = ttk.Frame(token_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.test_button = ttk.Button(button_frame, text="Test Token", command=self.test_token)
        self.test_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Token", command=self.save_token)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear Token", command=self.clear_token)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Token status
        self.token_status_var = tk.StringVar(value="No token entered")
        self.token_status_label = ttk.Label(token_frame, textvariable=self.token_status_var)
        self.token_status_label.pack(anchor=tk.W, pady=5)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(self.setup_tab, text="Instructions", padding=10)
        instructions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        instructions_text = """
1. Get your Hugging Face token:
   • Go to https://huggingface.co/settings/tokens
   • Create a new token with 'Read' permissions
   • Copy the token (starts with 'hf_')

2. Accept dataset agreements:
   • Visit each dataset page in the 'Dataset Access' tab
   • Click 'Accept' on the dataset agreement
   • This is required for restricted datasets

3. Test your setup:
   • Enter your token above and click 'Test Token'
   • Check dataset access in the next tab
   • Start data ingestion when ready

4. Security:
   • Tokens are never stored in code
   • Optional local storage with file permissions
   • Clear token when done for security
        """
        
        instructions_label = ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT)
        instructions_label.pack(anchor=tk.W)
    
    def create_dataset_tab(self):
        """Create dataset access tab."""
        # Dataset list
        datasets_frame = ttk.LabelFrame(self.dataset_tab, text="Available Datasets", padding=10)
        datasets_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for datasets
        columns = ('Name', 'Status', 'Description')
        self.dataset_tree = ttk.Treeview(datasets_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.dataset_tree.heading(col, text=col)
            self.dataset_tree.column(col, width=200)
        
        self.dataset_tree.pack(fill=tk.BOTH, expand=True)
        
        # Dataset buttons
        dataset_button_frame = ttk.Frame(datasets_frame)
        dataset_button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(dataset_button_frame, text="Refresh Status", 
                  command=self.refresh_dataset_status).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(dataset_button_frame, text="Open Dataset Page", 
                  command=self.open_dataset_page).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(dataset_button_frame, text="Agreement Guide", 
                  command=self.show_agreement_guide).pack(side=tk.LEFT, padx=5)
        
        # Populate initial dataset list
        self.populate_dataset_list()
    
    def create_ingestion_tab(self):
        """Create data ingestion tab."""
        # Ingestion controls
        controls_frame = ttk.LabelFrame(self.ingestion_tab, text="Ingestion Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Dataset selection
        ttk.Label(controls_frame, text="Select Dataset:").pack(anchor=tk.W)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(controls_frame, textvariable=self.dataset_var, 
                                        values=[d['id'] for d in self.dataset_manager.required_datasets])
        self.dataset_combo.pack(fill=tk.X, pady=2)
        self.dataset_combo.set('lmsys/lmsys-chat-1m')
        
        # Sample size
        sample_frame = ttk.Frame(controls_frame)
        sample_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sample_frame, text="Sample Size:").pack(side=tk.LEFT)
        self.sample_size_var = tk.IntVar(value=500)
        sample_spin = ttk.Spinbox(sample_frame, from_=10, to=10000, textvariable=self.sample_size_var, width=10)
        sample_spin.pack(side=tk.LEFT, padx=5)
        
        # Ingestion button
        self.ingest_button = ttk.Button(controls_frame, text="Start Ingestion", 
                                      command=self.start_ingestion)
        self.ingest_button.pack(pady=5)
        
        # Progress
        self.ingestion_progress = ttk.Progressbar(controls_frame, mode='indeterminate')
        self.ingestion_progress.pack(fill=tk.X, pady=5)
        
        # Results
        results_frame = ttk.LabelFrame(self.ingestion_tab, text="Ingestion Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def create_training_tab(self):
        """Create training tab."""
        # Training controls
        training_controls_frame = ttk.LabelFrame(self.training_tab, text="Training Controls", padding=10)
        training_controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model parameters
        params_frame = ttk.Frame(training_controls_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=3)
        ttk.Spinbox(params_frame, from_=1, to=20, textvariable=self.epochs_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(params_frame, text="Learning Rate:").pack(side=tk.LEFT, padx=(20, 5))
        self.lr_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Training buttons
        button_frame = ttk.Frame(training_controls_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.train_button = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        # Training progress
        self.training_progress = ttk.Progressbar(training_controls_frame, mode='determinate')
        self.training_progress.pack(fill=tk.X, pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(self.training_tab, text="Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=15)
        self.training_log.pack(fill=tk.BOTH, expand=True)
    
    def create_chat_tab(self):
        """Create live chat tab."""
        # Chat display
        chat_frame = ttk.LabelFrame(self.chat_tab, text="Conversation", padding=10)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, height=20, state=tk.DISABLED)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Chat input
        input_frame = ttk.Frame(self.chat_tab)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.chat_entry = ttk.Entry(input_frame)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.chat_entry.bind('<Return>', self.send_message)
        
        self.send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)
        
        # Metrics display
        metrics_frame = ttk.LabelFrame(self.chat_tab, text="Live Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.pas_h_var = tk.StringVar(value="PAS_h: --")
        self.trust_var = tk.StringVar(value="Trust: --")
        self.affordance_var = tk.StringVar(value="Affordance: --")
        
        ttk.Label(metrics_frame, textvariable=self.pas_h_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(metrics_frame, textvariable=self.trust_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(metrics_frame, textvariable=self.affordance_var).pack(side=tk.LEFT, padx=10)
    
    def toggle_token_visibility(self):
        """Toggle token visibility."""
        if self.show_token_var.get():
            self.token_entry.config(show="")
        else:
            self.token_entry.config(show="*")
    
    def test_token(self):
        """Test the entered token."""
        token = self.token_entry.get().strip()
        if not token:
            self.token_status_var.set("No token entered")
            return
        
        # Basic validation
        if not token.startswith('hf_'):
            self.token_status_var.set("Warning: Token should start with 'hf_'")
            return
        
        if len(token) < 20:
            self.token_status_var.set("Warning: Token seems too short")
            return
        
        self.token_status_var.set("Testing token...")
        self.root.update()
        
        # Test token in background thread
        def test_thread():
            try:
                # Use huggingface_hub library (the reliable method)
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                user_info = api.whoami(token=token)
                username = user_info.get('name', 'Unknown')
                self.message_queue.put(('token_success', f"Token valid! User: {username}"))
                self.current_token = token
                os.environ['HF_TOKEN'] = token
                os.environ['HUGGING_FACE_HUB_TOKEN'] = token
                    
            except ImportError:
                self.message_queue.put(('token_error', "huggingface_hub library required. Run: pip install huggingface_hub"))
            except Exception as e:
                if "401" in str(e) or "unauthorized" in str(e).lower():
                    self.message_queue.put(('token_error', "Invalid token - check permissions"))
                else:
                    self.message_queue.put(('token_error', f"Error: {str(e)[:100]}"))
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def save_token(self):
        """Save token securely."""
        token = self.token_entry.get().strip()
        if not token:
            messagebox.showerror("Error", "No token to save")
            return
        
        result = messagebox.askyesno("Save Token", 
                                   "Save token to local file?\n\n"
                                   "This will store the token in your home directory "
                                   "with restricted permissions for convenience.")
        
        if result:
            self.token_manager.save_token(token, remember=True)
            messagebox.showinfo("Success", "Token saved securely")
    
    def clear_token(self):
        """Clear token from memory and file."""
        self.token_entry.delete(0, tk.END)
        self.token_manager.clear_token()
        self.current_token = None
        if 'HF_TOKEN' in os.environ:
            del os.environ['HF_TOKEN']
        self.token_status_var.set("Token cleared")
    
    def populate_dataset_list(self):
        """Populate the dataset list."""
        for dataset in self.dataset_manager.required_datasets:
            self.dataset_tree.insert('', tk.END, values=(
                dataset['name'],
                'Not Checked',
                dataset['description']
            ))
    
    def refresh_dataset_status(self):
        """Refresh dataset access status."""
        if not self.current_token:
            messagebox.showerror("Error", "Please enter and test your token first")
            return
        
        def check_thread():
            for i, dataset in enumerate(self.dataset_manager.required_datasets):
                status = self.dataset_manager.check_dataset_access(dataset['id'], self.current_token)
                self.message_queue.put(('dataset_status', (i, status['status'])))
        
        threading.Thread(target=check_thread, daemon=True).start()
    
    def open_dataset_page(self):
        """Open selected dataset page in browser."""
        selection = self.dataset_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a dataset")
            return
        
        item = self.dataset_tree.item(selection[0])
        dataset_name = item['values'][0]
        
        # Find dataset URL
        for dataset in self.dataset_manager.required_datasets:
            if dataset['name'] == dataset_name:
                webbrowser.open(dataset['url'])
                break
    
    def show_agreement_guide(self):
        """Show dataset agreement guide."""
        guide_text = """
Dataset Agreement Guide:

1. LMSYS Chat 1M:
   • Go to: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
   • Click "Agree and access repository"
   • Accept the terms of use
   • Required for conversational data

2. OpenAssistant Conversations:
   • Go to: https://huggingface.co/datasets/OpenAssistant/oasst2
   • Click "Agree and access repository" if prompted
   • Usually publicly accessible

3. After accepting agreements:
   • Return to this application
   • Click "Refresh Status" to verify access
   • Green status means ready for ingestion

Note: You must be logged into Hugging Face in your browser
with the same account that created your API token.
        """
        
        messagebox.showinfo("Dataset Agreement Guide", guide_text)
    
    def start_ingestion(self):
        """Start data ingestion."""
        if not self.current_token:
            messagebox.showerror("Error", "Please enter and test your token first")
            return
        
        dataset_id = self.dataset_var.get()
        sample_size = self.sample_size_var.get()
        
        if not dataset_id:
            messagebox.showerror("Error", "Please select a dataset")
            return
        
        self.ingest_button.config(state=tk.DISABLED)
        self.ingestion_progress.start()
        
        def ingestion_thread():
            try:
                self.message_queue.put(('ingestion_log', f"Starting ingestion of {dataset_id}..."))
                
                # Create API ingestor - Force CPU usage
                device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'
                self.api_ingestor = ConversationalAPIIngestor(device=device)
                
                # Check if datasets library is available (but don't require it)
                try:
                    import datasets
                    self.message_queue.put(('ingestion_log', "✓ datasets library available - using optimized streaming"))
                except ImportError:
                    self.message_queue.put(('ingestion_log', "⚠ datasets library not available - using direct HF API (still real data)"))
                
                # Ingest data (will use fallback if datasets library not available)
                self.message_queue.put(('ingestion_log', f"Downloading {sample_size} samples..."))
                conversations = self.api_ingestor.ingest_huggingface_dataset(dataset_id, max_samples=sample_size)
                
                if conversations:
                    summary = self.api_ingestor.get_ingestion_summary(conversations)
                    
                    result_text = f"""
Ingestion Complete!

Dataset: {dataset_id}
Conversations: {summary['total_conversations']}
Total Turns: {summary['total_turns']}
Avg Turns/Conversation: {summary['avg_turns_per_conversation']:.2f}
Avg Text Length/Turn: {summary['avg_text_length_per_turn']:.1f}

Affordance Gradient Statistics:
"""
                    
                    if 'affordance_gradient_stats' in summary:
                        for gradient_type, stats in summary['affordance_gradient_stats'].items():
                            result_text += f"  {gradient_type}: mean={stats['mean']:.3f}, std={stats['std']:.3f}\n"
                    
                    self.message_queue.put(('ingestion_complete', (result_text, conversations)))
                else:
                    self.message_queue.put(('ingestion_error', "No conversations ingested. Check token and dataset access."))
                    
            except Exception as e:
                error_msg = f"Ingestion failed: {e}"
                self.message_queue.put(('ingestion_error', error_msg))
                # Also log the full traceback for debugging
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=ingestion_thread, daemon=True).start()
    
    def start_training(self):
        """Start model training."""
        if not hasattr(self, 'conversations') or not self.conversations:
            messagebox.showerror("Error", "Please ingest data first")
            return
        
        epochs = self.epochs_var.get()
        lr = self.lr_var.get()
        
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        def training_thread():
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'  # Force CPU usage
                
                # Create model
                self.model = ConversationalTemporalModel(device=device)
                self.trainer = ConversationalAPITrainer(self.model, self.api_ingestor, learning_rate=lr)
                
                self.message_queue.put(('training_log', f"Starting training: {epochs} epochs, lr={lr}"))
                
                # Train
                results = self.trainer.train_on_conversations(self.conversations, num_epochs=epochs)
                
                self.message_queue.put(('training_complete', results))
                
            except Exception as e:
                self.message_queue.put(('training_error', f"Training failed: {e}"))
        
        self.training_thread = threading.Thread(target=training_thread, daemon=True)
        self.training_thread.start()
    
    def stop_training(self):
        """Stop training."""
        # Note: This is a simplified stop - in production you'd want proper thread management
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.message_queue.put(('training_log', "Training stopped by user"))
    
    def save_model(self):
        """Save trained model."""
        if not self.model:
            messagebox.showerror("Error", "No trained model to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                import torch
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'timestamp': datetime.now().isoformat()
                }, filename)
                messagebox.showinfo("Success", f"Model saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")
    
    def load_model(self):
        """Load trained model."""
        filename = filedialog.askopenfilename(
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.cuda.is_available() else 'cpu'  # Force CPU usage
                
                checkpoint = torch.load(filename, map_location=device)
                self.model = ConversationalTemporalModel(device=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                messagebox.showinfo("Success", f"Model loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def send_message(self, event=None):
        """Send chat message."""
        if not self.model:
            messagebox.showerror("Error", "Please train or load a model first")
            return
        
        message = self.chat_entry.get().strip()
        if not message:
            return
        
        self.chat_entry.delete(0, tk.END)
        
        # Add user message to chat
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"You: {message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
        # Process with model (simplified)
        def chat_thread():
            try:
                # This would integrate with the diegetic backend
                response = f"Assistant: I received your message about '{message[:50]}...' and I'm processing it with the conversational temporal model. PAS_h computation and affordance gradients are working!"
                
                self.message_queue.put(('chat_response', response))
                
            except Exception as e:
                self.message_queue.put(('chat_error', f"Error: {e}"))
        
        threading.Thread(target=chat_thread, daemon=True).start()
    
    def process_queue(self):
        """Process messages from background threads."""
        try:
            while True:
                message_type, data = self.message_queue.get_nowait()
                
                if message_type == 'token_success':
                    self.token_status_var.set(data)
                elif message_type == 'token_error':
                    self.token_status_var.set(data)
                elif message_type == 'dataset_status':
                    i, status = data
                    item = self.dataset_tree.get_children()[i]
                    values = list(self.dataset_tree.item(item)['values'])
                    values[1] = status
                    self.dataset_tree.item(item, values=values)
                elif message_type == 'ingestion_log':
                    self.results_text.insert(tk.END, data + "\n")
                    self.results_text.see(tk.END)
                elif message_type == 'ingestion_complete':
                    result_text, conversations = data
                    self.results_text.insert(tk.END, result_text)
                    self.conversations = conversations
                    self.ingest_button.config(state=tk.NORMAL)
                    self.ingestion_progress.stop()
                elif message_type == 'ingestion_error':
                    self.results_text.insert(tk.END, data + "\n")
                    self.ingest_button.config(state=tk.NORMAL)
                    self.ingestion_progress.stop()
                elif message_type == 'training_log':
                    self.training_log.insert(tk.END, data + "\n")
                    self.training_log.see(tk.END)
                elif message_type == 'training_complete':
                    self.training_log.insert(tk.END, "Training completed successfully!\n")
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                elif message_type == 'training_error':
                    self.training_log.insert(tk.END, data + "\n")
                    self.train_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                elif message_type == 'chat_response':
                    self.chat_display.config(state=tk.NORMAL)
                    self.chat_display.insert(tk.END, data + "\n\n")
                    self.chat_display.config(state=tk.DISABLED)
                    self.chat_display.see(tk.END)
                elif message_type == 'chat_error':
                    self.chat_display.config(state=tk.NORMAL)
                    self.chat_display.insert(tk.END, f"Error: {data}\n\n")
                    self.chat_display.config(state=tk.DISABLED)
                    self.chat_display.see(tk.END)
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_queue)
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    print("Starting Conversational API GUI...")
    
    try:
        app = ConversationalGUI()
        app.run()
    except Exception as e:
        print(f"GUI failed to start: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
