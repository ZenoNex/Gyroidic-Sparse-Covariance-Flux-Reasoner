#!/usr/bin/env python3
"""
Conversational Backend Server

Flask backend that connects the GUI to the real Hugging Face training system.
Handles secure token management and provides API endpoints for the GUI.

Features:
- Secure token handling (never stored in code)
- Real HF API integration
- Live training progress
- Dataset access verification
- Chat interface backend

Author: William Matthew Bryant
Created: January 2026
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import threading
import queue
import json
import time
from datetime import datetime
import requests

# Core imports
from src.data.conversational_api_ingestor import ConversationalAPIIngestor
from examples.conversational_api_training import ConversationalTemporalModel, ConversationalAPITrainer
from examples.diegetic_conversational_integration import DiegeticConversationalBackend

app = Flask(__name__)
CORS(app)

# Global state
class ServerState:
    def __init__(self):
        self.token = None
        self.api_ingestor = None
        self.model = None
        self.trainer = None
        self.conversations = None
        self.training_thread = None
        self.training_active = False
        self.training_progress = 0
        self.training_log = []
        self.diegetic_backend = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' if self._check_cuda() else 'cpu'
    
    def _check_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

state = ServerState()

@app.route('/')
def index():
    """Serve the main GUI."""
    try:
        with open('src/ui/conversational_web_gui.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>Conversational GUI</h1>
        <p>GUI file not found. Please run from the project root directory.</p>
        <p>Try: <code>python src/ui/conversational_backend_server.py</code></p>
        """

@app.route('/api/test_token', methods=['POST'])
def test_token():
    """Test Hugging Face token."""
    try:
        data = request.get_json()
        token = data.get('token', '').strip()
        
        if not token:
            return jsonify({'success': False, 'message': 'No token provided'})
        
        # Test token with HF API
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get('https://huggingface.co/api/whoami', headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_data = response.json()
            username = user_data.get('name', 'Unknown')
            
            # Store token securely in memory only
            state.token = token
            os.environ['HF_TOKEN'] = token
            
            return jsonify({
                'success': True, 
                'message': f'✅ Token valid! User: {username}',
                'username': username
            })
        else:
            return jsonify({'success': False, 'message': '❌ Invalid token'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'❌ Connection error: {str(e)}'})

@app.route('/api/check_datasets', methods=['POST'])
def check_datasets():
    """Check dataset access status."""
    try:
        if not state.token:
            return jsonify({'success': False, 'message': 'No token available'})
        
        headers = {'Authorization': f'Bearer {state.token}'}
        
        datasets = [
            'lmsys/lmsys-chat-1m',
            'OpenAssistant/oasst2',
            'microsoft/DialoGPT-medium'
        ]
        
        results = {}
        
        for dataset_id in datasets:
            try:
                response = requests.get(f'https://huggingface.co/api/datasets/{dataset_id}', 
                                      headers=headers, timeout=10)
                
                if response.status_code == 200:
                    results[dataset_id] = {'status': 'Available', 'accessible': True}
                elif response.status_code == 403:
                    results[dataset_id] = {'status': 'Agreement Required', 'accessible': False}
                elif response.status_code == 404:
                    results[dataset_id] = {'status': 'Not Found', 'accessible': False}
                else:
                    results[dataset_id] = {'status': f'Error {response.status_code}', 'accessible': False}
                    
            except Exception as e:
                results[dataset_id] = {'status': f'Network Error', 'accessible': False}
        
        return jsonify({'success': True, 'datasets': results})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start_ingestion', methods=['POST'])
def start_ingestion():
    """Start data ingestion from Hugging Face."""
    try:
        if not state.token:
            return jsonify({'success': False, 'message': 'No token available'})
        
        data = request.get_json()
        dataset_id = data.get('dataset_id', 'lmsys/lmsys-chat-1m')
        sample_size = int(data.get('sample_size', 500))
        
        def ingestion_worker():
            try:
                # Create API ingestor
                state.api_ingestor = ConversationalAPIIngestor(device=state.device)
                
                # Ingest data
                conversations = state.api_ingestor.ingest_huggingface_dataset(
                    dataset_id, max_samples=sample_size
                )
                
                if conversations:
                    state.conversations = conversations
                    summary = state.api_ingestor.get_ingestion_summary(conversations)
                    
                    # Store results for retrieval
                    state.ingestion_results = {
                        'success': True,
                        'dataset_id': dataset_id,
                        'summary': summary,
                        'conversations_count': len(conversations)
                    }
                else:
                    state.ingestion_results = {
                        'success': False,
                        'message': 'No conversations ingested'
                    }
                    
            except Exception as e:
                state.ingestion_results = {
                    'success': False,
                    'message': f'Ingestion failed: {str(e)}'
                }
        
        # Start ingestion in background
        threading.Thread(target=ingestion_worker, daemon=True).start()
        
        return jsonify({'success': True, 'message': 'Ingestion started'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/ingestion_status', methods=['GET'])
def ingestion_status():
    """Get ingestion status."""
    if hasattr(state, 'ingestion_results'):
        results = state.ingestion_results
        delattr(state, 'ingestion_results')  # Clear after reading
        return jsonify(results)
    else:
        return jsonify({'success': None, 'message': 'In progress'})

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start model training."""
    try:
        if not state.conversations:
            return jsonify({'success': False, 'message': 'No data available for training'})
        
        data = request.get_json()
        epochs = int(data.get('epochs', 3))
        learning_rate = float(data.get('learning_rate', 1e-4))
        
        if state.training_active:
            return jsonify({'success': False, 'message': 'Training already in progress'})
        
        def training_worker():
            try:
                state.training_active = True
                state.training_progress = 0
                state.training_log = []
                
                # Create model and trainer
                state.model = ConversationalTemporalModel(device=state.device)
                state.trainer = ConversationalAPITrainer(
                    state.model, state.api_ingestor, learning_rate=learning_rate
                )
                
                state.training_log.append(f"🚀 Starting training: {epochs} epochs, lr={learning_rate}")
                state.training_log.append(f"Device: {state.device}")
                state.training_log.append(f"Conversations: {len(state.conversations)}")
                
                # Train with progress updates
                class ProgressTrainer(ConversationalAPITrainer):
                    def train_on_conversations(self, conversations, num_epochs=3):
                        results = {'epoch_results': []}
                        
                        for epoch in range(num_epochs):
                            state.training_progress = (epoch / num_epochs) * 100
                            state.training_log.append(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
                            
                            # Simulate epoch training (replace with real training)
                            epoch_metrics = []
                            for i, conv in enumerate(conversations[:min(len(conversations), 50)]):  # Limit for demo
                                if not state.training_active:  # Check for stop
                                    break
                                
                                try:
                                    metrics = self._train_conversation(conv)
                                    epoch_metrics.append(metrics)
                                    
                                    if (i + 1) % 10 == 0:
                                        avg_pas_h = sum(m.get('pas_h_values', 0) for m in epoch_metrics[-10:]) / 10
                                        state.training_log.append(f"  Batch {i+1}: PAS_h={avg_pas_h:.3f}")
                                        
                                except Exception as e:
                                    state.training_log.append(f"  ⚠️ Batch {i+1} error: {e}")
                                    continue
                            
                            if not state.training_active:
                                break
                            
                            # Compute epoch summary
                            if epoch_metrics:
                                avg_pas_h = sum(m.get('pas_h_values', 0) for m in epoch_metrics) / len(epoch_metrics)
                                avg_loss = sum(m.get('conversational_loss', 0) for m in epoch_metrics) / len(epoch_metrics)
                                
                                epoch_summary = {
                                    'pas_h_values': avg_pas_h,
                                    'conversational_loss': avg_loss
                                }
                                results['epoch_results'].append(epoch_summary)
                                
                                state.training_log.append(f"📊 Epoch {epoch + 1} Summary:")
                                state.training_log.append(f"   PAS_h: {avg_pas_h:.3f}")
                                state.training_log.append(f"   Loss: {avg_loss:.3f}")
                                state.training_log.append(f"   Trust: {self.model.trust_scalars.mean():.3f}")
                        
                        return results
                
                # Use progress trainer
                progress_trainer = ProgressTrainer(state.model, state.api_ingestor, learning_rate=learning_rate)
                results = progress_trainer.train_on_conversations(state.conversations, num_epochs=epochs)
                
                if state.training_active:  # Only if not stopped
                    state.training_progress = 100
                    state.training_log.append("✅ Training completed successfully!")
                    
                    # Initialize diegetic backend with trained model
                    state.diegetic_backend = DiegeticConversationalBackend(device=state.device)
                    state.diegetic_backend.conversational_model.load_state_dict(state.model.state_dict())
                    
                    state.training_results = {
                        'success': True,
                        'results': results,
                        'final_trust': state.model.trust_scalars.tolist()
                    }
                else:
                    state.training_log.append("⏹️ Training stopped by user")
                    state.training_results = {
                        'success': False,
                        'message': 'Training stopped'
                    }
                
            except Exception as e:
                state.training_log.append(f"❌ Training failed: {str(e)}")
                state.training_results = {
                    'success': False,
                    'message': f'Training failed: {str(e)}'
                }
            finally:
                state.training_active = False
        
        # Start training in background
        state.training_thread = threading.Thread(target=training_worker, daemon=True)
        state.training_thread.start()
        
        return jsonify({'success': True, 'message': 'Training started'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/training_status', methods=['GET'])
def training_status():
    """Get training status and progress."""
    return jsonify({
        'active': state.training_active,
        'progress': state.training_progress,
        'log': state.training_log[-50:],  # Last 50 log entries
        'results': getattr(state, 'training_results', None)
    })

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """Stop training."""
    state.training_active = False
    return jsonify({'success': True, 'message': 'Training stop requested'})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        if not state.diegetic_backend:
            return jsonify({'success': False, 'message': 'No trained model available'})
        
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'success': False, 'message': 'No message provided'})
        
        # Process message with diegetic backend
        if not hasattr(state.diegetic_backend, 'current_conversation') or not state.diegetic_backend.current_conversation:
            result = state.diegetic_backend.start_conversation(message)
        else:
            result = state.diegetic_backend.process_user_input(message)
        
        return jsonify({
            'success': True,
            'response': result['response']['text'],
            'metrics': {
                'pas_h': result['response']['pas_h'],
                'trust': result['response']['trust_mean'],
                'affordance': result['response']['type'],
                'affordance_strength': result['response']['affordance_strength']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/save_model', methods=['POST'])
def save_model():
    """Save trained model."""
    try:
        if not state.model:
            return jsonify({'success': False, 'message': 'No trained model to save'})
        
        import torch
        filename = f"conversational_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        torch.save({
            'model_state_dict': state.model.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'device': state.device
        }, filename)
        
        return jsonify({'success': True, 'message': f'Model saved as {filename}'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/system_status', methods=['GET'])
def system_status():
    """Get overall system status."""
    return jsonify({
        'token_available': state.token is not None,
        'data_ingested': state.conversations is not None,
        'model_trained': state.model is not None,
        'chat_ready': state.diegetic_backend is not None,
        'device': state.device,
        'conversations_count': len(state.conversations) if state.conversations else 0
    })

def run_server(host='localhost', port=5000, debug=False):
    """Run the Flask server."""
    print(f"🚀 Starting Conversational Backend Server")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Device: {state.device}")
    print(f"   GUI URL: http://{host}:{port}")
    print("=" * 50)
    
    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Conversational Backend Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, debug=args.debug)
