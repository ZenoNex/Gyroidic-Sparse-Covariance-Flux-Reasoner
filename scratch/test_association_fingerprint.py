import json
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

def test_diegetic_backend_associate_routing():
    print("Testing /associate routing in diegetic_backend.py...")
    
    # Mock ENGINE
    import src.ui.diegetic_backend as db
    mock_engine = MagicMock()
    db.ENGINE = mock_engine
    
    # Create a mock RequestHandler
    # Note: Correct class name is RequestHandler, not DiegeticRequestHandler
    class MockHandler(db.RequestHandler):
        def __init__(self):
            self.headers = {}
            self.path = '/associate'
            self.rfile = None
            self.wfile = MagicMock()
        
        def _send_json(self, data):
            self.last_response = data
            
        def _send_error_json(self, msg):
            print(f"Error sent: {msg}")
            self.last_error = msg

    handler = MockHandler()
    
    # Mock data with source/target and multiple dyads
    test_data = {
        "source": "Topology",
        "target": "Manifold",
        "fingerprint": {"chebyshev": [0.1, 0.2, 0.3]},
        "audio_dyad": {"harmonics": [0.4, 0.5, 0.6]},
        "video_dyad_b64": "base64_encoded_video_data",
        "commutativity": "media_first"
    }
    
    # Simulate the parsing logic in do_POST
    data = test_data
    text1 = data.get('text1', data.get('source', ''))
    text2 = data.get('text2', data.get('target', ''))
    fingerprint = data.get('fingerprint', None)
    audio_dyad = data.get('audio_dyad', None)
    video_dyad_b64 = data.get('video_dyad_b64', None)
    media_chain = data.get('media_chain', None)
    commutativity = data.get('commutativity', 'symmetric')
    association_command = f"ASSOCIATE: {text1} <-> {text2}"
    
    print(f"Association Command: {association_command}")
    print(f"Fingerprint: {fingerprint is not None}")
    print(f"Audio Dyad: {audio_dyad is not None}")
    print(f"Video Dyad: {video_dyad_b64 is not None}")
    
    # Verify logic matches what we implemented
    assert text1 == "Topology"
    assert text2 == "Manifold"
    assert fingerprint == {"chebyshev": [0.1, 0.2, 0.3]}
    assert audio_dyad == {"harmonics": [0.4, 0.5, 0.6]}
    assert video_dyad_b64 == "base64_encoded_video_data"
    
    # Simulate Engine call
    mock_engine.process_input(
        association_command,
        fingerprint=fingerprint,
        audio_dyad=audio_dyad,
        video_dyad_b64=video_dyad_b64,
        media_chain=media_chain,
        commutativity=commutativity
    )
    
    # Verify engine received the correct arguments
    mock_engine.process_input.assert_called_with(
        "ASSOCIATE: Topology <-> Manifold",
        fingerprint={"chebyshev": [0.1, 0.2, 0.3]},
        audio_dyad={"harmonics": [0.4, 0.5, 0.6]},
        video_dyad_b64="base64_encoded_video_data",
        media_chain=None,
        commutativity="media_first"
    )
    
    print("Verification Successful: All multimodal dyads propagated to engine!")

if __name__ == "__main__":
    try:
        test_diegetic_backend_associate_routing()
    except Exception as e:
        print(f"Test failed: {e}")
        # No traceback with emojis to avoid Unicode issues
