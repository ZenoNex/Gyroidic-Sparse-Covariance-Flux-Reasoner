from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

MAIN_BACKEND_URL = "http://localhost:5000"

@app.route('/api/status', methods=['GET'])
def get_status():
    """Proxy status to main backend"""
    try:
        res = requests.get(f"{MAIN_BACKEND_URL}/api/system_status", timeout=2)
        if res.status_code == 200:
            return jsonify(res.json())
        return jsonify({'error': 'Main backend error'}), res.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("[Voxelboxter Backend] Listening on port 1337 and forwarding to Main Backend at 5000")
    app.run(host='0.0.0.0', port=1337, debug=False)
