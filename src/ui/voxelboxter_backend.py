import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

MAIN_BACKEND_URL = os.environ.get("MAIN_BACKEND_URL", "http://localhost:8000")

@app.route('/api/status', methods=['GET'])
@app.route('/api/system_status', methods=['GET'])
def get_status():
    """Proxy status to main backend"""
    try:
        res = requests.get(f"{MAIN_BACKEND_URL}/api/system_status", timeout=2)
        if res.status_code == 200:
            return jsonify(res.json())
        return jsonify({'error': 'Main backend error'}), res.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/scripting/resources', methods=['GET'])
def get_scripting_resources():
    """Query material properties and resource parameters from ResourceRegistry."""
    try:
        from src.scripting.node_environment import ResourceRegistry
        registry = ResourceRegistry()
        resources = [
            {
                "name": r.name,
                "material_id": r.material_id,
                "density": r.density,
                "hardness": r.hardness,
                "energy_density": r.energy_density,
                "base_value": r.base_value,
                "tags": r.tags
            }
            for r in registry.all_resources()
        ]
        return jsonify({"status": "ok", "resources": resources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/scripting/ore_dict', methods=['GET'])
def get_ore_dictionary():
    """Query unified canonical material tags from OreDictionary."""
    try:
        from src.scripting.node_environment import OreDictionary
        ore_dict = OreDictionary()
        mappings = {
            tag: ore_dict.get_materials(tag)
            for tag in ore_dict._tag_to_materials
        }
        return jsonify({"status": "ok", "tags": mappings})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("VOXELBOXTER_PORT", 1337))
    print(f"[Voxelboxter Backend] Listening on port {port} and forwarding to Main Backend at {MAIN_BACKEND_URL}")
    app.run(host='0.0.0.0', port=port, debug=False)
