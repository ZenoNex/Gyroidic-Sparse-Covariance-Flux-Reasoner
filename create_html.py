#!/usr/bin/env python3
"""Create the HTML file directly using Python."""

html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wikipedia Knowledge Ingestion System</title>
    <style>
        body {
            font-family: 'Consolas', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #1a1a1a;
            border-radius: 12px;
            border: 1px solid #404040;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            color: #4a9eff;
        }
        .panel {
            background: #1a1a1a;
            border-radius: 12px;
            border: 1px solid #404040;
            padding: 20px;
            margin-bottom: 20px;
        }
        .panel-title {
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #4a9eff;
        }
        .input-field {
            width: 100%;
            padding: 12px;
            background: #2a2a2a;
            border: 1px solid #404040;
            border-radius: 8px;
            color: #e0e0e0;
            font-family: inherit;
            margin-bottom: 15px;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            background: #4a9eff;
            color: white;
            cursor: pointer;
            font-family: inherit;
        }
        .btn:hover {
            background: #3b82f6;
        }
        .log-container {
            background: #0a0a0a;
            border: 1px solid #404040;
            border-radius: 8px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 0.85rem;
        }
        .log-entry {
            margin-bottom: 8px;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .log-info { color: #4a9eff; }
        .log-success { color: #4ade80; }
        .log-error { color: #ef4444; background: rgba(239, 68, 68, 0.1); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Wikipedia Knowledge Ingestion System</h1>
            <p>Automated training system with smart noise filtering and concept extraction</p>
        </div>

        <div class="panel">
            <div class="panel-title">Content Input</div>
            <textarea id="wiki-urls" class="input-field" rows="4" placeholder="https://en.wikipedia.org/wiki/Quantum_mechanics
https://en.wikipedia.org/wiki/Machine_learning"></textarea>
        </div>

        <div class="panel">
            <div class="panel-title">Extraction Options</div>
            <label><input type="checkbox" id="filter-noise" checked> Smart noise filtering</label><br>
            <label><input type="checkbox" id="auto-concepts" checked> Auto-detect source concepts</label><br><br>
            <button class="btn" onclick="startExtraction()">🚀 Start Training</button>
        </div>

        <div class="panel">
            <div class="panel-title">Training Status</div>
            <div class="log-container" id="log-container">
                <div class="log-entry log-info">🔧 Wikipedia Knowledge Ingestion System initialized</div>
                <div class="log-entry log-info">📚 Ready to process Wikipedia content</div>
            </div>
        </div>
    </div>

    <script>
        function startExtraction() {
            addLog('🚀 Starting Wikipedia extraction...', 'info');
            addLog('✅ System is working! HTML loaded successfully.', 'success');
        }

        function addLog(message, type = 'info') {
            const container = document.getElementById('log-container');
            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }

        document.addEventListener('DOMContentLoaded', function() {
            addLog('🔧 System ready for Wikipedia knowledge ingestion', 'info');
        });
    </script>
</body>
</html>'''

# Write the file
import os
os.makedirs('src/ui', exist_ok=True)

with open('src/ui/wikipedia_trainer.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ HTML file created successfully!")
print(f"📊 File size: {os.path.getsize('src/ui/wikipedia_trainer.html')} bytes")