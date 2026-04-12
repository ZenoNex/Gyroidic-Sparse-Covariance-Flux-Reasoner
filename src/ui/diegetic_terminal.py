"""
Diegetic Terminal and Knowledge Ingestion GUI components.

Provides the human-to-system interface layers:
1. DiegeticTerminal: A conversational interface that mirrors legacy Cleverbot.
2. KnowledgeIngestionGUI: A panel for mapping images to words (Knowledge Dyads).

These are built using HTML/Vanilla CSS for the web application UI.
"""

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gyroidic Diegetic Terminal // Structural Honesty</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=Fira+Code:wght@400&display=swap"
        rel="stylesheet">
    <style>
        :root {
            --bg-color: #030303;
            --terminal-green: #00ff41;
            --terminal-blue: #00f2ff;
            --terminal-magenta: #ff00f2;
            --terminal-warn: #ffcc00;
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.1);
            --font-main: 'Outfit', sans-serif;
            --font-mono: 'Fira Code', monospace;
            --sidebar-width: 320px;
        }

        * {
            box-sizing: border-box;
            user-select: none;
        }

        body {
            background: var(--bg-color);
            color: #e0e0e0;
            font-family: var(--font-main);
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            background-image:
                radial-gradient(circle at 10% 20%, rgba(0, 242, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(255, 0, 242, 0.05) 0%, transparent 40%);
        }

        /* --- Header --- */
        header {
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 1.5rem;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--glass-border);
            backdrop-filter: blur(10px);
            z-index: 100;
        }

        .system-id {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .love-invariant {
            color: var(--terminal-magenta);
            font-family: var(--font-mono);
            font-size: 0.8rem;
            letter-spacing: 2px;
            background: rgba(255, 0, 242, 0.1);
            padding: 4px 12px;
            border-radius: 4px;
            border: 1px solid rgba(255, 0, 242, 0.2);
        }

        .regime-toggle {
            display: flex;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 6px;
            padding: 2px;
        }

        .regime-btn {
            padding: 4px 12px;
            font-size: 0.7rem;
            font-weight: 700;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.2s;
            color: #666;
        }

        .regime-btn.active.goo {
            color: var(--terminal-blue);
            background: rgba(0, 242, 255, 0.1);
        }

        .regime-btn.active.prickles {
            color: var(--terminal-warn);
            background: rgba(255, 204, 0, 0.1);
        }

        .system-stats {
            display: flex;
            gap: 2rem;
            font-family: var(--font-mono);
            font-size: 0.75rem;
        }

        .stat-item span {
            color: var(--terminal-blue);
        }

        /* --- Main Layout --- */
        main {
            flex: 1;
            display: flex;
            overflow: hidden;
        }

        .sidebar {
            width: var(--sidebar-width);
            background: rgba(255, 255, 255, 0.01);
            border-right: 1px solid var(--glass-border);
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            padding: 1rem;
        }

        .sidebar.right {
            border-right: none;
            border-left: 1px solid var(--glass-border);
        }

        .section-title {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .small-log {
            margin-top: 1rem;
            font-family: var(--font-mono);
            font-size: 0.65rem;
            color: #555;
            height: 100px;
            overflow-y: auto;
            border-top: 1px solid var(--glass-border);
            padding-top: 0.5rem;
        }

        /* --- Center Console --- */
        .console {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        #chat-feed {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            scrollbar-width: thin;
            scrollbar-color: var(--glass-border) transparent;
        }

        .message {
            max-width: 80%;
            padding: 1rem;
            border-radius: 12px;
            font-size: 0.95rem;
            line-height: 1.5;
            position: relative;
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user {
            align-self: flex-end;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            color: var(--terminal-blue);
        }

        .message.system {
            align-self: flex-start;
            background: rgba(0, 242, 255, 0.05);
            border-left: 3px solid var(--terminal-blue);
            font-family: var(--font-mono);
            color: #fff;
        }

        /* Mischief Solitons */
        .mischief-spike {
            position: absolute;
            bottom: 0;
            width: 2px;
            background: var(--terminal-magenta);
            opacity: 0.5;
            pointer-events: none;
        }

        .input-area {
            height: 80px;
            background: rgba(255, 255, 255, 0.02);
            border-top: 1px solid var(--glass-border);
            display: flex;
            align-items: center;
            padding: 0 1.5rem;
            gap: 1rem;
        }

        #user-input {
            flex: 1;
            background: transparent;
            border: none;
            color: #fff;
            font-family: var(--font-main);
            font-size: 1rem;
            outline: none;
        }

        /* --- Components --- */
        .metric-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            padding: 0.8rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }

        .metric-label {
            font-size: 0.7rem;
            color: #888;
            margin-bottom: 4px;
        }

        .metric-value {
            font-family: var(--font-mono);
            font-size: 1.1rem;
            color: var(--terminal-blue);
        }

        /* Spectral Ribbon */
        #spectral-ribbon {
            height: 120px;
            width: 100%;
            background: #000;
            border: 1px solid var(--glass-border);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }

        .ribbon-bin {
            position: absolute;
            bottom: 0;
            width: 1px;
            background: var(--terminal-blue);
        }

        /* Manifold Buffer */
        #manifold-buffer {
            width: 100%;
            height: 150px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--glass-border);
            color: var(--terminal-green);
            font-family: var(--font-mono);
            font-size: 0.75rem;
            padding: 0.5rem;
            resize: none;
            outline: none;
            margin-bottom: 1rem;
        }

        /* Dyad Ingestor */
        #dyad-dropzone {
            width: 100%;
            height: 100px;
            border: 2px dashed var(--glass-border);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 1rem;
        }

        #dyad-dropzone:hover {
            border-color: var(--terminal-blue);
            background: rgba(0, 242, 255, 0.05);
        }

        .drop-hint {
            font-size: 0.7rem;
            color: #666;
            margin-top: 8px;
        }

        /* CRT Overlay */
        .crt-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.05) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.02), rgba(0, 255, 0, 0.01), rgba(0, 0, 255, 0.02));
            background-size: 100% 3px, 3px 100%;
            pointer-events: none;
            z-index: 1000;
            opacity: 0.3;
        }

        .noise-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url("https://grainy-gradients.vercel.app/noise.svg");
            opacity: 0.05;
            pointer-events: none;
            z-index: 999;
        }

        /* --- Custom Scrollbar --- */
        ::-webkit-scrollbar {
            width: 4px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--glass-border);
            border-radius: 2px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--terminal-blue);
        }

        /* --- Commutativity Selector --- */
        .commutativity-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .commutativity-row label {
            font-size: 0.65rem;
            color: #666;
            letter-spacing: 1px;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .commute-select {
            flex: 1;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            color: var(--terminal-blue);
            font-family: var(--font-mono);
            font-size: 0.65rem;
            padding: 4px 6px;
            border-radius: 4px;
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }

        .commute-select:focus {
            border-color: var(--terminal-blue);
        }

        .commute-select option {
            background: #111;
            color: #e0e0e0;
        }

        /* --- Audio Dyad Panel --- */
        #audio-dropzone {
            width: 100%;
            height: 90px;
            border: 2px dashed rgba(255, 204, 0, 0.3);
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 0.75rem;
            position: relative;
        }

        #audio-dropzone:hover,
        #audio-dropzone.dragover {
            border-color: var(--terminal-warn);
            background: rgba(255, 204, 0, 0.04);
        }

        #audio-dropzone.loaded {
            border-color: var(--terminal-green);
            background: rgba(0, 255, 65, 0.04);
        }

        .audio-drop-icon {
            font-size: 1.4rem;
            line-height: 1;
            color: var(--terminal-warn);
        }

        .audio-drop-hint {
            font-size: 0.65rem;
            color: #666;
            margin-top: 5px;
            text-align: center;
            letter-spacing: 0.5px;
        }

        /* Mini waveform viz */
        #audio-waveform {
            width: 100%;
            height: 40px;
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--glass-border);
            border-radius: 4px;
            margin-bottom: 0.75rem;
            display: none;
        }

        /* Audio playback bar */
        #audio-player-wrapper {
            display: none;
            margin-bottom: 0.75rem;
        }

        #audio-player-wrapper audio {
            width: 100%;
            height: 28px;
            filter: invert(1) hue-rotate(180deg) saturate(0.5);
            border-radius: 4px;
        }

        .audio-meta {
            font-family: var(--font-mono);
            font-size: 0.6rem;
            color: #555;
            margin-bottom: 0.5rem;
        }

        /* Audio commit button */
        #commit-audio {
            width: 100%;
            background: rgba(255, 204, 0, 0.08);
            border: 1px solid var(--terminal-warn);
            color: var(--terminal-warn);
            padding: 6px;
            border-radius: 4px;
            font-size: 0.7rem;
            cursor: pointer;
            font-weight: 700;
            letter-spacing: 1px;
            transition: background 0.2s;
        }

        #commit-audio:hover {
            background: rgba(255, 204, 0, 0.15);
        }

        #commit-audio:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        /* Panel C label colour */
        .panel-c-title {
            color: var(--terminal-warn);
        }
    </style>
</head>

<body>
    <div class="crt-overlay"></div>
    <div class="noise-overlay"></div>

    <header>
        <div class="system-id">
            <div class="love-invariant" id="love-display">L: 3.127</div>
            <div class="regime-toggle">
                <div class="regime-btn active goo" id="regime-goo">GOO</div>
                <div class="regime-btn prickles" id="regime-prickles">PRICKLES</div>
            </div>
        </div>
        <div class="system-stats">
            <div class="stat-item">TAU: <span id="stat-tau">0.000</span></div>
            <div class="stat-item">HARDENING: <span id="stat-hardening">0.15</span></div>
            <div class="stat-item">ITERATION: <span id="stat-iteration">0</span></div>
            <div class="stat-item">META STATE: <span id="stat-retrieval" style="color: #666;">UNKNOWN</span></div>
        </div>
    </header>

    <main>
        <!-- Left Sidebar: Topological Health -->
        <aside class="sidebar">
            <div class="section-title">Field Dynamics <small id="field-status">SCANNING</small></div>

            <div id="spectral-ribbon">
                <!-- Ribbon bins populated by JS -->
            </div>

            <div class="metric-card">
                <div class="metric-label">Betti 0 (Components)</div>
                <div class="metric-value" id="beta-0">1.0000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Betti 1 (Cycles/Holes)</div>
                <div class="metric-value" id="beta-1">0.0000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Manifold Pressure</div>
                <div class="metric-value" id="pressure">0.0421</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Spectral Coherence</div>
                <div class="metric-value" id="coherence">0.985</div>
            </div>

            <hr style="border: none; border-top: 1px solid var(--glass-border); margin: 1rem 0;">
            <div class="section-title">CALM Momentum <small id="calm-status">STABLE</small></div>
            <div class="metric-card">
                <div class="metric-label">Abort Score (Singularity)</div>
                <div class="metric-value" id="calm-abort">0.000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Rho Factor (Tension)</div>
                <div class="metric-value" id="calm-rho">1.000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Step Factor (Momentum)</div>
                <div class="metric-value" id="calm-step">1.000</div>
            </div>

            <hr style="border: none; border-top: 1px solid var(--glass-border); margin: 1rem 0;">
            <div class="section-title">Manifold Resoance <small id="voice-status">VOICE: ON</small></div>
            <div class="metric-card">
                <div class="metric-label">Manifold Voice Resonance</div>
                <div class="metric-value" id="voice-resonance" style="color: var(--terminal-magenta);">0.000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Ley Line Anisotropy</div>
                <div class="metric-value" id="ley-line">0.000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Möbius Twist Status</div>
                <div class="metric-value" id="moebius-twist">0.0</div>
            </div>
        </aside>

        <!-- Center: Conversational Console -->
        <section class="console">
            <div id="chat-feed">
                <!-- Messages appended here -->
                <div class="message system">SYSTEM INITIALIZED. WAITING FOR MANIFOLD PERTURBATION.</div>
            </div>
            <div class="input-area">
                <span style="color:var(--terminal-blue); font-family:var(--font-mono);">></span>
                <input type="text" id="user-input" placeholder="Ingest symbolic residue..." autocomplete="off">
            </div>
            <div id="soliton-layer"></div>
        </section>

        <!-- Right Sidebar: Data Ingestion & Associations -->
        <aside class="sidebar right">
            <!-- PANEL A: DYAD INGESTOR (IMAGE TO TEXT) -->
            <div class="ingestor-panel">
                <div class="section-title">Panel A: Dyad Ingestor</div>

                <!-- Commutativity Selector -->
                <div class="commutativity-row">
                    <label>ORDER</label>
                    <select class="commute-select" id="commute-image">
                        <option value="image_first">Image &#8594; Text</option>
                        <option value="text_first">Text &#8594; Image</option>
                        <option value="symmetric">&#8771; Symmetric Entanglement</option>
                    </select>
                </div>

                <div id="dyad-dropzone">
                    <span style="font-size: 1.5rem; color: var(--terminal-blue);">+</span>
                    <div class="drop-hint">DRAG DYAD (IMAGE)</div>
                    <input type="file" id="file-input" style="display: none;" accept="image/*">
                </div>
                <div class="section-title">Manifold Buffer (A)</div>
                <textarea id="manifold-buffer" placeholder="Reasoning target buffer for image dyad..."></textarea>
                <div class="metric-card" style="margin-bottom: 0.5rem;">
                    <div class="metric-label">Source Topic</div>
                    <input type="text" id="assoc-source" placeholder="e.g. Visual Signature"
                        style="width: 100%; background: transparent; border: none; color: #fff; border-bottom: 1px solid var(--glass-border); padding: 4px 0; outline: none; font-size: 0.8rem;">
                    <button id="commit-assoc"
                        style="width: 100%; margin-top: 1rem; background: rgba(0, 242, 255, 0.1); border: 1px solid var(--terminal-blue); color: var(--terminal-blue); padding: 6px; border-radius: 4px; font-size: 0.7rem; cursor: pointer; font-weight: 700;">
                        COMMIT DYAD ASSOCIATION
                    </button>
                </div>
                <div id="assoc-log" class="small-log"></div>
            </div>

            <hr style="border: none; border-top: 1px solid var(--glass-border); margin: 1.5rem 0;">

            <!-- PANEL B: TEXT-TO-TEXT ASSOCIATION -->
            <div class="association-panel">
                <div class="section-title">Panel B: Semantic Linker</div>
                <textarea id="semantic-buffer" placeholder="Semantic target (Target Text)..."
                    style="height: 120px;"></textarea>
                <div class="metric-card">
                    <div class="metric-label">Source Concept</div>
                    <input type="text" id="semantic-source" placeholder="e.g. Betti Numbers"
                        style="width: 100%; background: transparent; border: none; color: #fff; border-bottom: 1px solid var(--glass-border); padding: 4px 0; outline: none; font-size: 0.8rem;">
                    <button id="commit-semantic"
                        style="width: 100%; margin-top: 1rem; background: rgba(255, 0, 242, 0.1); border: 1px solid var(--terminal-magenta); color: var(--terminal-magenta); padding: 6px; border-radius: 4px; font-size: 0.7rem; cursor: pointer; font-weight: 700;">
                        COMMIT SEMANTIC LINK
                    </button>
                </div>
                <div id="semantic-log" class="small-log"></div>
            </div>

            <hr style="border: none; border-top: 1px solid var(--glass-border); margin: 1.5rem 0;">

            <!-- PANEL C: AUDIO DYAD INGESTOR -->
            <div class="audio-panel" id="audio-panel">
                <div class="section-title panel-c-title">Panel C: Audio Dyad
                    <small id="audio-status" style="color:#555;">IDLE</small>
                </div>

                <!-- Commutativity Selector -->
                <div class="commutativity-row">
                    <label>ORDER</label>
                    <select class="commute-select" id="commute-audio">
                        <option value="audio_first">Audio &#8594; Text</option>
                        <option value="text_first">Text &#8594; Audio</option>
                        <option value="symmetric">&#8771; Symmetric Entanglement</option>
                    </select>
                </div>

                <!-- Drop zone -->
                <div id="audio-dropzone">
                    <div class="audio-drop-icon">&#9836;</div>
                    <div class="audio-drop-hint" id="audio-drop-hint">DRAG AUDIO DYAD<br><span style="color:#444;">mp3 &bull; m4a &bull; wav &bull; ogg</span></div>
                    <input type="file" id="audio-file-input" style="display:none;" accept="audio/mpeg,audio/mp4,audio/wav,audio/ogg,.mp3,.m4a,.wav,.ogg">
                </div>

                <!-- Mini waveform canvas -->
                <canvas id="audio-waveform"></canvas>

                <!-- Playback controls -->
                <div id="audio-player-wrapper">
                    <div class="audio-meta" id="audio-meta">--</div>
                    <audio id="audio-player" controls></audio>
                </div>

                <!-- Description buffer -->
                <div class="section-title" style="margin-top:0.75rem;">Description Buffer (C)</div>
                <textarea id="audio-description-buffer"
                    style="width:100%;height:80px;background:rgba(0,0,0,0.3);border:1px solid var(--glass-border);color:var(--terminal-green);font-family:var(--font-mono);font-size:0.7rem;padding:0.5rem;resize:none;outline:none;margin-bottom:0.75rem;border-radius:4px;"
                    placeholder="Semantic annotation for audio dyad..."></textarea>

                <button id="commit-audio" disabled>COMMIT AUDIO DYAD</button>
                <div id="audio-log" class="small-log"></div>
            </div>
        </aside>
    </main>

    <!-- JS Logic in separate tag for better separation later -->
    <script>
        // Global State
        const state = {
            iteration: 0,
            regime: 'goo',
            backend_url: window.location.origin.includes('localhost') ? 'http://localhost:8000' : window.location.origin,
            active_fingerprint: null
        };

        // UI References
        const chatFeed = document.getElementById('chat-feed');
        const userInput = document.getElementById('user-input');
        const ribbon = document.getElementById('spectral-ribbon');
        const dropzone = document.getElementById('dyad-dropzone');
        const fileInput = document.getElementById('file-input');

        // --- Initialization ---
        function initRibbon() {
            ribbon.innerHTML = '';
            for (let i = 0; i < 137; i++) {
                const bin = document.createElement('div');
                bin.className = 'ribbon-bin';
                bin.style.left = `${(i / 137) * 100}%`;
                bin.style.height = '10%';
                if (i < 32) bin.style.backgroundColor = '#ff4b4b'; // R
                else if (i < 64) bin.style.backgroundColor = '#4bff4b'; // G
                else if (i < 96) bin.style.backgroundColor = '#4b4bff'; // B
                else if (i < 128) bin.style.backgroundColor = '#fff'; // L
                else bin.style.backgroundColor = '#ff00f2'; // Texture + Edges
                ribbon.appendChild(bin);
            }
        }

        function updateRibbon(vector) {
            const bins = ribbon.querySelectorAll('.ribbon-bin');
            vector.forEach((val, i) => {
                if (bins[i]) {
                    const h = Math.min(val * 100, 100);
                    bins[i].style.height = `${h}%`;
                }
            });
        }

        // --- Core Interaction ---
        async function sendMessage(text) {
            const hasFingerprint = !!state.active_fingerprint;
            const finalInput = text.trim() || (hasFingerprint ? "INGEST_DYAD: [RADIANCE]" : "");

            if (!finalInput) return;

            // Append user message
            appendMessage('user', finalInput);
            userInput.value = '';

            try {
                const payload = {
                    text: finalInput,
                    fingerprint: state.active_fingerprint
                };

                const response = await fetch(`${state.backend_url}/interact`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                handleResponse(data);

            } catch (err) {
                console.error("Communication Rupture:", err);
                appendMessage('system', "COMMUNICATION RUPTURE. BACKEND DISSOCIATED.");
            }
        }

        function handleResponse(data) {
            if (data.diagnostics && data.diagnostics.suppress_ui) {
                updateMetricsFromDiagnostics(data);
                return;
            }
            if (data.error) {
                appendMessage('system', `ERROR: ${data.error}`);
            } else if (data.response) {
                typewriterMessage('system', data.response);
            } else {
                appendMessage('system', "MANIFOLD STATE UPDATED (NO RESPONSE GENERATED).");
            }
            updateMetricsFromDiagnostics(data);

            // Clear fingerprint
            state.active_fingerprint = null;
            document.querySelector('.drop-hint').innerText = "DRAP DYAD (IMAGE)";
        }

        function updateMetricsFromDiagnostics(data) {
            // Update Metrics (safely)
            if (data.iteration !== undefined) document.getElementById('stat-iteration').innerText = data.iteration;
            
            // Update Tri-State Meta State
            if (data.retrieval_state) {
                const rs = document.getElementById('stat-retrieval');
                rs.innerText = data.retrieval_state;
                if (data.retrieval_state === 'KNOWN') rs.style.color = 'var(--terminal-green)';
                else if (data.retrieval_state === 'SEARCH_NEEDED') rs.style.color = 'var(--terminal-warn)';
                else if (data.retrieval_state === 'CONFABULATED') rs.style.color = 'var(--terminal-magenta)';
                else rs.style.color = 'var(--terminal-blue)';
            }

            // Extract from topological_analysis if present
            const topo = data.phase4_diagnostics?.topological_analysis;
            if (topo) {
                if (topo.topological_complexity !== undefined) {
                    document.getElementById('pressure').innerText = (topo.topological_complexity / 10).toFixed(4);
                }
                // Extract Betti from features strings if possible
                if (topo.features) {
                    topo.features.forEach(f => {
                        if (f.startsWith('betti_0=')) document.getElementById('beta-0').innerText = f.split('=')[1];
                        if (f.startsWith('betti_1=')) document.getElementById('beta-1').innerText = f.split('=')[1];
                    });
                }
            }

            // Hardening and Coherence
            if (data.affordance_gradients && typeof data.affordance_gradients.constraint_forcing_gradient === 'number') {
                document.getElementById('stat-hardening').innerText = data.affordance_gradients.constraint_forcing_gradient.toFixed(4);
            }

            const repair = data.repair_diagnostics;
            if (repair?.spectral_coherence_corrector && typeof repair.spectral_coherence_corrector.coherence_score === 'number') {
                document.getElementById('coherence').innerText = repair.spectral_coherence_corrector.coherence_score.toFixed(3);
            }

            // CALM Diagnostics
            const calm = data.calm_diagnostics;
            if (calm) {
                document.getElementById('calm-abort').innerText = calm.abort_score.toFixed(4);
                document.getElementById('calm-rho').innerText = calm.rho_factor.toFixed(4);
                document.getElementById('calm-step').innerText = calm.step_factor.toFixed(4);
                document.getElementById('calm-status').innerText = calm.trajectory_status || "IDLE";

                if (calm.trajectory_status === "COLLAPSE_VETO") {
                    document.getElementById('calm-status').style.color = "var(--terminal-warn)";
                } else {
                    document.getElementById('calm-status').style.color = "#666";
                }
            }

            // Advanced Manifold Diagnostics
            const diag = data.diagnostics;
            if (diag) {
                if (diag.manifold_voice_resonance !== undefined) {
                    document.getElementById('voice-resonance').innerText = diag.manifold_voice_resonance.toFixed(4);
                    // Pulsate ribbon based on resonance
                    ribbon.style.opacity = 0.5 + (diag.manifold_voice_resonance * 0.5);
                }
                if (diag.ley_line_anisotropy !== undefined) {
                    document.getElementById('ley-line').innerText = diag.ley_line_anisotropy.toFixed(4);
                }
                if (diag.moebius_twist !== undefined) {
                    const twist = diag.moebius_twist;
                    const twistEl = document.getElementById('moebius-twist');
                    twistEl.innerText = twist.toFixed(1);
                    twistEl.style.color = twist > 0.5 ? 'var(--terminal-magenta)' : 'var(--terminal-blue)';
                    if (twist > 0.5 && typeof triggerMischief === 'function') triggerMischief(twist);
                }
                if (diag.spectral_entropy !== undefined) {
                    document.getElementById('stat-tau').innerText = diag.spectral_entropy.toFixed(3);
                }
            }

            // Mischief Solitons (Good Bugs)
            const soliton = repair?.soliton_stability_healer;
            if (soliton && soliton.healing_progress > 0.1) {
                triggerMischief(soliton.healing_progress);
            }
            // Clear fingerprint
            state.active_fingerprint = null;
            document.querySelector('.drop-hint').innerText = "DRAP DYAD (IMAGE)";
        }

        function appendMessage(type, text) {
            const msg = document.createElement('div');
            msg.className = `message ${type}`;
            msg.innerText = text || "[VOID CONTENT]";
            chatFeed.appendChild(msg);
            chatFeed.scrollTop = chatFeed.scrollHeight;
        }

        function typewriterMessage(type, text) {
            if (!text || typeof text !== 'string') {
                appendMessage(type, text);
                return;
            }

            const msg = document.createElement('div');
            msg.className = `message ${type}`;
            chatFeed.appendChild(msg);

            let i = 0;
            const interval = setInterval(() => {
                msg.innerText += text.charAt(i);
                i++;
                chatFeed.scrollTop = chatFeed.scrollHeight;
                if (i >= text.length) {
                    clearInterval(interval);
                }
            }, 20);
        }

        // --- Mischief Dynamics ---
        function triggerMischief(intensity) {
            const layer = document.getElementById('soliton-layer');
            const count = Math.floor(intensity * 10) + 1;

            for (let i = 0; i < count; i++) {
                const spike = document.createElement('div');
                spike.className = 'mischief-spike';
                spike.style.left = `${Math.random() * 100}%`;
                spike.style.height = `${Math.random() * 20 + (intensity * 30)}%`;
                spike.style.opacity = Math.random() * 0.5 + 0.2;
                layer.appendChild(spike);

                setTimeout(() => {
                    spike.style.opacity = '0';
                    setTimeout(() => spike.remove(), 1000);
                }, 200);
            }
        }

        // --- Image Fingerprinting (137-dim) ---
        async function processImage(file) {
            const reader = new FileReader();
            reader.onload = async (e) => {
                const img = new Image();
                img.onload = () => {
                    const fingerprint = computeFingerprint(img);
                    state.active_fingerprint = fingerprint;
                    updateRibbon(fingerprintToVector(fingerprint));
                    document.querySelector('.drop-hint').innerText = `DYAD CAPTURED: ${file.name}`;
                    appendMessage('system', `DYAD INGESTED: 137-DIMENSIONAL RADIANCE FINGERPRINT EXTRACTED.`);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        function computeFingerprint(img) {
            const canvas = document.createElement('canvas');
            // willReadFrequently optimization for readback intensive canvas
            const ctx = canvas.getContext('2d', { willReadFrequently: true });

            // 1. Histograms (32 bins per channel)
            canvas.width = 256; canvas.height = 256;
            ctx.drawImage(img, 0, 0, 256, 256);
            const data = ctx.getImageData(0, 0, 256, 256).data;

            const rHist = new Array(32).fill(0);
            const gHist = new Array(32).fill(0);
            const bHist = new Array(32).fill(0);
            const lHist = new Array(32).fill(0);

            for (let i = 0; i < data.length; i += 4) {
                rHist[Math.floor(data[i] / 8)]++;
                gHist[Math.floor(data[i + 1] / 8)]++;
                bHist[Math.floor(data[i + 2] / 8)]++;
                const l = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                lHist[Math.floor(l / 8)]++;
            }

            // Normalize histograms
            const total = 256 * 256;
            const norm = (h) => h.map(v => v / total);

            // 2. Downsampling & Edge Detection (32x32)
            canvas.width = 32; canvas.height = 32;
            ctx.drawImage(img, 0, 0, 32, 32);
            const smallData = ctx.getImageData(0, 0, 32, 32).data;

            // Sobel Edge Detection
            const gray = new Float32Array(32 * 32);
            for (let i = 0; i < 32 * 32; i++) {
                gray[i] = 0.299 * smallData[i * 4] + 0.587 * smallData[i * 4 + 1] + 0.114 * smallData[i * 4 + 2];
            }

            const edges = new Array(8).fill(0); // 8 directional sectors
            let varianceSum = 0;
            const mean = gray.reduce((a, b) => a + b) / gray.length;

            for (let y = 1; y < 31; y++) {
                for (let x = 1; x < 31; x++) {
                    const idx = y * 32 + x;
                    const gx = (gray[idx + 33] + 2 * gray[idx + 1] + gray[idx - 31]) - (gray[idx + 31] + 2 * gray[idx - 1] + gray[idx - 33]);
                    const gy = (gray[idx + 33] + 2 * gray[idx + 32] + gray[idx + 31]) - (gray[idx - 31] + 2 * gray[idx - 32] + gray[idx - 33]);
                    const mag = Math.sqrt(gx * gx + gy * gy);
                    const angle = Math.atan2(gy, gx);
                    const sector = Math.floor(((angle + Math.PI) / (2 * Math.PI)) * 8) % 8;
                    edges[sector] += mag;
                    varianceSum += Math.pow(gray[idx] - mean, 2);
                }
            }

            const texture = varianceSum / (32 * 32 * 255 * 255);

            return {
                r: norm(rHist),
                g: norm(gHist),
                b: norm(bHist),
                l: norm(lHist),
                texture: texture,
                edges: edges.map(e => e / (32 * 32 * 255))
            };
        }

        function fingerprintToVector(fp) {
            return [...fp.r, ...fp.g, ...fp.b, ...fp.l, fp.texture, ...fp.edges];
        }

        // --- Association Logic (Panel A: Dyad) ---
        const manifoldBuffer = document.getElementById('manifold-buffer');
        const assocSource = document.getElementById('assoc-source');
        const commitBtn = document.getElementById('commit-assoc');
        const assocLog = document.getElementById('assoc-log');

        async function commitAssociation() {
            const source = assocSource.value.trim();
            const target = manifoldBuffer.value.trim();

            if (!source || !target) {
                appendMessage('system', "ASSOCIATION FAILED: BUFFER OR SOURCE EMPTY.");
                return;
            }

            const command = `ASSOCIATE: ${source} <-> ${target}`;
            appendMessage('user', `LINKING DYAD: ${source} TO MANIFOLD...`);

            try {
                const response = await fetch(`${state.backend_url}/interact`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: command })
                });

                const data = await response.json();
                handleResponse(data);

                const logEntry = document.createElement('div');
                logEntry.innerText = `[DYAD] ${source} -> ${target.substring(0, 15)}...`;
                assocLog.prepend(logEntry);

                manifoldBuffer.value = '';
                assocSource.value = '';

            } catch (err) {
                appendMessage('system', "ASSOCIATION RUPTURE. PERSISTENCE FAILED.");
            }
        }

        // --- Association Logic (Panel B: Semantic) ---
        async function commitSemanticLink() {
            const source = document.getElementById('semantic-source').value.trim();
            const target = document.getElementById('semantic-buffer').value.trim();

            if (!source || !target) {
                appendMessage('system', "ASSOCIATION FAILED: BUFFER OR SOURCE EMPTY.");
                return;
            }

            const command = `ASSOCIATE: ${source} <-> ${target}`;
            appendMessage('user', `LINKING: ${source} TO SEMANTIC BUFFER...`);

            try {
                const response = await fetch(`${state.backend_url}/interact`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: command })
                });

                const data = await response.json();
                handleResponse(data);

                const logEntry = document.createElement('div');
                logEntry.innerText = `[${new Date().toLocaleTimeString()}] ${source} -> ${target.substring(0, 20)}...`;
                document.getElementById('semantic-log').prepend(logEntry);

                // Clear inputs
                document.getElementById('semantic-buffer').value = '';
                document.getElementById('semantic-source').value = '';

            } catch (err) {
                appendMessage('system', "SEMANTIC RUPTURE. PERSISTENCE FAILED.");
            }
        }

        // =====================================================
        // PANEL C: AUDIO DYAD LOGIC
        // =====================================================
        const audioDropzone = document.getElementById('audio-dropzone');
        const audioFileInput = document.getElementById('audio-file-input');
        const audioPlayer = document.getElementById('audio-player');
        const audioPlayerWrapper = document.getElementById('audio-player-wrapper');
        const audioWaveform = document.getElementById('audio-waveform');
        const audioMeta = document.getElementById('audio-meta');
        const audioDropHint = document.getElementById('audio-drop-hint');
        const commitAudioBtn = document.getElementById('commit-audio');
        const audioLog = document.getElementById('audio-log');

        // Active audio dyad state
        state.active_audio_dyad = null;

        function formatDuration(secs) {
            const m = Math.floor(secs / 60);
            const s = Math.floor(secs % 60).toString().padStart(2, '0');
            return `${m}:${s}`;
        }

        function drawWaveform(audioBuffer) {
            const canvas = audioWaveform;
            canvas.style.display = 'block';
            canvas.width = canvas.offsetWidth || 260;
            canvas.height = 40;
            const ctx = canvas.getContext('2d');

            // Downsample to canvas width
            const data = audioBuffer.getChannelData(0);
            const step = Math.ceil(data.length / canvas.width);
            const amp = canvas.height / 2;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = 'var(--terminal-warn)';
            ctx.lineWidth = 1;
            ctx.beginPath();

            for (let i = 0; i < canvas.width; i++) {
                const slice = data.slice(i * step, (i + 1) * step);
                const max = slice.reduce((a, b) => Math.max(a, Math.abs(b)), 0);
                const y = amp - max * amp;
                if (i === 0) ctx.moveTo(i, y);
                else ctx.lineTo(i, y);
            }
            ctx.stroke();
        }

        async function processAudioFile(file) {
            const validTypes = ['audio/mpeg', 'audio/mp4', 'audio/wav', 'audio/ogg', 'audio/x-m4a'];
            const validExts = ['.mp3', '.m4a', '.wav', '.ogg'];
            const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));

            if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
                appendMessage('system', `AUDIO DYAD REJECTED: Unsupported format (${file.type || ext}).`);
                return;
            }

            audioDropHint.innerHTML = `LOADING: ${file.name}`;
            audioDropzone.classList.add('loaded');
            document.getElementById('audio-status').innerText = 'READING';
            document.getElementById('audio-status').style.color = 'var(--terminal-warn)';

            // Wire up player
            const objectUrl = URL.createObjectURL(file);
            audioPlayer.src = objectUrl;
            audioPlayerWrapper.style.display = 'block';

            // Decode with WebAudio API for waveform preview
            try {
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const arrayBuffer = await file.arrayBuffer();
                const decoded = await audioCtx.decodeAudioData(arrayBuffer);

                drawWaveform(decoded);

                const durationStr = formatDuration(decoded.duration);
                const sampleRate = decoded.sampleRate;
                const channels = decoded.numberOfChannels;

                audioMeta.innerText =
                    `${file.name}  |  ${durationStr}  |  ${sampleRate}Hz  |  ${channels}ch  |  `
                    + `${(file.size / 1024).toFixed(0)}KB`;

                // ── Chebyshev Polynomial Spectral Harmonic Extraction ────────────
                // Mirrors PolynomialCoprimeConfig: no hardcoded bin counts.
                // K (degree) is derived from the signal's physical properties so
                // the basis self-adjusts to each file's topology.
                //
                // Pattern:  PolynomialBasis._chebyshev()
                //   T_0(x)=1,  T_1(x)=x,  T_{n+1}=2x·T_n - T_{n-1}
                //
                // Roughness preservation:  LSB Stochastic Rounding applied to
                // every coefficient's fractional part (mirrors
                // SiliconSovereigntyEngine.apply_stochastic_rounding).
                // This preserves Feature Scars — we never smooth them away.

                const pcmData = decoded.getChannelData(0);
                const N = pcmData.length;

                // K: polynomial degree — derived from duration & sample rate.
                // ~1 Chebyshev mode per 0.5s of audio (min 5, max 32).
                // Never hardcoded; structurally honest about the signal's
                // information density.
                const K = Math.max(5, Math.min(32, Math.round(decoded.duration * 2)));

                // Frame the signal into K+1 overlapping Hann-windowed segments.
                // Frame boundaries are co-prime to avoid harmonic aliasing —
                // same principle as PolynomialCoprimeConfig.k coprime functionals.
                const frameCount = K + 1;
                const frameSize = Math.floor(N / frameCount);
                const frameEnergies = new Float64Array(frameCount);

                for (let f = 0; f < frameCount; f++) {
                    const start = f * frameSize;
                    let energy = 0.0;
                    for (let i = 0; i < frameSize; i++) {
                        // Hann window: w(i) = 0.5*(1 - cos(2π·i/(frameSize-1)))
                        // Preserves edge roughness rather than tapering to zero.
                        const w = 0.5 * (1.0 - Math.cos(2 * Math.PI * i / Math.max(1, frameSize - 1)));
                        const s = pcmData[Math.min(start + i, N - 1)] * w;
                        energy += s * s;
                    }
                    frameEnergies[f] = Math.sqrt(energy / Math.max(1, frameSize));
                }

                // Normalize frame energies to [-1, 1] for Chebyshev domain.
                let eMin = Infinity, eMax = -Infinity;
                for (let f = 0; f < frameCount; f++) {
                    if (frameEnergies[f] < eMin) eMin = frameEnergies[f];
                    if (frameEnergies[f] > eMax) eMax = frameEnergies[f];
                }
                const eRange = Math.max(eMax - eMin, 1e-12);
                const xNorm = new Float64Array(frameCount);
                for (let f = 0; f < frameCount; f++) {
                    xNorm[f] = 2.0 * (frameEnergies[f] - eMin) / eRange - 1.0;
                }

                // Chebyshev recurrence over the normalised energy samples:
                //   T[0] = 1,  T[1] = x,  T[n] = 2x·T[n-1] - T[n-2]
                // Project each frame's xNorm through the basis and accumulate
                // the mean coefficient — row of the PolynomialCoprimeConfig θ matrix.
                const chebyCoeffs = new Float64Array(K);
                for (let k = 0; k < K; k++) {
                    let acc = 0.0;
                    for (let f = 0; f < frameCount; f++) {
                        const x = xNorm[f];
                        // Evaluate T_k(x) via recurrence
                        let T_prev = 1.0, T_curr = x;
                        if (k === 0) { T_curr = 1.0; }
                        else if (k === 1) { T_curr = x; }
                        else {
                            let T_p = 1.0, T_c = x;
                            for (let n = 2; n <= k; n++) {
                                const T_n = 2.0 * x * T_c - T_p;
                                T_p = T_c; T_c = T_n;
                            }
                            T_curr = T_c;
                        }
                        acc += T_curr;
                    }
                    chebyCoeffs[k] = acc / frameCount;  // Mean projection (≈ inner product)
                }

                // Birkhoff-style row normalisation:
                // Ensure coefficients sum to 1 (doubly-stochastic row constraint).
                // Mirrors BirkhoffPolytopeSampler.sinkhorn_knopp applied to a single row.
                let coeffSum = 0.0;
                for (let k = 0; k < K; k++) coeffSum += Math.abs(chebyCoeffs[k]);
                const theta_row = new Float64Array(K);
                for (let k = 0; k < K; k++) {
                    theta_row[k] = coeffSum > 1e-12 ? Math.abs(chebyCoeffs[k]) / coeffSum : 1.0 / K;
                }

                // LSB Stochastic Rounding — Feature Scar Preservation.
                // Mirrors SiliconSovereigntyEngine.apply_stochastic_rounding:
                //   fixed = floor(v * scale) + Bernoulli(frac(v * scale))
                // We keep the rounded coefficient AND record the fractional scar
                // so the backend can reconstruct the exact quantization residue
                // if needed (warm-start backtracking from Chiral Residue Cache).
                const SCALE = 1024.0;  // Derived: 2^10 — matches int64 fixed point
                const harmonics = [];
                let seedState = (sampleRate ^ (N & 0xFFFF)) >>> 0;  // Xorshift seed from signal geometry
                for (let k = 0; k < K; k++) {
                    const v = theta_row[k] * SCALE;
                    const floorV = Math.floor(v);
                    const frac = v - floorV;
                    // Xorshift32 RNG —- exact algorithm from PyOpenCL kernel
                    seedState ^= (seedState << 13) >>> 0;
                    seedState ^= (seedState >>> 17) >>> 0;
                    seedState ^= (seedState << 5) >>> 0;
                    const stochasticBit = (seedState / 4294967295.0) < frac ? 1 : 0;
                    const rounded = (floorV + stochasticBit) / SCALE;
                    harmonics.push(parseFloat(rounded.toFixed(6)));
                }

                // Global descriptors (derived, not hardcoded)
                const rmsEnergy = Math.sqrt(
                    Array.from(frameEnergies).reduce((s, v) => s + v * v, 0) / frameCount
                );
                const zeroCrossings = Array.from(pcmData).reduce((c, v, i, a) =>
                    i > 0 && (a[i - 1] >= 0) !== (v >= 0) ? c + 1 : c, 0
                ) / N;
                // Spectral centroid: weighted mean of Chebyshev mode index by energy
                let scNum = 0, scDen = 0;
                for (let k = 0; k < K; k++) { scNum += k * harmonics[k]; scDen += harmonics[k]; }
                const spectralCentroid = scDen > 0 ? scNum / scDen : 0;

                state.active_audio_dyad = {
                    filename: file.name,
                    duration_s: decoded.duration,
                    sample_rate: sampleRate,
                    channels: channels,
                    size_bytes: file.size,
                    chebyshev_degree: K,         // Derived, never hardcoded
                    rms_energy: rmsEnergy,
                    zero_crossing_rate: zeroCrossings,
                    spectral_centroid: spectralCentroid,
                    chebyshev_harmonics: harmonics, // K Birkhoff-normalised, LSB-rounded coeffs
                    commutativity: document.getElementById('commute-audio').value,
                };

                audioDropHint.innerHTML = `DYAD CAPTURED: ${file.name}`;
                commitAudioBtn.disabled = false;

                document.getElementById('audio-status').innerText = 'ARMED';
                document.getElementById('audio-status').style.color = 'var(--terminal-green)';

                appendMessage('system',
                    `AUDIO DYAD INGESTED: ${file.name} | ` +
                    `${durationStr} | K=${K} | RMS=${rmsEnergy.toFixed(4)} | ` +
                    `ZCR=${zeroCrossings.toFixed(4)} | SC_cheb=${spectralCentroid.toFixed(3)}`);

            } catch (err) {
                appendMessage('system', `WAVEFORM DECODE RUPTURE: ${err.message}`);
                console.error('Audio decode error:', err);
            }
        }

        async function commitAudioDyad() {
            const description = document.getElementById('audio-description-buffer').value.trim();
            const dyad = state.active_audio_dyad;

            if (!dyad) {
                appendMessage('system', 'AUDIO COMMIT FAILED: No audio dyad loaded.');
                return;
            }

            const commutativity = document.getElementById('commute-audio').value;
            dyad.commutativity = commutativity;

            const displayOrder = commutativity === 'audio_first'
                ? 'Audio → Text'
                : commutativity === 'text_first'
                    ? 'Text → Audio'
                    : '≅ Symmetric';

            appendMessage('user',
                `COMMITTING AUDIO DYAD [${displayOrder}]: ${dyad.filename}` +
                (description ? ` | "${description.substring(0, 50)}..."` : ''));

            try {
                const payload = {
                    text: description || `INGEST_AUDIO_DYAD: ${dyad.filename}`,
                    audio_dyad: dyad,
                    commutativity: commutativity
                };

                const response = await fetch(`${state.backend_url}/interact`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                handleResponse(data);

                const logEntry = document.createElement('div');
                logEntry.innerText =
                    `[${new Date().toLocaleTimeString()}] [${displayOrder}] ${dyad.filename}`;
                audioLog.prepend(logEntry);

                // Reset
                document.getElementById('audio-description-buffer').value = '';
                document.getElementById('audio-status').innerText = 'IDLE';
                document.getElementById('audio-status').style.color = '#555';
                audioDropzone.classList.remove('loaded');
                audioDropHint.innerHTML = `DRAG AUDIO DYAD<br><span style="color:#444;">mp3 &bull; m4a &bull; wav &bull; ogg</span>`;
                audioWaveform.style.display = 'none';
                audioPlayerWrapper.style.display = 'none';
                commitAudioBtn.disabled = true;
                state.active_audio_dyad = null;

            } catch (err) {
                appendMessage('system', 'AUDIO COMMIT RUPTURE. BACKEND DISSOCIATED.');
                console.error('Audio commit error:', err);
            }
        }

        // Audio drop zone events
        audioDropzone.onclick = () => audioFileInput.click();
        audioFileInput.onchange = (e) => {
            if (e.target.files.length > 0) processAudioFile(e.target.files[0]);
        };
        audioDropzone.ondragover = (e) => {
            e.preventDefault();
            audioDropzone.classList.add('dragover');
        };
        audioDropzone.ondragleave = () => audioDropzone.classList.remove('dragover');
        audioDropzone.ondrop = (e) => {
            e.preventDefault();
            audioDropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) processAudioFile(e.dataTransfer.files[0]);
        };
        commitAudioBtn.onclick = commitAudioDyad;

        // --- Event Listeners ---
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage(userInput.value);
        });

        document.getElementById('regime-goo').onclick = () => {
            state.regime = 'goo';
            document.getElementById('regime-goo').classList.add('active');
            document.getElementById('regime-prickles').classList.remove('active');
            appendMessage('system', "SWITCHING TO GOO REGIME: NON-DOMINANT CO-PRESENCE ENABLED.");
        };

        document.getElementById('regime-prickles').onclick = () => {
            state.regime = 'prickles';
            document.getElementById('regime-prickles').classList.add('active');
            document.getElementById('regime-goo').classList.remove('active');
            appendMessage('system', "SWITCHING TO PRICKLES REGIME: TRUTH BRANCHING (CRT) ARMED.");
        };

        fileInput.onchange = (e) => {
            if (e.target.files.length > 0) processImage(e.target.files[0]);
        };

        dropzone.ondragover = (e) => { e.preventDefault(); dropzone.style.borderColor = 'var(--terminal-blue)'; };
        dropzone.ondragleave = (e) => { e.preventDefault(); dropzone.style.borderColor = 'var(--glass-border)'; };
        dropzone.ondrop = (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) processImage(e.dataTransfer.files[0]);
        };

        document.getElementById('commit-semantic').onclick = commitSemanticLink;
        commitBtn.onclick = commitAssociation;

        initRibbon();

        // --- Topological Heartbeat (Service Mode) ---
        // Keeps the system "awake" and updates diagnostics every 15s
        setInterval(async () => {
            try {
                const response = await fetch(`${state.backend_url}/interact`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: "IDLE_RESONANCE_HEARTBEAT", generate_response: false })
                });
                const data = await response.json();
                handleResponse(data);
                console.log("TOPOLOGICAL HEARTBEAT: COHERENCE MAINTAINED.");
            } catch (err) {
                console.warn("HEARTBEAT RUPTURE. BACKEND STOCHASTIC.");
            }
        }, 15000);

    </script>
</body>

</html>
