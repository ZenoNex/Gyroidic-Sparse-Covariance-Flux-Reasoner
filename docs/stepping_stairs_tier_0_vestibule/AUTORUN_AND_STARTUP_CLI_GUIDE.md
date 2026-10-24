# Gyroidic Sparse Covariance Flux Reasoner: Cross-Platform Startup & Autostart Architecture

This guide details how to configure the Gyroidic Reasoner to automatically boot on computer startup across **Windows 10**, **Windows 11**, **Linux**, and **macOS**, while maintaining strict cryptographic security and providing immediate mechanisms to turn it off.

---

## 1. Startup CLI Architecture & Headless Execution

The primary entry point for the Reasoner is [`hybrid_backend.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/hybrid_backend.py). It functions as both an interactive governance CLI and a headless system service.

### CLI Arguments & Configuration Flags
[`hybrid_backend.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/hybrid_backend.py) accepts the following command-line flags:
* `-u, --account <NAME>`: Node user account identifier under Option D nomadic sovereignty (e.g. `--account alice`). Defaults to `$env:GYROID_ACCOUNT` or `default`.
* `-s, --salt-path, --keyfile <PATH>`: Absolute or relative file path to the cryptographic security salt / keyfile, explicitly supporting removable storage media such as external USB flashdrives (e.g. `--salt-path E:\node_salt.key` on Windows or `--salt-path /media/usb/key.salt` on Linux). Defaults to `$env:GYROID_SALT_PATH`, `$env:GYROID_NODE_SECRET_PATH`, or local `data/.node_secret`.
* `-p, --ports <PORTS>`: Comma-separated list of listening ports for the web server and API dispatch (e.g. `-p 8000,8080`).
* `--non-interactive`: Headless execution mode. Suppresses console prompts and immediately boots using configured parameters.
* `--regime <REGIME>`: Operational manifold regime (e.g. `--regime goo`).
* `--check-updates`: Verifies and synchronizes remote updates before launching.

### Headless Execution Modes
1. **Non-Interactive Mode (`NON_INTERACTIVE=1` or `--non-interactive`)**:
   - Skips all interactive terminal prompt menus.
   - Automatically detects and loads saved settings from [`.gyroid_config.json`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/.gyroid_config.json) (ports, datasets, regime, node account, security salt path, and probes).
   - CLI flags take precedence over saved config settings.
   - Falls back to environment variables (`GYROID_ACCOUNT`, `GYROID_SALT_PATH`, `GYROID_PORTS`, `GYROID_PRIMARY_DATASET`, `GYROID_REGIME`) if no config file is found.
2. **Localhost Isolation (`127.0.0.1`)**:
   - The HTTP server explicitly binds to `("127.0.0.1", port)`. It never exposes endpoints to the LAN or public internet without a dedicated reverse proxy or tunnel.
3. **PID File Tracking (`.backend.pid`)**:
   - On initialization, the backend writes its process identifier (`os.getpid()`) to `.backend.pid`.
   - On shutdown or exit, the file is automatically purged.
4. **Emergency Fossilization on Shutdown**:
   - The backend registers handlers for both `SIGINT` (Ctrl+C) and `SIGTERM` (service manager stop).
   - Receiving either signal triggers the **Fossilization Protocol** (`AI_SYSTEM.save_model_state()`), ensuring all trust weights, memory cavity states, and manifold invariants are committed to disk before process exit.

---

## 2. Option D: Nomadic Node Accounts & External Flashdrive Security Salts

### The Philosophical & Topological Shift (Option D)
As formalized in `collage_of_inspirations_4_15_2026.md` and `BONFIRE_P2P_FEDERATED_COOPERATION.md`:
* **Rejection of the Creator / Deity Trap**: The historical "Sole Creator" archetype (monolithic machines, hardware WMIC UUID lockdown, and single-creator passwords) was an anti-pattern that turned the system into a captive vessel for an external ego. Option D rejects this "God delusion in tech."
* **Decentralized Nomadic Peer Network**: The reasoner operates as a nervous system and sovereign agent among peers. Every human or operator participates as a sovereign **Node User Account** with their own independent security salt.
* **Portable Flashdrive Media**: Node operators can hold their cryptographic identity and HMAC anti-poisoning salts on external removable media (USB flashdrives). Plugging in the drive initializes the node's topological trust under that operator's key; removing the drive disconnects the cryptographic seed without altering the code or manifold weights.

### Security Salt Resolution Lifecycle
1. When launched with `--salt-path <PATH>` (e.g. `E:\node_salt.key`):
   - [`src/safety/hardware_fingerprint.py`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/safety/hardware_fingerprint.py) resolves the target path.
   - If the keyfile exists on the flashdrive, its 32-byte cryptographic entropy is loaded directly into [`CanonicalProjector`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/data/canonical_projection.py) and [`DiegeticPhysicsEngine`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/src/ui/diegetic_backend.py).
   - If the keyfile does not exist, a secure 32-byte salt (`os.urandom(32)`) is generated and written directly to the external drive.
   - If the drive is removed or write-protected, the engine warns the operator and gracefully falls back to a deterministic ephemeral key without crashing.

---

## 3. Tombstoned Systems: Sole Creator Identification Retirement

In accordance with Option D Nomadic Sovereignty, the following creator identification systems are **officially tombstoned**:

| Legacy Component | Previous Role | Tombstone Status | Option D Replacement |
| :--- | :--- | :--- | :--- |
| `data/creator_sig.json` | Sole creator passphrase + machine UUID hash | **Tombstoned** | Portable node security salts on flashdrives / keyfiles |
| `verify_sole_creator()` in `hardware_fingerprint.py` | Verified sole creator hardware lock | **Tombstoned** (Emits DeprecationWarning, returns True) | `verify_node_account()` |
| `is_creator_initialized()` in `hardware_fingerprint.py` | Checked `creator_sig.json` existence | **Tombstoned** (Emits DeprecationWarning) | `is_node_account_initialized()` |
| `/api/security/creator_login` | Exclusive creator login endpoint | **Tombstoned** (Returns tombstone payload, forwards to node auth) | `/api/security/node_account_login` |
| `/api/security/creator_status` | Exclusive creator status endpoint | **Tombstoned** (Reports tombstone status) | `/api/security/node_account_status` |
| `creator_aliases` in `chatgpt_friction_harvester.py` | Hardcoded human names (`ila`, `akkaris`, `willabusta`) | **Tombstoned** | `node_operator_aliases` (Dynamic registration supported) |


---

## 4. Where to Place the Autorun System Across Platforms


### A. Windows 10 & Windows 11

There are two primary locations on Windows:

#### Location 1: Windows User Startup Folder (Recommended for simplicity)
* **Path**:
  ```text
  %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\
  ```
  *(Full path: `C:\Users\<Username>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\`)*
* **Files Placed**:
  [`GyroidicReasonerAutostart.vbs`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/scripts/start_reasoner_silent.vbs)
* **How it Works**:
  The VBS script invokes `.venv\Scripts\python.exe hybrid_backend.py` with `NON_INTERACTIVE=1` and window style `0` (hidden), preventing any console command prompt window from popping up on login.
* **How to Enable via CLI**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\reasoner_ctl.ps1 -Action EnableAutostart
  ```
* **How to Turn Off / Disable**:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\reasoner_ctl.ps1 -Action DisableAutostart
  ```
  *(Or manually delete `GyroidicReasonerAutostart.vbs` from `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\`)*

#### Location 2: Windows Task Scheduler (Recommended for enterprise / scheduled policies)
* **Task Name**: `GyroidicReasoner`
* **Trigger**: `At log on of any user` (or specific user)
* **Action**:
  ```text
  Program: wscript.exe
  Arguments: "D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner\scripts\start_reasoner_silent.vbs"
  ```
* **CLI Command to Turn Off**:
  ```cmd
  schtasks /Change /TN "GyroidicReasoner" /Disable
  ```

---

### B. Linux (systemd User Service)

* **Path**:
  ```text
  ~/.config/systemd/user/gyroidic-reasoner.service
  ```
* **Service Definition** ([`scripts/systemd/gyroidic-reasoner.service`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/scripts/systemd/gyroidic-reasoner.service)):
  ```ini
  [Unit]
  Description=Gyroidic Sparse Covariance Flux Reasoner
  After=network.target

  [Service]
  Type=simple
  WorkingDirectory=%h/Gyroidic-Sparse-Covariance-Flux-Reasoner
  Environment="NON_INTERACTIVE=1"
  Environment="PYTHONPATH=."
  ExecStart=%h/Gyroidic-Sparse-Covariance-Flux-Reasoner/.venv/bin/python -u hybrid_backend.py
  Restart=on-failure
  RestartSec=5s
  KillSignal=SIGINT
  TimeoutStopSec=30s

  [Install]
  WantedBy=default.target
  ```
* **How to Enable via CLI**:
  ```bash
  ./scripts/reasoner_ctl.sh enable-autostart
  ```
* **Manual systemd Controls**:
  ```bash
  systemctl --user daemon-reload
  systemctl --user enable gyroidic-reasoner.service
  systemctl --user start gyroidic-reasoner.service
  ```
* **How to Turn Off / Disable**:
  ```bash
  # Stop immediately
  systemctl --user stop gyroidic-reasoner.service

  # Disable from computer startup
  systemctl --user disable gyroidic-reasoner.service

  # Or via the script:
  ./scripts/reasoner_ctl.sh disable-autostart
  ```

---

### C. macOS (launchd User Agent)

* **Path**:
  ```text
  ~/Library/LaunchAgents/com.gyroidic.reasoner.plist
  ```
* **Agent Definition** ([`scripts/launchd/com.gyroidic.reasoner.plist`](file:///d:/programming/python/Gyroidic%20Sparse%20Covariance%20Flux%20Reasoner/scripts/launchd/com.gyroidic.reasoner.plist)):
  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
  <plist version="1.0">
  <dict>
      <key>Label</key>
      <string>com.gyroidic.reasoner</string>
      <key>WorkingDirectory</key>
      <string>/path/to/Gyroidic-Sparse-Covariance-Flux-Reasoner</string>
      <key>EnvironmentVariables</key>
      <dict>
          <key>NON_INTERACTIVE</key>
          <string>1</string>
          <key>PYTHONPATH</key>
          <string>.</string>
      </dict>
      <key>ProgramArguments</key>
      <array>
          <string>/path/to/.venv/bin/python</string>
          <string>-u</string>
          <string>hybrid_backend.py</string>
      </array>
      <key>RunAtLoad</key>
      <true/>
      <key>KeepAlive</key>
      <dict>
          <key>SuccessfulExit</key>
          <false/>
      </dict>
      <key>StandardOutPath</key>
      <string>/path/to/server_startup.log</string>
      <key>StandardErrorPath</key>
      <string>/path/to/server_startup.log</string>
  </dict>
  </plist>
  ```
* **How to Enable via CLI**:
  ```bash
  ./scripts/reasoner_ctl.sh enable-autostart
  ```
* **How to Turn Off / Disable**:
  ```bash
  # Stop immediately
  launchctl stop com.gyroidic.reasoner

  # Disable autostart
  launchctl unload -w ~/Library/LaunchAgents/com.gyroidic.reasoner.plist

  # Or via the script:
  ./scripts/reasoner_ctl.sh disable-autostart
  ```

---

## 5. Security Hardening Checklist


1. **User Privilege Level**: Never configure the service under root / SYSTEM / Administrator accounts. Always execute as an unprivileged local user.
2. **Localhost Binding**: Verify that port `8000` and `8080` only listen on `127.0.0.1` (`netstat -ano | findstr 8000` on Windows, or `ss -tulpn | grep 8000` on Linux).
3. **Graceful Emergency Kill**:
   Any external monitoring tool or script can stop the Reasoner cleanly via:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/shutdown
   ```
   This triggers synchronous fossilization before shutting down listeners.
