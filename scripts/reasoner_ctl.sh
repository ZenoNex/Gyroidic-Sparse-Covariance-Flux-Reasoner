#!/usr/bin/env bash
# ==============================================================================
# Gyroidic Sparse Covariance Flux Reasoner - Universal Service Controller
# Supported OS: Linux, macOS, WSL / Git Bash (Windows 10 / 11)
# ==============================================================================

set -e

# Resolve repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${REPO_ROOT}/.backend.pid"
LOG_FILE="${REPO_ROOT}/server_startup.log"

# Select Python binary
if [ -f "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif [ -f "${REPO_ROOT}/.venv/Scripts/python.exe" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/Scripts/python.exe"
else
    PYTHON_BIN="python3"
fi

get_pid() {
    if [ -f "${PID_FILE}" ]; then
        cat "${PID_FILE}"
    else
        echo ""
    fi
}

is_running() {
    local pid
    pid=$(get_pid)
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        return 0
    fi
    # Fallback: check HTTP ping
    if curl -s --connect-timeout 1 http://127.0.0.1:8000/ping >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

start_reasoner() {
    if is_running; then
        echo "[STATUS] Reasoner is already running (PID: $(get_pid))."
        exit 0
    fi

    echo "[START] Launching Gyroidic Flux Reasoner in background..."
    cd "${REPO_ROOT}"
    export NON_INTERACTIVE=1
    export PYTHONPATH="."

    nohup "${PYTHON_BIN}" -u hybrid_backend.py >> "${LOG_FILE}" 2>&1 &
    local new_pid=$!
    echo "${new_pid}" > "${PID_FILE}"

    echo "[START] Process started with PID: ${new_pid}."
    echo "[START] Logs streaming to: ${LOG_FILE}"
    echo "[START] Waiting for HTTP readiness on http://127.0.0.1:8000..."

    local count=0
    while [ ${count} -lt 30 ]; do
        if curl -s --connect-timeout 1 http://127.0.0.1:8000/ping >/dev/null 2>&1; then
            echo "[OK] Reasoner online and ready at http://127.0.0.1:8000"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    echo "[WARN] Server started (PID: ${new_pid}), but has not yet answered HTTP ping. Check logs:"
    echo "       tail -n 30 \"${LOG_FILE}\""
}

stop_reasoner() {
    local pid
    pid=$(get_pid)

    echo "[STOP] Initiating graceful shutdown for Gyroidic Reasoner..."

    # Method 1: Send API shutdown request (triggers Fossilization Protocol)
    if curl -s -X POST --connect-timeout 2 http://127.0.0.1:8000/api/shutdown >/dev/null 2>&1; then
        echo "[STOP] API shutdown signal acknowledged. Awaiting clean fossilization save..."
    elif [ -n "${pid}" ]; then
        # Method 2: Send SIGINT directly to trigger KeyboardInterrupt fossilization handler
        echo "[STOP] Dispatching SIGINT to PID ${pid} to enforce manifold fossilization..."
        kill -INT "${pid}" 2>/dev/null || true
    fi

    local count=0
    while [ ${count} -lt 20 ]; do
        if ! is_running; then
            echo "[OK] Reasoner stopped cleanly."
            rm -f "${PID_FILE}"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done

    # Method 3: Hard kill if hung
    pid=$(get_pid)
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        echo "[WARN] Server did not exit in 20s. Forcing SIGTERM..."
        kill -TERM "${pid}" 2>/dev/null || true
        sleep 2
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[WARN] Forcing SIGKILL..."
            kill -9 "${pid}" 2>/dev/null || true
        fi
    fi
    rm -f "${PID_FILE}"
    echo "[STOP] Stopped."
}

status_reasoner() {
    local pid
    pid=$(get_pid)
    if is_running; then
        echo "[STATUS] ONLINE"
        echo "   PID:  ${pid:-unknown}"
        echo "   HTTP: http://127.0.0.1:8000 (UI / API)"
        echo "   Ping: $(curl -s --connect-timeout 2 http://127.0.0.1:8000/ping || echo 'No ping response')"
    else
        echo "[STATUS] OFFLINE"
        if [ -f "${PID_FILE}" ]; then
            echo "   Stale PID file detected: ${PID_FILE} (PID ${pid} is not running)"
        fi
    fi
}

enable_autostart() {
    local uname_out
    uname_out="$(uname -s)"

    case "${uname_out}" in
        Linux*)
            local service_dir="${HOME}/.config/systemd/user"
            mkdir -p "${service_dir}"
            local service_file="${service_dir}/gyroidic-reasoner.service"

            cat > "${service_file}" <<EOF
[Unit]
Description=Gyroidic Sparse Covariance Flux Reasoner
After=network.target

[Service]
Type=simple
WorkingDirectory=${REPO_ROOT}
Environment="NON_INTERACTIVE=1"
Environment="PYTHONPATH=."
ExecStart=${PYTHON_BIN} -u hybrid_backend.py
Restart=on-failure
RestartSec=5s
KillSignal=SIGINT
TimeoutStopSec=30s

[Install]
WantedBy=default.target
EOF
            systemctl --user daemon-reload
            systemctl --user enable gyroidic-reasoner.service
            echo "[OK] systemd user service installed and enabled:"
            echo "     ${service_file}"
            echo "     Start now:   systemctl --user start gyroidic-reasoner.service"
            echo "     Disable:     ${0} disable-autostart"
            ;;

        Darwin*)
            local agent_dir="${HOME}/Library/LaunchAgents"
            mkdir -p "${agent_dir}"
            local plist_file="${agent_dir}/com.gyroidic.reasoner.plist"

            cat > "${plist_file}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.gyroidic.reasoner</string>
    <key>WorkingDirectory</key>
    <string>${REPO_ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>NON_INTERACTIVE</key>
        <string>1</string>
        <key>PYTHONPATH</key>
        <string>.</string>
    </dict>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
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
    <string>${LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_FILE}</string>
</dict>
</plist>
EOF
            launchctl load -w "${plist_file}"
            echo "[OK] macOS launchd LaunchAgent installed and loaded:"
            echo "     ${plist_file}"
            echo "     Disable: ${0} disable-autostart"
            ;;

        *)
            echo "[INFO] Running on Windows/WSL. For native Windows 10/11 autostart, run:"
            echo "       powershell -ExecutionPolicy Bypass -File scripts/reasoner_ctl.ps1 -EnableAutostart"
            ;;
    esac
}

disable_autostart() {
    local uname_out
    uname_out="$(uname -s)"

    case "${uname_out}" in
        Linux*)
            systemctl --user disable gyroidic-reasoner.service 2>/dev/null || true
            systemctl --user stop gyroidic-reasoner.service 2>/dev/null || true
            rm -f "${HOME}/.config/systemd/user/gyroidic-reasoner.service"
            systemctl --user daemon-reload
            echo "[OK] Linux systemd service removed and disabled."
            ;;

        Darwin*)
            local plist_file="${HOME}/Library/LaunchAgents/com.gyroidic.reasoner.plist"
            launchctl unload -w "${plist_file}" 2>/dev/null || true
            rm -f "${plist_file}"
            echo "[OK] macOS LaunchAgent removed and disabled."
            ;;

        *)
            echo "[INFO] For native Windows 10/11 disable, run:"
            echo "       powershell -ExecutionPolicy Bypass -File scripts/reasoner_ctl.ps1 -DisableAutostart"
            ;;
    esac
}

case "${1}" in
    start)
        start_reasoner
        ;;
    stop)
        stop_reasoner
        ;;
    restart)
        stop_reasoner
        sleep 2
        start_reasoner
        ;;
    status)
        status_reasoner
        ;;
    enable-autostart)
        enable_autostart
        ;;
    disable-autostart)
        disable_autostart
        ;;
    *)
        echo "Usage: ${0} {start|stop|restart|status|enable-autostart|disable-autostart}"
        exit 1
        ;;
esac
