# How to Access the Diegetic Terminal Interface

## [ERR] WRONG WAY (Causes CORS Errors)
Do NOT open the HTML file directly in your browser:
```
file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/diegetic_terminal.html
```

This causes the error:
```
Access to fetch at 'file:///D:/interact' from origin 'null' has been blocked by CORS policy
```

## [OK] CORRECT WAY

### Step 1: Start the Backend
```powershell
# Directly via Python virtual environment:
.venv\scripts\python.exe hybrid_backend.py

# OR in the background via service controller:
.\scripts\reasoner_ctl.ps1 -Action Start
```

Wait for this message:
```
[OK] Hybrid backend running at http://localhost:8000
```

### Step 2: Access via HTTP Server
Open your browser and go to:
```
http://localhost:8000
```

### Step 3: Use the Terminal
- Chat in the main area
- Use the side panels for associations
- Everything will work correctly!

*Note: Running `hybrid_backend.py` and accessing the UI via the browser is the preferred system architecture. However, individual Python scripts like `src/ui/diegetic_terminal.py` are preserved for user choice.*

## [START] Quick Start Script
Run this to automatically start the backend and open the terminal:
```powershell
.venv\scripts\python.exe open_terminal.py
```

## Why This Happens
- The HTML file contains JavaScript that makes HTTP requests to `/interact`, `/associate`, etc.
- When opened as `file://`, these become `file:///D:/interact` which browsers block for security
- When served via HTTP server, these become `http://localhost:8000/interact` which works correctly

## Troubleshooting
If you get connection errors:
1. Make sure `python hybrid_backend.py` is running
2. Check that you see "[OK] Hybrid backend running at http://localhost:8000"
3. Access via `http://localhost:8000` (not file://)
4. Check browser console for any remaining errors