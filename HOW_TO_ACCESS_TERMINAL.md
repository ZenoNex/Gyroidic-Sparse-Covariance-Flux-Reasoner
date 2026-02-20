# How to Access the Diegetic Terminal Interface

## ❌ WRONG WAY (Causes CORS Errors)
Do NOT open the HTML file directly in your browser:
```
file:///D:/programming/python/Gyroidic Sparse Covariance Flux Reasoner/src/ui/diegetic_terminal.html
```

This causes the error:
```
Access to fetch at 'file:///D:/interact' from origin 'null' has been blocked by CORS policy
```

## ✅ CORRECT WAY

### Step 1: Start the Backend
```bash
python hybrid_backend.py
```

Wait for this message:
```
✅ Hybrid backend running at http://localhost:8000
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

## 🚀 Quick Start Script
Run this to automatically open the terminal:
```bash
python open_terminal.py
```

## Why This Happens
- The HTML file contains JavaScript that makes HTTP requests to `/interact`, `/associate`, etc.
- When opened as `file://`, these become `file:///D:/interact` which browsers block for security
- When served via HTTP server, these become `http://localhost:8000/interact` which works correctly

## Troubleshooting
If you get connection errors:
1. Make sure `python hybrid_backend.py` is running
2. Check that you see "✅ Hybrid backend running at http://localhost:8000"
3. Access via `http://localhost:8000` (not file://)
4. Check browser console for any remaining errors