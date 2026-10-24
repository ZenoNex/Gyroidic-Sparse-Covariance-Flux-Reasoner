#!/usr/bin/env python3
"""
Helper script to open the diegetic terminal correctly.
"""

import webbrowser
import requests
import time

def open_terminal():
    """Open the diegetic terminal in the default browser."""
    print(" Opening Diegetic Terminal Interface")
    print("=" * 40)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/ping", timeout=3)
        if response.status_code == 200:
            print("[OK] Backend is running")
            
            # Open the terminal in the default browser
            terminal_url = "http://localhost:8000"
            print(f"[START] Opening terminal at: {terminal_url}")
            
            webbrowser.open(terminal_url)
            
            print("\n Terminal Interface Features:")
            print("   • Chat with Gyroidic AI")
            print("   • Image-to-text associations")
            print("   • Text-to-text associations")
            print("   • Wikipedia knowledge integration")
            print("\n[WARN]  IMPORTANT: Always access via http://localhost:8000")
            print("   Do NOT open the HTML file directly (causes CORS errors)")
            
        else:
            print(f"[ERR] Backend not responding (status: {response.status_code})")
            print(" Start the backend first with: python hybrid_backend.py")
            
    except requests.exceptions.ConnectionError:
        print("[ERR] Backend not running")
        print(" Start the backend first with: python hybrid_backend.py")
        print("   Then run this script again")
        
    except Exception as e:
        print(f"[ERR] Error checking backend: {e}")

if __name__ == "__main__":
    open_terminal()