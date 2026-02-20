#!/usr/bin/env python3
"""
Conversational API GUI Launcher

Simple launcher for the conversational API integration GUI.
Handles dependencies and provides fallback options.

Author: William Matthew Bryant
Created: January 2026
"""

import sys
import os
import subprocess

def check_tkinter():
    """Check if tkinter is available."""
    try:
        import tkinter
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages."""
    required_packages = [
        'torch',
        'numpy', 
        'requests',
        'datasets',
        'transformers'
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
    
    print("Package installation complete")

def main():
    """Main launcher."""
    print("Conversational API GUI Launcher")
    print("=" * 40)
    
    # Check tkinter
    if not check_tkinter():
        print("ERROR: tkinter not available")
        print("   On Ubuntu/Debian: sudo apt-get install python3-tk")
        print("   On CentOS/RHEL: sudo yum install tkinter")
        print("   On macOS: tkinter should be included with Python")
        return False
    
    print("SUCCESS: tkinter available")
    
    # Check if we should install requirements
    try:
        import torch
        import requests
        print("SUCCESS: Core packages available")
    except ImportError:
        print("INFO: Some packages missing, installing...")
        install_requirements()
    
    # Launch GUI
    try:
        print("INFO: Launching GUI...")
        
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # Import and run GUI
        from src.ui.conversational_gui import ConversationalGUI
        
        app = ConversationalGUI()
        app.run()
        
        return True
        
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        print("   Try running: python -m pip install torch numpy requests datasets")
        return False
    except Exception as e:
        print(f"ERROR: GUI error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print(f"\nAlternative: Run the fallback demo instead:")
        print(f"   python examples/fallback_conversational_demo.py")
        
        # Ask if user wants to run fallback
        try:
            response = input("\nRun fallback demo? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print("Running fallback demo...")
                os.system("python examples/fallback_conversational_demo.py")
        except KeyboardInterrupt:
            print("\nGoodbye!")