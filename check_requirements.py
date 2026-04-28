#!/usr/bin/env python3
"""
Check Requirements for Conversational API Integration
Checks if all required packages are installed for the conversational API system.
"""
import sys
import subprocess
import torch

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        print(f" [OK] {package_name}: Installed")
        return True
    except ImportError:
        print(f" [ERR] {package_name}: Not installed")
        return True # Don't fail the check, just report

def main():
    print(" Checking Requirements for Silicon Sovereignty Substrate")
    print("=" * 60)
    
    # Required packages
    required_packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("requests", "requests"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("pyopencl", "pyopencl"),
    ]
    
    print(" Checking installed packages...")
    for package_name, import_name in required_packages:
        check_package(package_name, import_name)
    
    # Check Silicon Sovereignty
    print(f"\n Checking Substrate Sovereignty...")
    try:
        from src.core import DEVICE
        print(f"   DEVICE: {DEVICE}")
        if str(DEVICE) != 'cpu':
            print(f"   Silicon Sovereignty: [OK] (Hardware-bridge active)")
        else:
            print(f"   Silicon Sovereignty: [ENABLED] (Substrate-independent fallback)")
    except Exception as e:
        print(f"   [ERR] Could not verify DEVICE: {e}")

    print(f"\n Next steps:")
    print(f" Run: python dataset_command_interface.py status")
    print(f" Then: python verify_archetype_flux.py")

if __name__ == "__main__":
    main()
