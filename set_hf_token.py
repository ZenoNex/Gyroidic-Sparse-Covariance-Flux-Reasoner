#!/usr/bin/env python3
"""
Set HF Token in Environment

Simple script to set your HF token in the environment for testing.
"""

import os

def main():
    """Set HF token in environment."""
    print("Set HF Token in Environment")
    print("=" * 30)
    
    token = input("Enter your HF token: ").strip()
    
    if not token:
        print("No token provided!")
        return
    
    if not token.startswith('hf_'):
        print("Warning: Token should start with 'hf_'")
        proceed = input("Continue anyway? (y/n): ").lower().strip()
        if proceed not in ['y', 'yes']:
            return
    
    # Set environment variables
    os.environ['HF_TOKEN'] = token
    os.environ['HUGGING_FACE_HUB_TOKEN'] = token
    
    print(f"✓ Token set: {token[:10]}...{token[-5:]}")
    print("Environment variables set:")
    print("  - HF_TOKEN")
    print("  - HUGGING_FACE_HUB_TOKEN")
    
    print(f"\nYou can now run:")
    print(f"  python test_cpu_only_ingestion.py")
    print(f"  python launch_conversational_gui.py")

if __name__ == "__main__":
    main()