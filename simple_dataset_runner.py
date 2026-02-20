#!/usr/bin/env python3
"""
Simple Dataset Runner - Works around import issues.

Usage:
    python simple_dataset_runner.py
"""

import sys
import os
import subprocess

def main():
    """Simple dataset training interface."""
    print("🚀 Simple Gyroidic Dataset Runner")
    print("=" * 40)
    
    print("Available options:")
    print("1. Test multimodal system")
    print("2. Run temporal training")
    print("3. Test Mandelbulb augmentation")
    print("4. Start web interface")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                print("\n🎨 Testing multimodal system...")
                os.system("python test_image_simple.py")
                
            elif choice == '2':
                print("\n🧠 Running temporal training...")
                os.system("python examples/enhanced_temporal_training.py")
                
            elif choice == '3':
                print("\n🌀 Testing Mandelbulb augmentation...")
                os.system("python test_mandelbulb_simple.py")
                
            elif choice == '4':
                print("\n🌐 Starting web interface...")
                print("This will start the backend server.")
                print("Open http://localhost:8000 in your browser.")
                print("Press Ctrl+C to stop the server.")
                os.system("python src/ui/diegetic_backend.py")
                
            elif choice == '5':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()