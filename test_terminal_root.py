#!/usr/bin/env python3
"""
Test if the root path serves the diegetic terminal interface.
"""

import requests

def test_terminal_root():
    """Test if root path serves the diegetic terminal."""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            content = response.text
            print(f"✅ Root path accessible, content length: {len(content)}")
            
            # Check if it's the diegetic terminal (not file directory)
            if "GYROIDIC DIEGETIC TERMINAL" in content:
                print("✅ Diegetic terminal interface served successfully!")
                return True
            elif "Index of" in content or ".py" in content[:500]:
                print("❌ Still serving file directory instead of terminal")
                print(f"First 200 chars: {content[:200]}")
                return False
            else:
                print("⚠️  Unknown content served")
                print(f"First 200 chars: {content[:200]}")
                return False
        else:
            print(f"❌ Root path returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error accessing root path: {e}")
        return False

if __name__ == "__main__":
    test_terminal_root()