#!/usr/bin/env python3
"""
Debug what content is actually being served by the terminal interface.
"""

import requests

def debug_terminal_content():
    """Debug the terminal interface content."""
    print("🔍 Debugging Terminal Interface Content")
    print("=" * 40)
    
    try:
        response = requests.get("http://localhost:8000", timeout=5)
        if response.status_code == 200:
            html_content = response.text
            print(f"📄 HTML content length: {len(html_content)} characters")
            
            # Look for key sections
            sections_to_find = [
                "GYROIDIC DIEGETIC TERMINAL",
                "KNOWLEDGE ASSOCIATION PANEL", 
                "IMAGE→TEXT",
                "TEXT→TEXT",
                "image-text-content",
                "text-text-content",
                "association-form"
            ]
            
            print("\n🔍 Searching for key sections:")
            for section in sections_to_find:
                if section in html_content:
                    print(f"✅ Found: {section}")
                else:
                    print(f"❌ Missing: {section}")
            
            # Show first 500 characters to see what we're getting
            print(f"\n📝 First 500 characters of HTML:")
            print("-" * 50)
            print(html_content[:500])
            print("-" * 50)
            
            # Look for the specific tab text
            if "IMAGE→TEXT" in html_content:
                print("✅ IMAGE→TEXT tab found")
            elif "IMAGE" in html_content and "TEXT" in html_content:
                print("⚠️  IMAGE and TEXT found separately")
                # Find the context
                import re
                matches = re.findall(r'.{0,20}IMAGE.{0,20}', html_content, re.IGNORECASE)
                for match in matches[:3]:
                    print(f"   Context: {match}")
            else:
                print("❌ No IMAGE/TEXT references found")
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_terminal_content()