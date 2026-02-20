#!/usr/bin/env python3
"""
Emoji Removal Script

Removes all Unicode emojis from GUI files to fix Tcl compatibility issues.
"""

import re
import os
from pathlib import Path

def remove_emojis(text):
    """Remove all emojis from text."""
    # Unicode ranges for emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "]+", 
        flags=re.UNICODE
    )
    
    # Remove emojis but keep the space after them
    cleaned = emoji_pattern.sub('', text)
    
    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned

def fix_file(filepath):
    """Fix emojis in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        cleaned_content = remove_emojis(content)
        
        if original_content != cleaned_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"✓ Fixed: {filepath}")
            return True
        else:
            print(f"- No emojis found: {filepath}")
            return False
            
    except Exception as e:
        print(f"✗ Error fixing {filepath}: {e}")
        return False

def main():
    """Main emoji removal process."""
    print("🔧 Emoji Removal Script")
    print("=" * 30)
    
    # Files to fix
    files_to_fix = [
        'src/ui/conversational_gui.py',
        'launch_conversational_gui.py',
        'examples/fallback_conversational_demo.py',
        'examples/debug_hf_connection.py',
        'test_hf_token.py',
        'check_requirements.py'
    ]
    
    fixed_count = 0
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"- File not found: {filepath}")
    
    print(f"\n📊 Summary:")
    print(f"   Files processed: {len(files_to_fix)}")
    print(f"   Files fixed: {fixed_count}")
    
    if fixed_count > 0:
        print(f"\n✅ Emoji removal complete!")
        print(f"   Try running: python launch_conversational_gui.py")
    else:
        print(f"\n💡 No emojis found to remove")

if __name__ == "__main__":
    main()