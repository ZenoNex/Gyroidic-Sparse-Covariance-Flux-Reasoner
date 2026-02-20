#!/usr/bin/env python3
"""
Fix syntax errors introduced by the import fixer.
"""

import os
import re
from pathlib import Path

def fix_file_syntax(file_path: Path):
    """Fix syntax errors in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix misplaced path setup code
        # Remove path setup that was inserted in the wrong place
        lines = content.split('\n')
        fixed_lines = []
        skip_next = 0
        
        for i, line in enumerate(lines):
            if skip_next > 0:
                skip_next -= 1
                continue
                
            # Look for misplaced path setup
            if ('if os.path.dirname(os.path.abspath(__file__))' in line and 
                i > 0 and lines[i-1].strip() and not lines[i-1].strip().startswith('#')):
                
                # This path setup is misplaced, skip it and the next few lines
                skip_count = 0
                for j in range(i, min(i+10, len(lines))):
                    if 'sys.path.insert' in lines[j] or 'os.path.join' in lines[j]:
                        skip_count += 1
                    else:
                        break
                skip_next = skip_count - 1
                continue
            
            fixed_lines.append(line)
        
        # Add proper path setup at the top
        if 'sys.path.insert' not in content[:500]:  # If no path setup in first 500 chars
            # Find where to insert (after initial imports)
            insert_pos = 0
            for i, line in enumerate(fixed_lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    insert_pos = i + 1
                elif line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                    break
            
            path_setup = [
                '',
                '# Fix import paths',
                'import sys',
                'import os',
                'if os.path.dirname(os.path.abspath(__file__)) not in sys.path:',
                '    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))',
                'if os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") not in sys.path:',
                '    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))',
                ''
            ]
            
            fixed_lines[insert_pos:insert_pos] = path_setup
        
        new_content = '\n'.join(fixed_lines)
        
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix syntax errors in problematic files."""
    print("🔧 Fixing Syntax Errors")
    print("=" * 25)
    
    # Files that had syntax errors
    problem_files = [
        'src/training/trainer.py',
        'src/models/gyroid_reasoner.py'
    ]
    
    fixed_count = 0
    
    for file_path_str in problem_files:
        file_path = Path(file_path_str)
        if file_path.exists():
            print(f"🔧 Fixing {file_path}...")
            if fix_file_syntax(file_path):
                print(f"   ✅ Fixed syntax errors")
                fixed_count += 1
            else:
                print(f"   ⚠️  No changes needed")
        else:
            print(f"   ❌ File not found: {file_path}")
    
    print(f"\n📊 Fixed {fixed_count} files")
    
    if fixed_count > 0:
        print(f"🧪 Test the fixes:")
        print(f"   python debug_imports.py")

if __name__ == "__main__":
    main()