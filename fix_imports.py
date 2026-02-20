#!/usr/bin/env python3
"""
Fix Import Structure - Converts relative imports to absolute imports.

This script fixes the "attempted relative import beyond top-level package" errors
by converting all relative imports to absolute imports throughout the project.
"""

import os
import re
import sys
from pathlib import Path

def fix_file_imports(file_path: Path, src_root: Path):
    """Fix imports in a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Pattern to match relative imports
        patterns = [
            (r'from \.\.([a-zA-Z_][a-zA-Z0-9_]*)', r'from src.\1'),  # from ..module
            (r'from \.([a-zA-Z_][a-zA-Z0-9_]*)', r'from src.{}\1'.format(file_path.parent.name + '.')),  # from .module
        ]
        
        for pattern, replacement in patterns:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                changes_made.extend(matches)
        
        # Special handling for training module imports
        if 'src/training' in str(file_path):
            # Fix imports within training modules
            content = re.sub(r'from \.\.core\.', 'from src.core.', content)
            content = re.sub(r'from \.\.models\.', 'from src.models.', content)
            content = re.sub(r'from \.\.topology\.', 'from src.topology.', content)
            content = re.sub(r'from \.\.optimization\.', 'from src.optimization.', content)
        
        # Special handling for models module imports
        if 'src/models' in str(file_path):
            content = re.sub(r'from \.\.core\.', 'from src.core.', content)
            content = re.sub(r'from \.\.training\.', 'from src.training.', content)
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return changes_made
        
        return []
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return []

def add_path_setup(file_path: Path):
    """Add sys.path setup to files that need it."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if path setup already exists
        if 'sys.path.append' in content or 'sys.path.insert' in content:
            return False
        
        # Add path setup after imports
        lines = content.split('\n')
        insert_pos = 0
        
        # Find position after initial imports and docstring
        in_docstring = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
                
            # Find last import
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_pos = i + 1
        
        # Insert path setup
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
        
        lines[insert_pos:insert_pos] = path_setup
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding path setup to {file_path}: {e}")
        return False

def main():
    """Fix all import issues in the project."""
    print("🔧 Fixing Project Import Structure")
    print("=" * 40)
    
    project_root = Path('.')
    src_root = project_root / 'src'
    
    if not src_root.exists():
        print("❌ src directory not found!")
        return 1
    
    # Find all Python files in src
    python_files = []
    for pattern in ['**/*.py']:
        python_files.extend(src_root.glob(pattern))
    
    print(f"🔍 Found {len(python_files)} Python files in src/")
    
    total_changes = 0
    files_modified = 0
    
    # Fix imports in each file
    for file_path in python_files:
        if file_path.name == '__init__.py':
            continue  # Skip __init__.py files
            
        print(f"🔧 Processing {file_path.relative_to(project_root)}...")
        
        changes = fix_file_imports(file_path, src_root)
        if changes:
            print(f"   ✅ Fixed {len(changes)} imports: {', '.join(changes[:3])}{'...' if len(changes) > 3 else ''}")
            files_modified += 1
            total_changes += len(changes)
        
        # Add path setup for problematic files
        if any(problem in str(file_path) for problem in ['training', 'models']):
            if add_path_setup(file_path):
                print(f"   ✅ Added path setup")
    
    # Also fix the main files that import from src
    main_files = [
        'dataset_command_interface.py',
        'dataset_ingestion_system.py'
    ]
    
    for main_file in main_files:
        file_path = project_root / main_file
        if file_path.exists():
            print(f"🔧 Processing {main_file}...")
            if add_path_setup(file_path):
                print(f"   ✅ Added path setup")
    
    print(f"\n📊 Import Fix Summary:")
    print(f"   Files modified: {files_modified}")
    print(f"   Total imports fixed: {total_changes}")
    
    if total_changes > 0:
        print(f"\n✅ Import structure fixed!")
        print(f"🧪 Test the fixes:")
        print(f"   python debug_imports.py")
        print(f"   python dataset_command_interface.py --help")
    else:
        print(f"\n⚠️  No import issues found to fix")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())