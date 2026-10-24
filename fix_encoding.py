#!/usr/bin/env python3
"""
Fix encoding issues in the backend file.
"""

def fix_encoding():
    try:
        # Try to read with different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        content = None
        for encoding in encodings:
            try:
                with open('src/ui/diegetic_backend.py', 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"[OK] Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                print(f"[ERR] Failed with {encoding}: {e}")
        
        if content is None:
            print("[ERR] Could not read file with any encoding")
            return
        
        # Clean the content - remove problematic characters
        cleaned_content = content.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Write back as UTF-8
        with open('src/ui/diegetic_backend.py', 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print("[OK] File cleaned and saved as UTF-8")
        
        # Test if it's now valid
        import ast
        ast.parse(cleaned_content)
        print("[OK] Syntax is now valid")
        
    except Exception as e:
        print(f"[ERR] Error: {e}")

if __name__ == "__main__":
    fix_encoding()