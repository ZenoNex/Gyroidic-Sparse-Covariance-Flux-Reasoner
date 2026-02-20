

# Rewriting the whole file to be cleaner and log to file
import os
import ast
import sys

def check_syntax(directory, log_file):
    print(f"Checking syntax in {directory}...")
    log_file.write(f"Checking syntax in {directory}...\n")
    error_count = 0
    for root, dirs, files in os.walk(directory):
        if ".venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        source = f.read()
                    ast.parse(source)
                except SyntaxError as e:
                    msg = f"SyntaxError in {path}: {e}"
                    print(msg)
                    log_file.write(msg + "\n")
                    error_count += 1
                except Exception as e:
                    msg = f"Error reading {path}: {e}"
                    print(msg)
                    log_file.write(msg + "\n")
    return error_count

if __name__ == "__main__":
    with open("syntax_errors_utf8.log", "w", encoding="utf-8") as log_file:
        src_errors = check_syntax("src", log_file)
        examples_errors = check_syntax("examples", log_file)
        total = src_errors + examples_errors
        
        if total == 0:
            msg = "No syntax errors found."
            print(msg)
            log_file.write(msg + "\n")
            sys.exit(0)
        else:
            msg = f"Found {total} syntax errors."
            print(msg)
            log_file.write(msg + "\n")
            sys.exit(1)