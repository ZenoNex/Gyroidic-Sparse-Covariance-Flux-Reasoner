#!/usr/bin/env python3
"""
Robust Dataset CLI Wrapper.

Provides simplified aliases and forwards commands to the core 
DatasetIngestionSystem while maintaining anti-lobotomy compliance.
"""

import sys
import os
import argparse
import subprocess

def run_command(args_list):
    """Execute the core ingestion system with the given arguments."""
    python_exe = os.path.join(".venv", "scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = sys.executable  # Fallback to system python
    
    cmd = [python_exe, "dataset_ingestion_system.py"] + args_list
    
    try:
        # We use capture_output=False to let the user see the progress in real-time
        return subprocess.run(cmd, check=True).returncode
    except subprocess.CalledProcessError as e:
        return e.returncode
    except Exception as e:
        print(f"[ERR] Error executing command: {e}")
        return 1

def handle_quick_start(args):
    """Alias for model creation + IMDB ingestion + training."""
    parser = argparse.ArgumentParser(prog='dataset_cli.py quick-start')
    parser.add_argument('--dataset', help='Name for the dataset')
    parser.add_argument('--samples', type=int, help='Max samples to load')
    ns = parser.parse_args(args)
    
    name = ns.dataset or "imdb_quick"
    samples = ns.samples or 1000
    
    print(f"[START] Initializing Quick-Start for dataset: {name}")
    
    # 1. Create Model
    print("\n[1/3] Creating default temporal model...")
    run_command(["create-model", "--name", f"{name}_model", "--type", "temporal"])
    
    # 2. Add Dataset
    print(f"\n[2/3] Ingesting {samples} samples from IMDB...")
    run_command([
        "add-dataset", 
        "--name", name, 
        "--source", "huggingface", 
        "--path", "imdb", 
        "--max-samples", str(samples)
    ])
    
    # 3. Setup and Train
    print("\n[3/3] Setting up and starting training...")
    run_command(["setup-training", "--model", f"{name}_model", "--dataset", name, "--epochs", "5"])
    return run_command(["train", "--model", f"{name}_model", "--dataset", name])

def handle_add_wiki(args):
    """Alias for simplified Wikipedia ingestion."""
    parser = argparse.ArgumentParser(prog='dataset_cli.py add-wiki')
    parser.add_argument('--topics', required=True, help='Comma-separated topics (e.g. "Quantum_mechanics,Relativity")')
    parser.add_argument('--name', help='Name for the dataset')
    parser.add_argument('--samples', type=int, help='Max articles to load')
    ns = parser.parse_args(args)
    
    name = ns.name or f"wiki_{ns.topics.split(',')[0][:10]}"
    return run_command([
        "add-dataset", 
        "--name", name, 
        "--source", "wikipedia", 
        "--path", ns.topics,
        "--max-samples", str(ns.samples or 10)
    ])

def handle_train_local(args):
    """Alias for local directory ingestion and training."""
    parser = argparse.ArgumentParser(prog='dataset_cli.py train-local')
    parser.add_argument('--path', required=True, help='Path to local directory')
    parser.add_argument('--name', help='Name for the dataset')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    ns = parser.parse_args(args)
    
    name = ns.name or os.path.basename(ns.path.rstrip("/\\"))
    run_command([
        "add-dataset",
        "--name", name,
        "--source", "local",
        "--path", ns.path,
        "--preprocessing", "text"
    ])
    run_command(["create-model", "--name", f"{name}_model", "--type", "temporal"])
    run_command(["setup-training", "--model", f"{name}_model", "--dataset", name, "--epochs", str(ns.epochs or 10)])
    return run_command(["train", "--model", f"{name}_model", "--dataset", name])

def main():
    if len(sys.argv) < 2:
        run_command(["--help"])
        print("\n" + "="*30)
        print("Dataset CLI Wrapper Aliases:")
        print("  quick-start   Automated model creation + IMDB ingestion + training")
        print("  add-wiki      Simplified Wikipedia topic ingestion")
        print("  train-local   Local directory ingestion and training")
        print("  list-all      List all datasets, models, and sessions")
        sys.exit(0)
    
    command = sys.argv[1]
    command_args = sys.argv[2:]
    
    if command == 'quick-start':
        sys.exit(handle_quick_start(command_args))
    elif command == 'add-wiki':
        sys.exit(handle_add_wiki(command_args))
    elif command == 'train-local':
        sys.exit(handle_train_local(command_args))
    elif command == 'list-all':
        sys.exit(run_command(['list-all']))
    else:
        # Forward everything else to the core system
        sys.exit(run_command([command] + command_args))

if __name__ == "__main__":
    main()