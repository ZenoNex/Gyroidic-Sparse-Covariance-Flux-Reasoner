import os
import sys
import json
import urllib.request
import zipfile
import shutil
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = "ZenoNex/Gyroidic-Sparse-Covariance-Flux-Reasoner"
GITHUB_API = f"https://api.github.com/repos/{REPO}"
ZIP_URL = f"https://github.com/{REPO}/archive/refs/heads/main.zip"

def get_local_version():
    version_file = Path(".version")
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.0.0-unknown"

def hash_file(filepath):
    """Compute SHA256 of a file."""
    if not os.path.exists(filepath):
        return None
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def fetch_latest_remote_info():
    """Fetch latest commit SHA on main as a proxy for version if no releases exist."""
    print(f"[*] Checking {GITHUB_API}/commits/main")
    req = urllib.request.Request(f"{GITHUB_API}/commits/main")
    req.add_header("User-Agent", "Gyroidic-Updater-Client")
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data["sha"][:7], data["commit"]["message"].split("\n")[0]
    except Exception as e:
        print(f"[!] Error fetching remote version: {e}")
        return None, None

def download_and_extract(staging_dir):
    print(f"[*] Downloading main branch archive...")
    zip_path = os.path.join(staging_dir, "update.zip")
    req = urllib.request.Request(ZIP_URL)
    req.add_header("User-Agent", "Gyroidic-Updater-Client")
    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    
    print(f"[*] Extracting archive to staging...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(staging_dir)
    
    # The zip usually contains a root folder like "Gyroidic-Sparse-Covariance-Flux-Reasoner-main"
    extracted_dirs = [d for d in os.listdir(staging_dir) if os.path.isdir(os.path.join(staging_dir, d))]
    if len(extracted_dirs) == 1:
        return os.path.join(staging_dir, extracted_dirs[0])
    return staging_dir

def compute_diff(staging_root, local_root):
    """Compare staging directory to local directory and classify differences."""
    diff = {
        "new": [],
        "modified": [],
        "removed_remote": [] # Orphan local files not in remote
    }
    
    # We only care about src and docs for now to protect the environment and root config.
    target_dirs = ["src", "docs"]
    
    # Check what remote has vs local
    for target in target_dirs:
        remote_target = os.path.join(staging_root, target)
        if not os.path.exists(remote_target):
            continue
            
        for root, _, files in os.walk(remote_target):
            for file in files:
                remote_path = os.path.join(root, file)
                rel_path = os.path.relpath(remote_path, staging_root)
                local_path = os.path.join(local_root, rel_path)
                
                if not os.path.exists(local_path):
                    diff["new"].append((rel_path, remote_path))
                else:
                    if hash_file(remote_path) != hash_file(local_path):
                        diff["modified"].append((rel_path, remote_path))
                        
    # Identify local orphans (files we have that remote doesn't)
    for target in target_dirs:
        local_target = os.path.join(local_root, target)
        if not os.path.exists(local_target):
            continue
            
        for root, _, files in os.walk(local_target):
            # Ignore __pycache__
            if "__pycache__" in root:
                continue
            for file in files:
                if file.endswith(".pyc"):
                    continue
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_root)
                remote_path = os.path.join(staging_root, rel_path)
                
                if not os.path.exists(remote_path):
                    diff["removed_remote"].append(rel_path)
                    
    return diff

def interactive_prompt(diff, local_root):
    """Prompt the user for each file to ensure no accidental downgrades."""
    print("\n" + "="*50)
    print("UPDATE DIFF ANALYSIS COMPLETE")
    print("="*50)
    print(f"New Files: {len(diff['new'])}")
    print(f"Modified Files: {len(diff['modified'])}")
    print(f"Local Orphans (Not in remote): {len(diff['removed_remote'])}")
    
    total_changes = sum(len(v) for v in diff.values())
    if total_changes == 0:
        print("[*] Your local codebase is perfectly synchronized with the remote main branch.")
        return

    print("\n[!] WARNING: You have unpushed local changes. We will review files one-by-one.")
    
    for category, prefix in [("new", "[NEW]"), ("modified", "[MOD]")]:
        for rel_path, remote_path in diff[category]:
            local_path = os.path.join(local_root, rel_path)
            
            while True:
                choice = input(f"Apply {prefix} {rel_path}? [y/n/q]: ").strip().lower()
                if choice == 'y':
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    shutil.copy2(remote_path, local_path)
                    print(f"  -> Applied {rel_path}")
                    break
                elif choice == 'n':
                    print(f"  -> Skipped {rel_path} (Local preserved)")
                    break
                elif choice == 'q':
                    print("[*] Aborting update process. All remaining files preserved.")
                    return
                
    if diff["removed_remote"]:
        print("\n[!] The following files exist locally but are NOT in the remote repository.")
        print("[!] These are likely your unpushed creations. We will NOT delete them.")
        for rel_path in diff["removed_remote"][:10]:
            print(f"  - {rel_path}")
        if len(diff["removed_remote"]) > 10:
            print(f"  ... and {len(diff['removed_remote']) - 10} more.")

def main():
    print("=== Gyroidic Update Client ===")
    local_version = get_local_version()
    print(f"Local Version Baseline: {local_version}")
    
    remote_hash, msg = fetch_latest_remote_info()
    if not remote_hash:
        sys.exit(1)
        
    print(f"Remote Latest (main): {remote_hash} - {msg}")
    
    proceed = input("\nDo you want to stage this update for analysis? [y/N]: ").strip().lower()
    if proceed != 'y':
        print("[*] Update aborted.")
        sys.exit(0)
        
    local_root = os.getcwd()
    
    with TemporaryDirectory() as staging_dir:
        extracted_root = download_and_extract(staging_dir)
        print("[*] Analyzing cryptographic diffs against local source...")
        diff = compute_diff(extracted_root, local_root)
        interactive_prompt(diff, local_root)

    print("\n[*] Update session finished safely.")
    
if __name__ == "__main__":
    main()
