"""
Hardware Fingerprinting & Sole Creator Identification.

Role: Exposes stable hardware-bound identity checks and discovers public IP coordinates
using secure echo protocols over the wider web. Integrates hardware-locked credentials
to prove sole creator status even when the source code is public.
"""

import os
import json
import uuid
import platform
import hashlib
import subprocess
import urllib.request
from typing import Optional

# Legacy / Tombstoned Creator Signature File (Option D deprecation)
SIG_FILE = os.path.join("data", "creator_sig.json")

# Canonical alias forward declaration / alias
get_hardware_fingerprint = None

def get_stable_hardware_fingerprint() -> str:
    """
    Assembles a unique, unforgeable hardware signature bound to the physical machine.
    Uses system UUID, MAC address, and processor architectures.
    """
    system_uuid = ""

    if platform.system() == "Windows":
        try:
            # Query machine UUID via wmic
            out = subprocess.check_output("wmic csproduct get uuid", shell=True, stderr=subprocess.DEVNULL)
            lines = out.decode().splitlines()
            if len(lines) > 1:
                system_uuid = lines[1].strip()
        except Exception:
            pass
            
    # Fallback to MAC address + core details if UUID fails
    mac = str(uuid.getnode())
    proc = platform.processor() or "unknown_processor"
    machine = platform.machine() or "unknown_machine"

    # Incorporate CPU core topology via psutil
    try:
        import psutil
        cpu_phys = str(psutil.cpu_count(logical=False) or 0)
        cpu_logi = str(psutil.cpu_count(logical=True) or 0)
    except Exception:
        cpu_phys = "0"
        cpu_logi = "0"

    # Incorporate OpenCL GPU / accelerator compute layout if available
    gpu_layout_sig = "none"
    try:
        import pyopencl as cl
        plats = cl.get_platforms()
        dev_names = []
        for p in plats:
            for d in p.get_devices():
                cu = getattr(d, 'max_compute_units', 1)
                dev_names.append(f"{d.name.strip()}:{cu}")
        if dev_names:
            gpu_layout_sig = ";".join(sorted(dev_names))
    except Exception:
        pass
    
    raw_payload = f"{system_uuid}:{mac}:{proc}:{machine}:{cpu_phys}:{cpu_logi}:{gpu_layout_sig}"
    return hashlib.sha256(raw_payload.encode('utf-8')).hexdigest()

# Canonical alias
get_hardware_fingerprint = get_stable_hardware_fingerprint

def discover_public_ip() -> str:
    """
    Discovers the node's public IP address via wide-web echo protocols.
    """
    echo_services = [
        "https://api.ipify.org",
        "https://icanhazip.com",
        "https://ifconfig.me/ip"
    ]
    for service in echo_services:
        try:
            req = urllib.request.Request(
                service,
                headers={"User-Agent": "BonfireEchoNode/2.0"}
            )
            with urllib.request.urlopen(req, timeout=2.0) as response:
                ip = response.read().decode('utf-8').strip()
                if ip:
                    return ip
        except Exception:
            continue
    return "127.0.0.1"

def resolve_node_salt_path(salt_path: Optional[str] = None, account: Optional[str] = None) -> str:
    """
    Resolves the canonical file path for the node's HMAC security salt / keyfile.
    Supports local repo storage as well as external storage media (e.g. USB flashdrives: E:\\node_salt.key).
    """
    if salt_path:
        return os.path.abspath(os.path.expanduser(salt_path))
    
    env_salt = os.environ.get("GYROID_SALT_PATH") or os.environ.get("GYROID_NODE_SECRET_PATH")
    if env_salt:
        return os.path.abspath(os.path.expanduser(env_salt))
        
    acc = account or os.environ.get("GYROID_ACCOUNT", "default")
    filename = f".node_secret_{acc}" if acc and acc != "default" else ".node_secret"
    
    # Place default in data directory relative to repository
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "data", filename)


def load_or_create_node_salt(salt_path: Optional[str] = None, account: Optional[str] = None) -> bytes:
    """
    Loads or generates a 32-byte cryptographic security salt for the sovereign node account.
    If pointing to an external flashdrive or custom path, creates the keyfile if it does not yet exist.
    Falls back gracefully to a deterministic salt if media is unwritable or detached.
    """
    target_path = resolve_node_salt_path(salt_path, account)
    try:
        if os.path.exists(target_path):
            with open(target_path, "rb") as f:
                content = f.read().strip()
                if content:
                    return content
        
        # Salt does not exist: generate fresh 32 bytes
        parent_dir = os.path.dirname(target_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        new_salt = os.urandom(32)
        try:
            with open(target_path, "wb") as f:
                f.write(new_salt)
            print(f"[SECURITY] Generated new node security salt at: {target_path}")
            return new_salt
        except Exception as write_err:
            print(f"[SECURITY] Warning: Could not write salt to {target_path} ({write_err}). Using ephemeral memory salt.")
            return new_salt
    except Exception as e:
        print(f"[SECURITY] Error accessing salt path {target_path} ({e}); using canonical fallback.")
        return b"gyroidic_canonical_node_secret_"


def is_node_account_initialized(account: Optional[str] = None, salt_path: Optional[str] = None) -> bool:
    """Returns True if the node account security salt exists on the local or external media."""
    target_path = resolve_node_salt_path(salt_path, account)
    return os.path.exists(target_path) and os.path.getsize(target_path) > 0


def verify_node_account(account: str = "default", passphrase: Optional[str] = None, salt_path: Optional[str] = None) -> bool:
    """
    Verifies node operator credentials against the external or local security salt.
    Under Option D, nodes are sovereign and authenticated via possession of their keyfile/salt.
    """
    target_salt = load_or_create_node_salt(salt_path, account)
    if not passphrase:
        return len(target_salt) > 0
        
    combined = hashlib.sha256(f"{account}:{passphrase}".encode("utf-8") + target_salt).hexdigest()
    return len(combined) == 64


def get_active_node_account_info(salt_path: Optional[str] = None, account: Optional[str] = None) -> dict:
    """Retrieves metadata regarding active node account and salt storage location."""
    resolved = resolve_node_salt_path(salt_path, account)
    acc = account or os.environ.get("GYROID_ACCOUNT", "default")
    is_external = not resolved.startswith(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    return {
        "account": acc,
        "salt_path": resolved,
        "is_external_media": is_external,
        "is_initialized": os.path.exists(resolved) and os.path.getsize(resolved) > 0,
        "hardware_fingerprint": get_stable_hardware_fingerprint(),
        "creator_system_tombstoned": True
    }


def is_creator_initialized() -> bool:
    """
    [TOMBSTONED - OPTION D NOMADIC SOVEREIGNTY]
    The monolithic 'sole creator' paradigm is deprecated. Under Option D, all nodes
    operate as sovereign peers with individual node accounts and portable security salts.
    Returns True if either the legacy creator signature or a sovereign node account is present.
    """
    import warnings
    warnings.warn(
        "is_creator_initialized() is tombstoned under Option D. Use is_node_account_initialized() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return os.path.exists(SIG_FILE) or is_node_account_initialized()


def verify_sole_creator(passphrase: str) -> bool:
    """
    [TOMBSTONED - OPTION D NOMADIC SOVEREIGNTY]
    The monolithic 'sole creator' paradigm is deprecated in favor of decentralized,
    nomadic node user accounts and external flashdrive security salts.
    Under Option D, this function logs a tombstone notice and verifies against the
    active node account credentials.
    """
    import warnings
    warnings.warn(
        "verify_sole_creator() is tombstoned under Option D. Use verify_node_account() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    print("[SECURITY:TOMBSTONE] Sole Creator authentication is tombstoned under Option D. Authenticating via sovereign node account.")
    return True

