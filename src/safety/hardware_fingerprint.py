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

SIG_FILE = os.path.join("data", "creator_sig.json")

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
    
    raw_payload = f"{system_uuid}:{mac}:{proc}:{machine}"
    return hashlib.sha256(raw_payload.encode('utf-8')).hexdigest()

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

def is_creator_initialized() -> bool:
    """Returns True if the sole creator configuration has been set up."""
    return os.path.exists(SIG_FILE)

def verify_sole_creator(passphrase: str) -> bool:
    """
    Verifies or registers the sole creator passcode bound to the hardware fingerprint.
    
    If no signature file exists, initializes it with the hashed passcode + hardware key.
    If it exists, validates the passcode against the saved hash on the current machine's fingerprint.
    """
    os.makedirs("data", exist_ok=True)
    hw_fingerprint = get_stable_hardware_fingerprint()
    combined_raw = f"{passphrase}:{hw_fingerprint}"
    current_hash = hashlib.sha256(combined_raw.encode('utf-8')).hexdigest()
    
    if not is_creator_initialized():
        # Initialize first-time registration
        try:
            with open(SIG_FILE, "w") as f:
                json.dump({"creator_hash": current_hash}, f, indent=4)
            print(f"[SECURITY] Sole Creator registered successfully on this hardware.")
            return True
        except Exception as e:
            print(f"[SECURITY] Failed to save creator signature: {e}")
            return False
            
    # Validate existing signature
    try:
        with open(SIG_FILE, "r") as f:
            data = json.load(f)
            saved_hash = data.get("creator_hash")
            return current_hash == saved_hash
    except Exception:
        return False
