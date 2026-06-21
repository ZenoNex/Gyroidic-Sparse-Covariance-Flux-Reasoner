"""
IVST (Unknowledge Domain) Audio/Video Encoder.

Parses MP4 artifacts to extract the causal structure of sound and video.
Uses FFmpeg/MoviePy for byte-level parsing without copyright infringement 
(extracts structural/causal metadata rather than raw copyrighted media).
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import hashlib
import json
import subprocess
import numpy as np

class IVSTEncoder:
    """
    Encoder for Independent Vector Spectral Topology (IVST).
    Extracts structural metadata and causality patterns from MP4/MKV artifacts
    rather than pure pixel data.
    """
    
    def __init__(self, sample_rate: int = 16000, fps: int = 2):
        self.sample_rate = sample_rate
        self.fps = fps
        self.ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable in PATH."""
        try:
            # Simple check if ffmpeg is available
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return "ffmpeg"
        except (subprocess.SubprocessError, FileNotFoundError):
            return "ffmpeg" # Assume it's in path or user will install it
            
    def probe_media_structure(self, filepath: Path) -> Dict[str, Any]:
        """
        Uses ffprobe to extract frame-level and stream-level structural metadata.
        This captures the 'causal structure' (I-frames, P-frames, audio bitrates)
        without extracting the actual copyrighted content.
        """
        if not filepath.exists():
            return {}
            
        try:
            cmd = [
                "ffprobe", 
                "-v", "quiet", 
                "-print_format", "json", 
                "-show_format", 
                "-show_streams", 
                str(filepath)
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                return {"error": "ffprobe failed"}
                
            metadata = json.loads(result.stdout)
            
            # Extract key causal structure points
            causal_structure = {
                "format": metadata.get("format", {}).get("format_name", "unknown"),
                "duration": float(metadata.get("format", {}).get("duration", 0.0)),
                "streams": []
            }
            
            for stream in metadata.get("streams", []):
                stream_info = {
                    "index": stream.get("index"),
                    "codec_type": stream.get("codec_type"),
                    "codec_name": stream.get("codec_name"),
                    "profile": stream.get("profile"),
                }
                if stream.get("codec_type") == "video":
                    stream_info["width"] = stream.get("width")
                    stream_info["height"] = stream.get("height")
                    stream_info["r_frame_rate"] = stream.get("r_frame_rate")
                elif stream.get("codec_type") == "audio":
                    stream_info["sample_rate"] = stream.get("sample_rate")
                    stream_info["channels"] = stream.get("channels")
                    
                causal_structure["streams"].append(stream_info)
                
            return causal_structure
            
        except Exception as e:
            return {"error": str(e)}

    def extract_audio_topology(self, filepath: Path) -> Dict[str, Any]:
        """
        Extracts audio and computes its topological fingerprint (Mel-spectrogram/MFCC structure).
        Does NOT save the audio, only the mathematical footprint.
        """
        try:
            # Extract raw audio bytes directly to memory using ffmpeg
            cmd = [
                self.ffmpeg_path,
                "-i", str(filepath),
                "-vn", # No video
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1", # Mono
                "-f", "s16le",
                "-" # Output to stdout
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0 or not stdout:
                return {"error": "Failed to extract audio topology"}
                
            # Convert to numpy array
            audio_data = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate simple causal topology (Energy envelope & zero-crossings)
            # This is a proxy for the actual audio content
            chunk_size = self.sample_rate // self.fps # FPS chunks per second
            
            energy_envelope = []
            zero_crossings = []
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                if len(chunk) == 0:
                    continue
                    
                energy = float(np.sum(chunk ** 2) / len(chunk))
                energy_envelope.append(energy)
                
                # Zero crossing rate (structural density proxy)
                zcr = float(np.sum(np.abs(np.diff(np.signbit(chunk)))) / len(chunk))
                zero_crossings.append(zcr)
                
            return {
                "energy_envelope": energy_envelope[:1000], # Cap size
                "zero_crossings": zero_crossings[:1000],
                "total_samples": len(audio_data),
                "audio_hash": hashlib.sha256(stdout).hexdigest()
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    def process_artifact(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Main entry point for processing an MP4/MKV artifact.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found: {filepath}")
            
        causal_structure = self.probe_media_structure(filepath)
        audio_topology = self.extract_audio_topology(filepath)
        
        # Pull structural honesty to sign the extraction
        try:
            from src.core.honest_jitter import harvest_honest_jitter
            jitter = harvest_honest_jitter((1,), device='cpu', scaled=True).item()
        except ImportError:
            jitter = 0.0
            
        return {
            "source_file": filepath.name,
            "causal_structure": causal_structure,
            "audio_topology": audio_topology,
            "ivst_signature": hashlib.sha256(f"{filepath.name}_{jitter}".encode()).hexdigest(),
            "honesty_jitter": jitter
        }
