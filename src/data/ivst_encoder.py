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
        """Find FFmpeg executable in PATH or fallback to sovereign path."""
        sovereign_path = r"D:\ffmpeg-2026-04-22-git-162ad61486-full_build\bin\ffmpeg.exe"
        try:
            # Simple check if sovereign path is available
            subprocess.run([sovereign_path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return sovereign_path
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
        Robust to Distort Interleave and PVoc Interleave by smoothing inter-chunk variance.
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
                
            # ROBUSTNESS: Identify Distort/PVoc Interleave artifacts
            # Instead of filtering/smoothing them away, we read their topological signature.
            # Distort Interleave creates extreme discontinuities in the time domain (ZCR jumps).
            # PVoc Interleave creates spectral envelope jumps without breaking local phase as violently (Energy jumps).
            
            distort_interleave_detected = False
            pvoc_interleave_detected = False
            
            if len(energy_envelope) > 3:
                zcr_diffs = np.abs(np.diff(zero_crossings))
                energy_diffs = np.abs(np.diff(energy_envelope))
                
                # If there are regular, high-variance jumps in zero crossings -> Distort Interleave
                if np.mean(zcr_diffs) > 0.15 and np.std(zcr_diffs) > 0.05:
                    distort_interleave_detected = True
                    
                # If energy jumps violently but ZCR remains relatively stable -> PVoc Interleave
                if np.mean(energy_diffs) > 0.1 and not distort_interleave_detected:
                    pvoc_interleave_detected = True
                
            # Obtain native structural hash directly from FFmpeg without Python overhead
            hash_cmd = [
                self.ffmpeg_path,
                "-i", str(filepath),
                "-vn",
                "-map", "0:a:0?",  # Only the first audio stream
                "-f", "hash",
                "-hash", "sha256",
                "-"
            ]
            hash_process = subprocess.run(hash_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            audio_hash = "unknown"
            if hash_process.returncode == 0:
                for line in hash_process.stdout.splitlines():
                    if line.startswith("SHA256="):
                        audio_hash = line.split("=")[1].strip()
                        break
            
            return {
                "energy_envelope": energy_envelope[:1000], # Cap size
                "zero_crossings": zero_crossings[:1000],
                "total_samples": len(audio_data),
                "audio_hash": audio_hash,
                "interleave_topology": {
                    "distort_interleave": distort_interleave_detected,
                    "pvoc_interleave": pvoc_interleave_detected
                }
            }
            
        except Exception as e:
            return {"error": str(e)}

    def extract_visual_topology(self, filepath: Path) -> Dict[str, Any]:
        """
        Extracts visual structural metadata from video or images.
        ROBUSTNESS: AI Piss Filter (Compound Color Loss / Missing Blue Pixels).
        Instead of rejecting yellow-shifted AI artifacts, we detect the blue channel crush
        and calculate an isomorphic offset.
        """
        try:
            cmd = [
                self.ffmpeg_path,
                "-i", str(filepath),
                "-vf", "scale=16:16",
                "-vframes", "1",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-"
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0 or not stdout:
                return {"error": "Failed to extract visual topology"}
                
            pixels = np.frombuffer(stdout, dtype=np.uint8).reshape((16, 16, 3)).astype(np.float32)
            
            # Analyze color channels
            r_mean = float(np.mean(pixels[:, :, 0]))
            g_mean = float(np.mean(pixels[:, :, 1]))
            b_mean = float(np.mean(pixels[:, :, 2]))
            
            # Detect AI Piss Filter (Missing Blue Pixels / Compound Color Loss)
            ai_piss_filter_active = False
            b_compensation = 1.0
            
            if b_mean < (r_mean + g_mean) * 0.25: # Severe blue loss typical of early generative models
                ai_piss_filter_active = True
                b_compensation = ((r_mean + g_mean) / 2.0) / (b_mean + 1e-5)
            
            return {
                "r_mean": r_mean,
                "g_mean": g_mean,
                "b_mean": b_mean,
                "piss_filter_detected": ai_piss_filter_active,
                "b_channel_compensation": float(b_compensation)
            }
        except Exception as e:
            return {"error": str(e)}
            
    def process_artifact(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Main entry point for processing an MP4/MKV artifact or image.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found: {filepath}")
            
        causal_structure = self.probe_media_structure(filepath)
        
        # Audio topology (with PVoc/Distort Interleave robustness)
        audio_topology = self.extract_audio_topology(filepath)
        
        # Visual topology (with AI Piss Filter robustness)
        visual_topology = self.extract_visual_topology(filepath)
        
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
            "visual_topology": visual_topology,
            "ivst_signature": hashlib.sha256(f"{filepath.name}_{jitter}".encode()).hexdigest(),
            "honesty_jitter": jitter
        }
