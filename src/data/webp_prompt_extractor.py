"""
WebP Prompt Extractor.
Extracts SHA256 encoded prompts hidden in ChatGPT exported WebP images.
"""
from pathlib import Path
from typing import Dict, List, Any
import hashlib

class WebPPromptExtractor:
    """
    Parses ChatGPT WebP image artifacts to extract embedded text prompts
    from RIFF chunks (EXIF, XMP).
    """
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def extract_from_webp(self, filepath: Path) -> Dict[str, Any]:
        """Extracts hidden prompts or EXIF/metadata chunks from a WebP file."""
        if not filepath.exists():
            return {}
            
        try:
            # Read raw bytes
            data = filepath.read_bytes()
            
            # Extract SHA256 of the whole image
            img_hash = hashlib.sha256(data).hexdigest()
            
            extracted_prompts = []
            
            # Attempt to parse basic RIFF structure
            # RIFF [4 bytes size] WEBP
            if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
                offset = 12
                while offset < len(data):
                    if offset + 8 > len(data):
                        break
                    chunk_id = data[offset:offset+4]
                    chunk_size = int.from_bytes(data[offset+4:offset+8], byteorder='little')
                    
                    if chunk_id in (b'EXIF', b'XMP ', b'ICCP'):
                        chunk_data = data[offset+8:offset+8+chunk_size]
                        # Try to extract text strings from binary chunk
                        try:
                            text = chunk_data.decode('utf-8', errors='ignore')
                            # Heuristic: look for DALL-E prompt markers or general prompts
                            if 'prompt' in text.lower() or '{' in text:
                                extracted_prompts.append(text.strip())
                        except Exception:
                            pass
                            
                    # Move to next chunk (padding byte if size is odd)
                    offset += 8 + chunk_size + (chunk_size % 2)
            
            # Deep hash of the prompts themselves
            prompt_hash = None
            if extracted_prompts:
                prompt_str = "\n".join(extracted_prompts)
                prompt_hash = hashlib.sha256(prompt_str.encode()).hexdigest()
                
            return {
                "file": filepath.name,
                "image_sha256": img_hash,
                "prompt_sha256": prompt_hash,
                "extracted_prompts": extracted_prompts
            }
        except Exception as e:
            return {"error": str(e), "file": filepath.name}
            
    def scan_directory(self) -> List[Dict[str, Any]]:
        """Scans the configured directory for WebP files."""
        results = []
        if self.data_dir.exists() and self.data_dir.is_dir():
            for webp_file in self.data_dir.rglob("*.webp"):
                results.append(self.extract_from_webp(webp_file))
        return results
