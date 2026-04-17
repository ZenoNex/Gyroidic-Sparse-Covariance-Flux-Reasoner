import time
import requests
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import datetime
import threading
from src.core.knowledge_dyad_fossilizer import KnowledgeDyad, DyadFossilizer

class ArXivSovereignIngestor:
    """
    Implements a 'Slow-Drip' non-teleological knowledge ingestor using ArXiv OAI-PMH.
    Complies with the 3-second rate limit to ensure non-invasive learning.
    
    This ingestor treats scientific metadata as 'Lore'—seeding the manifold with
    conceptual residues from late-2025/early-2026 mathematical research.
    """
    def __init__(self, fossilizer: DyadFossilizer, engine_dim: int, device: str = 'cpu'):
        self.fossilizer = fossilizer
        self.engine_dim = engine_dim
        self.device = device
        self.base_url = "http://export.arxiv.org/oai2"
        self.last_request_time = 0
        self.rate_limit_seconds = 4.0 # Conservatively above the 3s requirement
        
        # NS Map for ArXiv OAI-PMH
        self.ns = {
            'oai': 'http://www.openarchives.org/OAI/2.0/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'
        }
        
        # Simple projection to map ASCII frequency hashes to engine latent space
        # This provides a deterministic 'structural signature' for each abstract.
        self.text_proj = nn.Linear(128, engine_dim).to(device)
        nn.init.orthogonal_(self.text_proj.weight)

    def _wait_for_rate_limit(self):
        """Ensures compliance with ArXiv's anti-crawling policies."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _text_to_pseudo_residue(self, text: str) -> torch.Tensor:
        """Converts text into a deterministic latent signature."""
        # Create an ASCII frequency distribution (Histogram)
        emb = torch.zeros(128, device=self.device)
        for char in text[:1024]: # Limit to first 1024 chars for speed
            code = ord(char)
            if code < 128:
                emb[code] += 1.0
        
        # Layer normalization to prevent energy blowup
        emb = emb / (torch.norm(emb) + 1e-8)
        
        with torch.no_grad():
            res = self.text_proj(emb.unsqueeze(0))
        return res

    def ingest_latest_math(self, set_name: str = "math"):
        """Fetches the latest arrivals from ArXiv and fossilizes them into the manifold."""
        self._wait_for_rate_limit()
        params = {
            'verb': 'ListRecords',
            'metadataPrefix': 'oai_dc',
            'set': set_name
        }
        
        try:
            print(f"[INGEST] Querying ArXiv lore bank (set: {set_name})...")
            response = requests.get(self.base_url, params=params, timeout=20)
            if response.status_code == 200:
                self._parse_and_fossilize(response.text)
            else:
                print(f"[INGEST] Failed to reach ArXiv (HTTP {response.status_code}). Manifold remains local.")
        except Exception as e:
            print(f"[INGEST] Transport error: {e}. Ingestion suspended.")

    def _parse_and_fossilize(self, xml_text: str):
        """Parses OAI-PMH XML and converts records into permanent knowledge fossils."""
        try:
            root = ET.fromstring(xml_text)
            records = root.findall('.//oai:record', self.ns)
            
            ingested_count = 0
            for record in records[:5]: # Cap at 5 papers per pull to maintain non-teleological drift
                metadata = record.find('.//oai_dc:dc', self.ns)
                if metadata is not None:
                    title_elem = metadata.find('dc:title', self.ns)
                    desc_elem = metadata.find('dc:description', self.ns)
                    id_elem = metadata.find('dc:identifier', self.ns)
                    
                    title = title_elem.text if title_elem is not None else "Unknown Title"
                    abstract = desc_elem.text if desc_elem is not None else "No Abstract"
                    arxiv_id = id_elem.text if id_elem is not None else "No ID"
                    
                    # Construct the Dyadic Anchor
                    anchor = f"{title}. [{arxiv_id}] Abstract: {abstract[:400]}..."
                    
                    # Generate the irreducible structural residue
                    residue = self._text_to_pseudo_residue(anchor)
                    
                    # Create the Dyad (Text-only metadata capture)
                    # We use a distinct 'empty' fingerprint for external lore
                    dyad = KnowledgeDyad(
                        image_fingerprint=torch.zeros(137, device=self.device),
                        linguistic_description=anchor,
                        relevance_score=0.7,
                        timestamp=datetime.datetime.now().isoformat()
                    )
                    
                    # Fossilize locally
                    self.fossilizer.fossilize(dyad, residue)
                    ingested_count += 1
                    print(f" [LORE] Fossilized: {title[:50]}...")
            
            if ingested_count > 0:
                print(f"[INGEST] Successfully anchored {ingested_count} mathematical residues into the substrate.")
        except Exception as e:
            print(f"[INGEST] Parsing error: {e}. XML structure mismatch.")

    def start_sovereign_loop(self):
        """Starts the background ingestion thread."""
        def _loop():
            # Cycle through high-density mathematical and AI sets
            sets = ["math", "physics:quant-ph", "cs:AI", "math.LO", "math.HO"]
            while True:
                for s in sets:
                    self.ingest_latest_math(s)
                    # Large gap between sets to respect community resources
                    time.sleep(30)
                # Sleep for 1 hour after cycling all sets
                time.sleep(3600)
                
        bg_thread = threading.Thread(target=_loop, daemon=True)
        bg_thread.start()
        print("[INGEST] ArXiv Sovereign Ingestor active. Background lore capture in progress.")
