import time
import requests
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import datetime
import threading

from src.core.knowledge_dyad_fossilizer import KnowledgeDyad, DyadFossilizer
from src.data.textbook_filter import TextbookFilter
from src.data.canonical_projection import CanonicalProjector
from src.data.conversational_api_ingestor import ConversationalDataProcessor

class ArXivSovereignIngestor:
    """
    Implements a 'Slow-Drip' non-teleological knowledge ingestor using ArXiv OAI-PMH.
    Complies with the 3-second rate limit to ensure non-invasive learning.
    
    Retrofitted with:
    - TextbookFilter: Multi-dimensional quality gating (Structural Honesty).
    - CanonicalProjector: Topology-consistent manifold projection.
    - AffordanceGradients: Mapping lore to formal symbols and algorithmic density.
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
        
        # Standardized Processing Pipeline
        self.filter = TextbookFilter()
        self.projector = CanonicalProjector(dim=engine_dim, device=device)
        self.processor = ConversationalDataProcessor(device=device)

    def _wait_for_rate_limit(self):
        """Ensures compliance with ArXiv's anti-crawling policies."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

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
            
            admitted_count = 0
            rejected_count = 0
            
            for record in records[:5]: # Cap per pull to maintain non-teleological drift
                metadata = record.find('.//oai:record', self.ns) # Nested find
                dc = record.find('.//oai_dc:dc', self.ns)
                
                if dc is not None:
                    title_elem = dc.find('dc:title', self.ns)
                    desc_elem = dc.find('dc:description', self.ns)
                    id_elem = dc.find('dc:identifier', self.ns)
                    
                    title = title_elem.text if title_elem is not None else "Unknown Title"
                    abstract = desc_elem.text if desc_elem is not None else "No Abstract"
                    arxiv_id = id_elem.text if id_elem is not None else "No ID"
                    
                    full_content = f"Title: {title}\nAbstract: {abstract}"
                    
                    # 1. Quality Gating (Structural Honesty & Textbook Standards)
                    report = self.filter.assess(full_content, source=f"arxiv_{arxiv_id}")
                    
                    if not report.is_admissible:
                        rejected_count += 1
                        print(f" [LORE] Rejected: {title[:40]}... (Flags: {', '.join(report.flags)})")
                        continue
                    
                    # 2. Canonical Manifold Projection
                    proj = self.projector.project_text_to_state(full_content)
                    residue = proj['state'] # [1, engine_dim]
                    entropy = proj['entropy']
                    
                    # 3. Affordance Gradient Computation
                    gradients = self.processor.compute_affordance_gradients(full_content)
                    
                    # 4. Fossilization with full metadata
                    # We use a zero fingerprint for text-only lore
                    dyad = KnowledgeDyad(
                        image_fingerprint=torch.zeros(137, device=self.device),
                        linguistic_description=title,
                        relevance_score=float(report.instructive), # Use instructor score as relevance
                        metadata={
                            'arxiv_id': arxiv_id,
                            'abstract_preview': abstract[:200],
                            'quality': report.to_dict(),
                            'affordance_gradients': gradients,
                            'gyroid_entropy': entropy
                        }
                    )
                    
                    self.fossilizer.fossilize(dyad, residue)
                    admitted_count += 1
                    
                    # Descriptive status log
                    q_str = f"I:{report.instructive:.2f} A:{report.algorithmic:.2f} S:{report.structural_honesty:.2f}"
                    print(f" [LORE] Fossilized: {title[:50]}... ({q_str})")
            
            if admitted_count > 0:
                print(f"[INGEST] Successfully anchored {admitted_count} lore residues. Rejected {rejected_count} below threshold.")
        except Exception as e:
            print(f"[INGEST] Parsing error: {e}")

    def start_sovereign_loop(self):
        """Starts the background ingestion thread."""
        def _loop():
            # Cycle through high-density mathematical and AI sets
            sets = ["math", "physics:quant-ph", "cs:AI", "math.LO", "math.HO"]
            while True:
                for s in sets:
                    try:
                        self.ingest_latest_math(s)
                    except Exception as e:
                        print(f"[INGEST] Loop error in set {s}: {e}")
                    # Large gap between sets to respect community resources
                    time.sleep(60)
                # Sleep for 1 hour after cycling all sets
                time.sleep(3600)
                
        bg_thread = threading.Thread(target=_loop, daemon=True)
        bg_thread.start()
        print("[INGEST] ArXiv Sovereign Ingestor active. Quality-gated lore capture in progress.")
