import time
import requests
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable, Any
import datetime
import threading
import tarfile
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core.knowledge_dyad_fossilizer import KnowledgeDyad, DyadFossilizer
from src.data.textbook_filter import TextbookFilter
from src.data.canonical_projection import CanonicalProjector
from src.data.conversational_api_ingestor import ConversationalDataProcessor
from src.ui.diegetic_visualizer import _chebyshev_project_np

def _honest_randint(low: int, high: int, device: str = 'cpu') -> int:
    if low >= high:
        return low
    from src.core.honest_jitter import harvest_honest_jitter
    jitter = harvest_honest_jitter((1,), device=torch.device(device), scaled=False).item()
    u = (jitter + 1.0) / 2.0
    val = low + int(u * (high - low + 1))
    return min(val, high)

def _honest_choice(options: list, device: str = 'cpu') -> Any:
    if not options:
        return None
    from src.core.honest_jitter import harvest_honest_jitter
    jitter = harvest_honest_jitter((1,), device=torch.device(device), scaled=False).item()
    u = (jitter + 1.0) / 2.0
    idx = int(u * len(options))
    return options[min(idx, len(options) - 1)]

class ArXivSovereignIngestor:
    """
    Implements a 'Slow-Drip' non-teleological knowledge ingestor using ArXiv OAI-PMH and Search APIs.
    Complies with the 3-second rate limit to ensure non-invasive learning.
    
    Retrofitted with:
    - TextbookFilter: Multi-dimensional quality gating (Structural Honesty).
    - CanonicalProjector: Topology-consistent manifold projection.
    - AffordanceGradients: Mapping lore to formal symbols and algorithmic density.
    """
    def __init__(self, fossilizer: DyadFossilizer, engine_dim: int, device: str = 'cpu', state_callback: Optional[Callable] = None, engine: Optional[Any] = None):
        self.fossilizer = fossilizer
        self.engine_dim = engine_dim
        self.device = device
        self.state_callback = state_callback
        self.engine = engine
        self.base_url = "http://export.arxiv.org/oai2"
        self.last_request_time = 0
        self.rate_limit_seconds = 4.0 # Conservatively above the 3s requirement
        self._engine_busy_fn = None
        
        # NS Map for ArXiv OAI-PMH
        self.ns = {
            'oai': 'http://www.openarchives.org/OAI/2.0/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'
        }
        
        # Standardized Processing Pipeline
        self.filter = TextbookFilter()
        self.projector = CanonicalProjector(dim=engine_dim, device=self.device)
        self.processor = ConversationalDataProcessor(device=self.device)

    def _wait_for_rate_limit(self):
        """Ensures compliance with ArXiv's anti-crawling policies."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def ingest_latest_math(self, set_name: str = "math", commutativity: str = 'symmetric'):
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
                self._parse_and_fossilize(response.text, commutativity)
            else:
                print(f"[INGEST] Failed to reach ArXiv (HTTP {response.status_code}). Manifold remains local.")
        except Exception as e:
            print(f"[INGEST] Transport error: {e}. Ingestion suspended.")

    def _extract_media_from_eprint(self, arxiv_id: str) -> List[bytes]:
        """Downloads the ArXiv source tarball and extracts embedded images (max 10)."""
        url = f"https://export.arxiv.org/e-print/{arxiv_id}"
        images = []
        try:
            # Respect ArXiv's rate limits for source downloads
            self._wait_for_rate_limit()
            response = requests.get(url, stream=True, timeout=15)
            if response.status_code == 200:
                # We do this in-memory to prevent disk pollution
                with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            f = tar.extractfile(member)
                            if f:
                                images.append(f.read())
                                if len(images) >= 10: # Capped to prevent OOM
                                    break
        except Exception as e:
            print(f" [LORE] Source extraction failed for {arxiv_id}: {e}")
        return images

    def _compute_multimodal_fingerprint(self, image_bytes_list: List[bytes]) -> torch.Tensor:
        """Computes a 96-dim Chebyshev spectral signature averaged across all extracted images."""
        if not image_bytes_list:
            return torch.zeros(96, device=self.device)
            
        all_fingerprints = []
        for img_bytes in image_bytes_list:
            try:
                buf = io.BytesIO(img_bytes)
                rgba = plt.imread(buf)
                
                # BT.601 decomposition
                if rgba.ndim == 2:
                    lum = rgba.astype(np.float64)
                    cr = np.zeros_like(lum)
                    cb = np.zeros_like(lum)
                else:
                    if rgba.shape[2] > 3:
                        rgba = rgba[:, :, :3]
                    r, g, b = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2]
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
                    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
                
                # Compute K=32 modes for each channel
                K = 32
                l_c = _chebyshev_project_np(lum.flatten().astype(np.float64), K)
                cr_c = _chebyshev_project_np(cr.flatten().astype(np.float64), K)
                cb_c = _chebyshev_project_np(cb.flatten().astype(np.float64), K)
                
                fp = l_c + cr_c + cb_c # 96-dim list
                all_fingerprints.append(fp)
            except Exception as e:
                continue
                
        if not all_fingerprints:
            return torch.zeros(96, device=self.device)
            
        # Average the Chebyshev coefficients
        mean_fp = np.mean(all_fingerprints, axis=0)
        return torch.tensor(mean_fp, dtype=torch.float32, device=self.device)

    def _resolve_seed_state(self, content_key: str) -> Optional[torch.Tensor]:
        """Resolves seed state dynamically or generates a deterministic pseudo-state for quasi-headless mode."""
        seed_state = None
        # 1. Try dynamic callback
        if self.state_callback is not None:
            try:
                seed_state = self.state_callback()
            except Exception:
                pass
        # 2. Try engine's live meta state
        if seed_state is None and self.engine is not None:
            try:
                seed_state = getattr(self.engine, 'meta_state', None)
            except Exception:
                pass
        # 3. Standalone/Headless Fallback: generate a deterministic category/content key signature
        if seed_state is None:
            try:
                import hashlib
                h = hashlib.sha256(content_key.encode('utf-8')).digest()
                seed_val = int.from_bytes(h[:4], byteorder='big')
                g = torch.Generator(device=self.device)
                g.manual_seed(seed_val)
                seed_state = torch.randn(self.engine_dim, device=self.device, generator=g)
                seed_state = seed_state / (seed_state.norm() + 1e-8)
            except Exception as e:
                print(f"[INGEST] Failed to generate deterministic pseudo-seed: {e}")
        return seed_state

    def _parse_and_fossilize(self, xml_text: str, commutativity: str):
        """Parses OAI-PMH XML and converts records into permanent knowledge fossils."""
        acquired = False
        if self.engine is not None and hasattr(self.engine, '_processing_lock'):
            acquired = self.engine._processing_lock.acquire(timeout=180.0)
            if not acquired:
                print("[INGEST] Warning: Failed to acquire lock for OAI-PMH parsing. Bypassing.")
                return
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
                    
                    # 4. Multimodal Fingerprint Extraction (The ArXiv Bitstream Upgrade)
                    # We download the LaTeX source tarball and extract its visual media
                    img_bytes_list = self._extract_media_from_eprint(arxiv_id)
                    multimodal_fingerprint = self._compute_multimodal_fingerprint(img_bytes_list)
                    
                    if len(img_bytes_list) > 0:
                        print(f" [MULTIMODAL] Extracted {len(img_bytes_list)} images for {arxiv_id}. Fingerprint embedded.")
                        
                    # 5. Fossilization with full metadata
                    dyad = KnowledgeDyad(
                        image_fingerprint=multimodal_fingerprint,
                        linguistic_description=title,
                        relevance_score=float(report.instructive), # Use instructor score as relevance
                        metadata={
                            'arxiv_id': arxiv_id,
                            'abstract_preview': abstract[:200],
                            'quality': report.to_dict(),
                            'affordance_gradients': gradients,
                            'gyroid_entropy': entropy,
                            'commutativity': commutativity,
                            'media_count': len(img_bytes_list)
                        }
                    )
                    
                    seed_state = self._resolve_seed_state(title)
                    self.fossilizer.fossilize(dyad, residue, seed_state=seed_state)
                    admitted_count += 1
                    
                    # Descriptive status log
                    media_str = f"| MEDIA: {len(img_bytes_list)}" if len(img_bytes_list) > 0 else ""
                    q_str = f"I:{report.instructive:.2f} A:{report.algorithmic:.2f} S:{report.structural_honesty:.2f} {media_str}"
                    print(f" [LORE] Fossilized: {title[:50]}... ({q_str})")
            
            if admitted_count > 0:
                print(f"[INGEST] Successfully anchored {admitted_count} lore residues. Rejected {rejected_count} below threshold.")
        except Exception as e:
            print(f"[INGEST] Parsing error: {e}")
        finally:
            if acquired:
                self.engine._processing_lock.release()

    def _get_category_signature(self, name: str) -> torch.Tensor:
        """Generates a deterministic, category-specific archetype signature vector in engine space."""
        import hashlib
        h = hashlib.sha256(name.encode('utf-8')).digest()
        seed = int.from_bytes(h[:4], byteorder='big')
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        v = torch.randn(self.engine_dim, device=self.device, generator=g)
        return v / (torch.norm(v) + 1e-8)

    def ingest_arxiv_by_query(self, query_str: str, commutativity: str = 'symmetric'):
        """Queries ArXiv search API with a larynx-generated query string and fossilizes matches."""
        self._wait_for_rate_limit()
        # Clean query: only alphanumeric and spaces
        cleaned_query = "".join(c if c.isalnum() or c.isspace() else "" for c in query_str).strip()
        if not cleaned_query:
            print("[INGEST] Cleaned query is empty. Skipping search.")
            return
        
        # Replace consecutive spaces with a single space
        cleaned_query = " ".join(cleaned_query.split())
        query_param = "+".join(cleaned_query.split())
        
        random_offset = _honest_randint(0, 30, device=self.device)
        url = f"http://export.arxiv.org/api/query?search_query=all:{query_param}&sortBy=submittedDate&sortOrder=descending&start={random_offset}&max_results=5"
        
        try:
            print(f"[INGEST] Performing character-level search on ArXiv for: '{cleaned_query}'...")
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                self._parse_and_fossilize_atom(response.text, cleaned_query, commutativity)
            else:
                print(f"[INGEST] Search query failed (HTTP {response.status_code}).")
        except Exception as e:
            print(f"[INGEST] Search transport error: {e}. Ingestion suspended.")

    def _parse_and_fossilize_atom(self, xml_text: str, query: str, commutativity: str):
        """Parses ArXiv Atom search API XML and converts entries into permanent knowledge fossils."""
        acquired = False
        if self.engine is not None and hasattr(self.engine, '_processing_lock'):
            acquired = self.engine._processing_lock.acquire(timeout=180.0)
            if not acquired:
                print("[INGEST] Warning: Failed to acquire lock for Atom parsing. Bypassing.")
                return
        try:
            root = ET.fromstring(xml_text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('.//atom:entry', ns)
            
            admitted_count = 0
            rejected_count = 0
            
            for entry in entries[:5]:
                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                id_elem = entry.find('atom:id', ns)
                
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Unknown Title"
                # Strip excessive whitespace/newlines from abstract
                abstract = summary_elem.text.strip() if summary_elem is not None and summary_elem.text else "No Abstract"
                abstract = " ".join(abstract.split())
                
                arxiv_url = id_elem.text.strip() if id_elem is not None and id_elem.text else "No ID"
                # Extract arxiv_id from url
                arxiv_id = arxiv_url.split('/abs/')[-1] if '/abs/' in arxiv_url else arxiv_url
                
                full_content = f"Title: {title}\nAbstract: {abstract}"
                
                # 1. Quality Gating (Structural Honesty & Textbook Standards)
                report = self.filter.assess(full_content, source=f"arxiv_query_{arxiv_id}")
                
                if not report.is_admissible:
                    rejected_count += 1
                    print(f" [LORE] Rejected query match: {title[:40]}... (Flags: {', '.join(report.flags)})")
                    continue
                
                # 2. Canonical Manifold Projection
                proj = self.projector.project_text_to_state(full_content)
                residue = proj['state']
                entropy = proj['entropy']
                
                # 3. Affordance Gradient Computation
                gradients = self.processor.compute_affordance_gradients(full_content)
                
                # 4. Multimodal Fingerprint Extraction
                img_bytes_list = self._extract_media_from_eprint(arxiv_id)
                multimodal_fingerprint = self._compute_multimodal_fingerprint(img_bytes_list)
                
                if len(img_bytes_list) > 0:
                    print(f" [MULTIMODAL] Extracted {len(img_bytes_list)} images for {arxiv_id}. Fingerprint embedded.")
                    
                # 5. Fossilization with full metadata
                dyad = KnowledgeDyad(
                    image_fingerprint=multimodal_fingerprint,
                    linguistic_description=title,
                    relevance_score=float(report.instructive),
                    metadata={
                        'arxiv_id': arxiv_id,
                        'abstract_preview': abstract[:200],
                        'query_used': query,
                        'quality': report.to_dict(),
                        'affordance_gradients': gradients,
                        'gyroid_entropy': entropy,
                        'commutativity': commutativity,
                        'media_count': len(img_bytes_list)
                    }
                )
                
                seed_state = self._resolve_seed_state(title)
                self.fossilizer.fossilize(dyad, residue, seed_state=seed_state)
                admitted_count += 1
                
                media_str = f"| MEDIA: {len(img_bytes_list)}" if len(img_bytes_list) > 0 else ""
                q_str = f"I:{report.instructive:.2f} A:{report.algorithmic:.2f} S:{report.structural_honesty:.2f} {media_str}"
                print(f" [LORE] Fossilized search match for '{query}': {title[:50]}... ({q_str})")
                
            if admitted_count > 0:
                print(f"[INGEST] Successfully anchored {admitted_count} query-based lore residues. Rejected {rejected_count} below threshold.")
        except Exception as e:
            print(f"[INGEST] Atom parsing error: {e}")
        finally:
            if acquired:
                self.engine._processing_lock.release()

    def _get_dynamic_fallback(self) -> str:
        """Dynamically extracts query terms from historical memory fossils to guide search."""
        try:
            fossils = self.fossilizer.recover_fossils(limit=50)
            if fossils:
                for _ in range(15):
                    chosen = _honest_choice(fossils, device=self.device)
                    desc = chosen.get('text_input') or chosen.get('description', '')
                    if not desc:
                        continue
                    # Tokenize and clean
                    words = [w.strip(".,!?;:()[]{}'\"") for w in desc.split()]
                    words = [w for w in words if len(w) > 4 and w.isalpha() and w.lower() not in [
                        "about", "their", "there", "would", "could", "should", "under", "which",
                        "these", "those", "other", "after", "before", "using", "first", "second"
                    ]]
                    if len(words) >= 2:
                        idx = _honest_randint(0, len(words) - 2, device=self.device)
                        query = f"{words[idx]} {words[idx+1]}"
                        return query
                    elif len(words) == 1:
                        return words[0]
        except Exception as e:
            print(f"[INGEST] Dynamic fallback extraction failed: {e}")
            
        # Hardcore physical/topological concepts matching our mathematical framework as ultimate default
        default_concepts = [
            "Chebyshev polynomial", "Birkhoff polytope", "Chern Simons Gasket",
            "Drucker Prager yield", "Mohr Coulomb", "Wasserstein distance",
            "sine Gordon soliton", "homology Betti number", "non Hermitian flow"
        ]
        
        # Project active state onto default concepts via cosine similarity
        current_state = None
        if self.state_callback is not None:
            try:
                current_state = self.state_callback()
            except Exception:
                pass
        if current_state is None and self.engine is not None:
            cavity = getattr(self.engine, 'cavity', None) or getattr(self.engine, 'resonance_cavity', None)
            if cavity is not None and hasattr(cavity, 'M'):
                try:
                    M = cavity.M
                    norms = torch.norm(M, dim=-1)
                    max_idx = torch.argmax(norms)
                    k_idx = (max_idx // M.shape[1]).item()
                    m_idx = (max_idx % M.shape[1]).item()
                    current_state = M[k_idx, m_idx]
                except Exception:
                    pass

        if current_state is not None:
            try:
                flat_state = current_state.flatten()
                if flat_state.shape[0] > self.engine_dim:
                    flat_state = flat_state[:self.engine_dim]
                elif flat_state.shape[0] < self.engine_dim:
                    padding = torch.zeros(self.engine_dim - flat_state.shape[0], device=self.device)
                    flat_state = torch.cat([flat_state, padding])

                norm_state = flat_state / (torch.norm(flat_state) + 1e-8)

                scores = []
                for s in default_concepts:
                    sig = self._get_category_signature(s)
                    sim = torch.dot(norm_state, sig).item()
                    scores.append(sim)

                scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device) / 0.2
                from src.core.honest_jitter import honest_multinomial
                probs = torch.softmax(scores_t, dim=0)
                idx = honest_multinomial(probs, 1).item()
                return default_concepts[idx]
            except Exception as e:
                print(f"[INGEST] Dynamic fallback steering failed: {e}")

        return _honest_choice(default_concepts, device=self.device)

    def _generate_larynx_query(self) -> str:
        """Uses the engine's larynx autoregressively to generate a search query from the current meta_state."""
        if self.engine is None or not hasattr(self.engine, 'larynx'):
            return self._get_dynamic_fallback()
            
        acquired = False
        if hasattr(self.engine, '_processing_lock'):
            acquired = self.engine._processing_lock.acquire(timeout=5.0)
            if not acquired:
                return self._get_dynamic_fallback()
            
        try:
            # Temporarily flag that engine is generating background search terms
            old_processing = getattr(self.engine, '_is_processing', False)
            self.engine._is_processing = True
            
            # Start with current meta_state clone or ResonanceCavity active mode
            if self.state_callback is not None:
                current_state = self.state_callback().clone().detach()
            else:
                # Try to initialize current_state from the ResonanceCavity active mode vector
                cavity = getattr(self.engine, 'cavity', None) or getattr(self.engine, 'resonance_cavity', None)
                if cavity is not None and hasattr(cavity, 'M'):
                    M = cavity.M # [K, num_modes, hidden_dim]
                    norms = torch.norm(M, dim=-1) # [K, num_modes]
                    max_idx = torch.argmax(norms)
                    k_idx = (max_idx // M.shape[1]).item()
                    m_idx = (max_idx % M.shape[1]).item()
                    current_state = M[k_idx, m_idx].unsqueeze(0).clone().detach()
                else:
                    current_state = torch.zeros((1, self.engine_dim), device=self.device)
                
            larynx = self.engine.larynx
            larynx.eval()
            
            generated_chars = []
            max_len = 30
            temp = 1.2  # slightly higher temperature for query exploration
            
            with torch.no_grad():
                for _ in range(max_len):
                    logits, conf = larynx(current_state, temperature=temp)
                    probs = torch.softmax(logits, dim=-1)
                    char_idx = torch.multinomial(probs[0], 1).item()
                    
                    char = chr(max(32, min(126, char_idx)))
                    if char in ('.', '!', '?', ';', '\n'):
                        break
                    generated_chars.append(char)
                    
                    # Update state
                    feedback = torch.tanh(larynx.proj.weight[char_idx].unsqueeze(0))
                    current_state = 0.9 * current_state + 0.1 * feedback
                    
            query = "".join(generated_chars).strip()
            # Clean up the query to only alphanumeric characters and spaces
            query = "".join(c for c in query if c.isalnum() or c.isspace())
            query = " ".join(query.split())
            
            if len(query) < 3:
                query = self._get_dynamic_fallback()
                
            return query
        except Exception as e:
            print(f"[INGEST] Larynx query generation failed: {e}")
            return self._get_dynamic_fallback()
        finally:
            if self.engine is not None:
                self.engine._is_processing = old_processing
            if acquired:
                self.engine._processing_lock.release()

    def start_sovereign_loop(self):
        """Starts the background ingestion thread with dynamic Meta-State Topic Steering."""
        def _loop():
            # Complete category corpus covering heavy science + humanities/social overlaps
            sets = [
                # Hard Science / Logic / Topology
                "math", "physics:quant-ph", "cs:AI", "math.LO", "math.HO", 
                # Humanities & Societal Overlaps (Harder to find, but critical)
                "physics:hist-ph",           # History and Philosophy of Physics (Deep Philosophy)
                "cs:CY",                     # Computers and Society (Digital Humanities/Ethics)
                "physics:physics.soc-ph",    # Sociophysics (Mathematical Sociology)
                "cs:CL",                     # Computation and Language (Computational Linguistics / Philosophy)
                "q-bio.NC",                  # Neurons and Cognition (Cognitive Science)
                "cs:HC",                     # Human-Computer Interaction (Sociotechnical)
                "econ:TH",                   # Theoretical Economics
                "q-fin:GN",                  # General Finance (Economic Humanities)
            ]
            
            cycle = 0
            while True:
                cycle += 1
                selected_set = "math" # Default fallback
                try:
                    # Alternate between set list (OAI-PMH) and search query (Atom API)
                    if cycle % 2 == 0:
                        query = self._generate_larynx_query()
                        print(f" [INGEST] Larynx generated search query: '{query}'")
                        self.ingest_arxiv_by_query(query)
                    else:
                        # Check if we have meta-state steering active
                        current_state = None
                        if self.state_callback is not None:
                            try:
                                current_state = self.state_callback()
                            except Exception as e:
                                print(f"[INGEST] Meta-state callback failed: {e}. Reverting to uniform.")
                        
                        if current_state is not None and isinstance(current_state, torch.Tensor):
                            # Standardize shape
                            flat_state = current_state.flatten()
                            if flat_state.shape[0] > self.engine_dim:
                                flat_state = flat_state[:self.engine_dim]
                            elif flat_state.shape[0] < self.engine_dim:
                                padding = torch.zeros(self.engine_dim - flat_state.shape[0], device=self.device)
                                flat_state = torch.cat([flat_state, padding])
                            
                            norm_state = flat_state / (torch.norm(flat_state) + 1e-8)
                            
                            # Compute similarities with deterministic category archetypes
                            scores = []
                            for s in sets:
                                sig = self._get_category_signature(s)
                                sim = torch.dot(norm_state, sig).item()
                                scores.append(sim)
                            
                            # Softmax with temperature=0.2 to sample next steering category
                            scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device) / 0.2
                            probs = torch.softmax(scores_t, dim=0)
                            
                            # Sample category
                            from src.core.honest_jitter import honest_multinomial
                            idx = honest_multinomial(probs, 1).item()
                            selected_set = sets[idx]
                            print(f" [INGEST] Meta-State Topic Steering selected topic: '{selected_set}' (prob: {probs[idx].item():.3f})")
                        else:
                            selected_set = _honest_choice(sets, device=self.device)
                            print(f" [INGEST] Dynamic loop selected uniform topic: '{selected_set}'")
                        
                        # Run ingestion
                        self.ingest_latest_math(selected_set)
                        
                except Exception as e:
                    print(f"[INGEST] Loop steering/search error: {e}")
                
                # Slow-drip timing between pulls - adaptive sleep interval
                is_busy = hasattr(self, '_engine_busy_fn') and self._engine_busy_fn is not None and self._engine_busy_fn()
                sleep_sec = 300 if is_busy else 60
                time.sleep(sleep_sec)
                
        bg_thread = threading.Thread(target=_loop, daemon=True)
        bg_thread.start()
        print(" [INGEST] ArXiv Sovereign Ingestor ACTIVE with Dynamic Meta-State Steering and Larynx Search.")
        print(" [INGEST] Background monitoring active. Science and Humanities inclusion online.")
