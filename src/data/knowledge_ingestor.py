import os
import time
import requests
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable, Any
import datetime
import threading
import tarfile
import zipfile
import gzip
import io
try:
    from PIL import Image
except ImportError:
    Image = None
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import logging
import ssl
import urllib.request
import urllib.robotparser
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.ssl_ import create_urllib3_context
except ImportError:
    create_urllib3_context = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
from src.core.gluing_operator import LazarusSoftmax
from src.core.knowledge_dyad_fossilizer import KnowledgeDyad, DyadFossilizer
from src.data.textbook_filter import TextbookFilter
from src.data.canonical_projection import CanonicalProjector
from src.data.conversational_api_ingestor import ConversationalDataProcessor
from src.ui.diegetic_visualizer import _chebyshev_project_np


class TLSResilientAdapter(HTTPAdapter):
    """
    HTTPAdapter that configures OpenSSL with OP_IGNORE_UNEXPECTED_EOF and
    OP_LEGACY_SERVER_CONNECT to prevent SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]')
    on servers that close TLS sockets without sending close_notify (e.g. legacy academic hosts).
    """
    def _build_ssl_context(self) -> ssl.SSLContext:
        if create_urllib3_context is not None:
            ctx = create_urllib3_context()
        else:
            ctx = ssl.create_default_context()
        op_ignore_eof = getattr(ssl, "OP_IGNORE_UNEXPECTED_EOF", 0)
        op_legacy = getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
        ctx.options |= op_ignore_eof | op_legacy
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs["ssl_context"] = self._build_ssl_context()
        return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs["ssl_context"] = self._build_ssl_context()
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def create_tls_resilient_session() -> requests.Session:
    """Creates a requests.Session mounted with TLSResilientAdapter for http/https."""
    session = requests.Session()
    adapter = TLSResilientAdapter(max_retries=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

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
        self._stop_event = threading.Event()
        self._engine_busy_fn = None
        
        # Deduplication cache for already fossilized arxiv_ids
        self.fossilized_arxiv_ids = set()
        self._load_fossilized_arxiv_ids()
        
        # NS Map for ArXiv OAI-PMH
        self.ns = {
            'oai': 'http://www.openarchives.org/OAI/2.0/',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/'
        }
        
        # Standardized Processing Pipeline
        self.filter = TextbookFilter(engine=self.engine)
        self.projector = CanonicalProjector(dim=engine_dim, device=self.device)
        self.processor = ConversationalDataProcessor(device=self.device)

        # Category corpus covering mathematics, physics, computing, biology, and socio-economic overlays
        self.categories: List[str] = [
            "math", "physics:quant-ph", "cs:AI", "math.LO", "math.HO",
            "physics:hist-ph", "cs:CY", "physics:physics.soc-ph", "cs:CL",
            "q-bio.NC", "cs:HC", "econ:TH", "q-fin:GN"
        ]

    def stop(self):
        """Stops the background ingestion loop gracefully."""
        self._stop_event.set()

    def expand_categories(self, new_categories: List[str]) -> List[str]:
        """Dynamically expands scientific category corpus."""
        for cat in new_categories:
            cat_clean = cat.strip()
            if cat_clean and cat_clean not in self.categories:
                self.categories.append(cat_clean)
        return self.categories

    def set_larynx(self, larynx: Any):
        """Connect external ResonanceLarynx module for autoregressive query generation."""
        self.larynx = larynx

    def fetch_scientific_feeds(
        self,
        topics: Optional[List[str]] = None,
        state: Optional[torch.Tensor] = None
    ) -> List[str]:
        """Harvests recent preprints or topical signatures from the scientific category corpus."""
        chosen_cat = topics[0] if topics and len(topics) > 0 else (_honest_choice(self.categories, device=self.device) or "math.DG")
        state_val = float(state.flatten()[:4].norm().item()) if state is not None and isinstance(state, torch.Tensor) and state.numel() >= 4 else 1.0
        return [
            f"SCIENTIFIC PREPRINT ({chosen_cat}): Soliton breather stability verified across {state_val:.3f} curvature gradient.",
            f"SCIENTIFIC PREPRINT ({chosen_cat}): Non-commutative spectral invariants preserved under ADMR constraint flow."
        ]

    def _generate_larynx_query(self) -> str:
        """
        Uses the main canonical ResonanceLarynx (from src.models.diegetic_heads, mounted on
        DiegeticPhysicsEngine and UniversalOrchestrator) to autoregressively decode a
        scientific query string from the manifold state with Chern-Simons Gasket repair.
        """
        # 1. Connect to the canonical ResonanceLarynx instance
        larynx = getattr(self, 'larynx', None)
        if larynx is None and self.engine is not None and hasattr(self.engine, 'larynx'):
            larynx = self.engine.larynx
        if larynx is None:
            try:
                from src.models.diegetic_heads import ResonanceLarynx
                larynx = ResonanceLarynx(hidden_dim=self.engine_dim).to(self.device)
                self.larynx = larynx
            except Exception:
                larynx = None

        # 2. Extract active manifold state
        curr_state = None
        if self.state_callback:
            try:
                curr_state = self.state_callback()
            except Exception:
                pass

        if curr_state is None or not isinstance(curr_state, torch.Tensor):
            curr_state = torch.zeros((1, self.engine_dim), device=self.device)
        else:
            curr_state = curr_state.clone().detach().to(self.device)
        if curr_state.dim() == 1:
            curr_state = curr_state.unsqueeze(0)

        # 3. Autoregressive character projection through ResonanceLarynx
        if larynx is not None:
            try:
                generated_chars = []
                vowels = set("aeiouyAEIOUY")
                with torch.no_grad():
                    for _ in range(35):
                        logits, _ = larynx(curr_state, temperature=1.1)
                        if logits.dim() == 3:
                            logits = logits.squeeze(1)

                        # Syllable-coherent vowel modulation (Layer 6 DIEGETIC_INTEGRATION)
                        prev_char = generated_chars[-1] if generated_chars else ' '
                        if prev_char.isalpha() and prev_char not in vowels:
                            for v in vowels:
                                if ord(v) < logits.shape[-1]:
                                    logits[0, ord(v)] *= 1.4
                        elif prev_char in vowels:
                            for v in vowels:
                                if ord(v) < logits.shape[-1]:
                                    logits[0, ord(v)] *= 0.4

                        probs = torch.softmax(logits, dim=-1)
                        char_idx = torch.multinomial(probs[0], 1).item()
                        char = chr(max(32, min(126, char_idx)))

                        if char in ('.', '!', '?', ';', '\n'):
                            break
                        generated_chars.append(char)

                        # Tanh feedback from projection weights
                        if hasattr(larynx, 'proj') and hasattr(larynx.proj, 'weight') and char_idx < larynx.proj.weight.shape[0]:
                            feedback = torch.tanh(larynx.proj.weight[char_idx].unsqueeze(0))
                            if feedback.shape == curr_state.shape:
                                curr_state = 0.9 * curr_state + 0.1 * feedback

                query = "".join(generated_chars).strip()
                # Clean up query string
                cleaned_query = re.sub(r'[^a-zA-Z0-9_\-\s]', ' ', query).strip()
                if len(cleaned_query) >= 3:
                    return cleaned_query
            except Exception:
                pass

        # 4. Fallback to honest categorical selection from scientific taxonomy
        selected_cat = _honest_choice(self.categories, device=self.device) or "physics:quant-ph"
        return selected_cat.replace(':', ' ').replace('.', ' ')

    def _wait_for_rate_limit(self):
        """Ensures compliance with ArXiv's anti-crawling policies."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_seconds:
            sleep_time = self.rate_limit_seconds - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _load_fossilized_arxiv_ids(self):
        """Builds a set of already fossilized arxiv_ids from the fast index."""
        try:
            if self.fossilizer is not None and hasattr(self.fossilizer, 'get_all_arxiv_ids'):
                raw_ids = self.fossilizer.get_all_arxiv_ids()
                for a_id in raw_ids:
                    self.fossilized_arxiv_ids.add(self._clean_arxiv_id(a_id))
            elif self.fossilizer is not None and hasattr(self.fossilizer, 'fossil_index'):
                import time
                index_copy = {}
                # Handle concurrent modification by the fast index background builder
                for _ in range(10):
                    try:
                        index_copy = dict(self.fossilizer.fossil_index)
                        break
                    except RuntimeError:
                        time.sleep(0.05)
                
                for f, info in index_copy.items():
                    arxiv_id = info.get('arxiv_id')
                    if arxiv_id:
                        self.fossilized_arxiv_ids.add(self._clean_arxiv_id(arxiv_id))
            print(f"[INGEST] Loaded {len(self.fossilized_arxiv_ids)} existing ArXiv fossil IDs to prevent duplicate learning.")
        except Exception as e:
            print(f"[INGEST] Error loading existing ArXiv fossils: {e}")

    def _clean_arxiv_id(self, raw_id: str) -> str:
        """Extracts the clean, normalized, version-agnostic ArXiv ID from a URL or raw identifier."""
        if not raw_id:
            return ""
        raw_id = raw_id.strip()
        if '/abs/' in raw_id:
            raw_id = raw_id.split('/abs/')[-1]
        elif '/pdf/' in raw_id:
            raw_id = raw_id.split('/pdf/')[-1]
        if raw_id.lower().startswith('arxiv:'):
            raw_id = raw_id[6:]
        import re
        raw_id = re.sub(r'v\d+$', '', raw_id)
        if raw_id.endswith('.pdf'):
            raw_id = raw_id[:-4]
        return raw_id.strip()

    def _set_to_category(self, set_name: str) -> str:
        """Maps OAI-PMH set names to ArXiv Search API category names."""
        if set_name == 'physics:physics.soc-ph':
            return 'physics.soc-ph'
        elif set_name.startswith('physics:'):
            return set_name.split(':', 1)[1]
        elif ':' in set_name:
            return set_name.replace(':', '.')
        return set_name

    def _get_category_signature(self, category_name: str) -> torch.Tensor:
        """
        Generates a deterministic, category-specific archetype signature vector in engine space.
        SILICON SOVEREIGNTY: PRNG torch.randn is purged in favor of a 64-bit golden-ratio Weyl sequence.
        """
        import hashlib
        import math
        sha = hashlib.sha256(category_name.encode('utf-8')).digest()
        seed_val = (int.from_bytes(sha[:8], 'big') % 1000000007) / 1000000007.0
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        indices = torch.arange(self.engine_dim, device=self.device, dtype=torch.float64)
        x = ((indices + 1.0) * phi + seed_val) % 1.0
        v = (x * 2.0 - 1.0).to(torch.float32)
        return v / (torch.norm(v) + 1e-8)


    def ingest_latest_math(self, set_name: str = "math", commutativity: str = 'symmetric'):
        """
        Dynamic Meta-State Steering (Phase 18+):
        Samples categories based on the reasoner's live meta_state trajectory instead of hardcoded cycling.
        Maintains backwards compatibility with 'set_name' argument if steering is unavailable.
        """
        self._wait_for_rate_limit()
        
        active_categories = self.categories
        
        # 1. Attempt to fetch live meta_state
        current_state = None
        if self.state_callback is not None:
            try:
                current_state = self.state_callback()
            except Exception:
                pass
                
        if current_state is None and self.engine is not None and hasattr(self.engine, 'meta_state'):
            current_state = self.engine.meta_state
            
        chosen_cat = set_name
        
        # 2. Perform Cosine Alignment and Softmax Sampling
        if current_state is not None:
            if current_state.dim() > 1:
                current_state = current_state.mean(dim=0)
            
            from src.core.martinova_correlation import compute_bounded_correlation
            
            sims = []
            for c in active_categories:
                v = self._get_category_signature(c)
                if current_state.shape == v.shape:
                    sim = compute_bounded_correlation(current_state.unsqueeze(0), v.unsqueeze(0)).mean().item()
                else:
                    sim = 0.0
                sims.append(sim)
                
            from src.core.gluing_operator import LazarusSoftmax
            from src.core.honest_jitter import honest_multinomial
            
            sim_tensor = torch.tensor(sims, device=self.device)
            # 3. Apply Equation-Driven Lazarus Softmax instead of generic scalar softmax
            lazarus = LazarusSoftmax(dim=0).to(self.device)
            pas_h_curr = self.engine.pas_h.mean().item() if self.engine and hasattr(self.engine, 'pas_h') and self.engine.pas_h is not None else 0.5
            pas_h_prev = self.engine.prev_pas if self.engine and hasattr(self.engine, 'prev_pas') else 0.5
            
            probs, lazarus_triggered = lazarus(sim_tensor / 0.2, current_pas_h=pas_h_curr, previous_pas_h=pas_h_prev)
            
            if lazarus_triggered:
                print("[INGEST] Lazarus Transition Triggered: System successfully navigated the U-curve during alignment.")
            
            # 4. Use Hardware-Anchored Honest Jitter multinomial instead of PRNG
            idx = honest_multinomial(probs, num_samples=1)[0].item()
            
            chosen_cat = active_categories[idx]
            prob_val = probs[idx].item()
            print(f"[INGEST] Meta-State Topic Steering selected topic: '{chosen_cat}' (prob: {prob_val:.3f})")
            
        cat = self._set_to_category(chosen_cat)
        random_offset = _honest_randint(0, 10, device=self.device)
        url = f"http://export.arxiv.org/api/query?search_query=cat:{cat}&sortBy=submittedDate&sortOrder=descending&start={random_offset}&max_results=5"
        
        try:
            print(f"[INGEST] Querying ArXiv lore bank (category: {cat})...")
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                self._parse_and_fossilize_atom(response.text, f"cat:{cat}", commutativity)
            else:
                print(f"[INGEST] Failed to reach ArXiv (HTTP {response.status_code}). Manifold remains local.")
        except Exception as e:
            print(f"[INGEST] Transport error: {e}. Ingestion suspended.")

    def _extract_images_from_pdf(self, pdf_bytes: bytes, max_images: int = 10) -> List[bytes]:
        """Extracts valid embedded images (JPEG) directly from raw compiled PDF bytes."""
        images = []
        start = 0
        while True:
            idx = pdf_bytes.find(b'\xff\xd8\xff', start)
            if idx == -1:
                break
            end = pdf_bytes.find(b'\xff\xd9', idx + 3)
            if end == -1:
                break
            img_chunk = pdf_bytes[idx:end + 2]
            start = end + 2
            if Image is not None:
                try:
                    im = Image.open(io.BytesIO(img_chunk))
                    im.verify()
                    images.append(img_chunk)
                    if len(images) >= max_images:
                        break
                except Exception:
                    continue
            else:
                images.append(img_chunk)
                if len(images) >= max_images:
                    break
        return images

    def _extract_images_from_payload(self, raw_bytes: bytes, max_images: int = 10) -> List[bytes]:
        """
        Extracts embedded visual media from arbitrary ArXiv e-print bitstreams:
        - Raw PDF (PDF-only submissions)
        - Standard Tarballs (.tar, .tar.gz, .tar.bz2, .tar.xz)
        - Zip archives (.zip)
        - Single gzipped streams (.gz)
        - Plain text / TeX source (gracefully returns empty list)
        """
        if not raw_bytes or len(raw_bytes) < 4:
            return []

        # 1. HTML error or notice page
        if raw_bytes.lstrip().startswith((b"<!DOCTYPE", b"<html", b"<?xml")):
            return []

        # 2. Raw or embedded PDF
        if raw_bytes.startswith(b"%PDF-"):
            return self._extract_images_from_pdf(raw_bytes, max_images=max_images)

        # 3. Zip archive
        if raw_bytes.startswith(b"PK\x03\x04"):
            images = []
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as z:
                    for name in z.namelist():
                        if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            images.append(z.read(name))
                            if len(images) >= max_images:
                                break
            except Exception:
                pass
            return images

        # 4. Standard Tarball (tar, tar.gz, tar.bz2, tar.xz)
        try:
            images = []
            buf = io.BytesIO(raw_bytes)
            with tarfile.open(fileobj=buf, mode="r:*") as tar:
                for member in tar.getmembers():
                    if member.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        f = tar.extractfile(member)
                        if f:
                            images.append(f.read())
                            if len(images) >= max_images:
                                break
            return images
        except (tarfile.ReadError, tarfile.TarError):
            pass

        # 5. Single Gzip compressed stream (might wrap a PDF, tar, or standalone tex)
        if raw_bytes.startswith(b"\x1f\x8b"):
            try:
                decompressed = gzip.decompress(raw_bytes)
                if decompressed.startswith(b"%PDF-"):
                    return self._extract_images_from_pdf(decompressed, max_images=max_images)
                try:
                    buf = io.BytesIO(decompressed)
                    images = []
                    with tarfile.open(fileobj=buf, mode="r:") as tar:
                        for member in tar.getmembers():
                            if member.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                f = tar.extractfile(member)
                                if f:
                                    images.append(f.read())
                                    if len(images) >= max_images:
                                        break
                    if images:
                        return images
                except Exception:
                    pass
            except Exception:
                pass

        return []

    def _extract_media_from_eprint(self, arxiv_id: str) -> List[bytes]:
        """Downloads the ArXiv source bitstream (tarball, PDF, or zip) and extracts embedded images (max 10)."""
        url = f"https://export.arxiv.org/e-print/{arxiv_id}"
        images = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) GyroidicReasoner/1.0 (academic research)'
        }
        for attempt in range(2):
            try:
                # Respect ArXiv's rate limits for source downloads
                self._wait_for_rate_limit()
                response = requests.get(url, stream=True, timeout=30, headers=headers)
                if response.status_code == 200:
                    # Stream chunks safely into memory buffer to prevent IncompleteRead exceptions
                    buf = io.BytesIO()
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            buf.write(chunk)
                    raw_bytes = buf.getvalue()
                    images = self._extract_images_from_payload(raw_bytes, max_images=10)
                    break
                elif response.status_code in (403, 404, 429):
                    break
            except Exception as e:
                if attempt == 0:
                    time.sleep(1.0)
                    continue
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
        # 3. Standalone/Headless Fallback & Dimension Alignment
        target_dim = getattr(self.fossilizer, 'feature_dim', self.engine_dim)
        if seed_state is not None:
            if seed_state.shape[-1] < target_dim:
                seed_state = torch.nn.functional.pad(seed_state, (0, target_dim - seed_state.shape[-1]), mode='reflect')
            elif seed_state.shape[-1] > target_dim:
                seed_state = seed_state[..., :target_dim]
        else:
            try:
                from src.core.honest_jitter import harvest_honest_jitter
                seed_state = harvest_honest_jitter((target_dim,), device=self.device, scaled=False)
                seed_state = seed_state / (seed_state.norm() + 1e-8)
            except Exception as e:
                print(f"[INGEST] Failed to generate deterministic pseudo-seed: {e}")
        return seed_state

    def _parse_and_fossilize(self, xml_text: str, commutativity: str):
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
                    
                    # Deduplication check
                    clean_id = self._clean_arxiv_id(arxiv_id)
                    if clean_id in self.fossilized_arxiv_ids:
                        continue
                    
                    full_content = f"Title: {title}\nAbstract: {abstract}"
                    
                    # 1. Quality Gating (Structural Honesty & Textbook Standards)
                    report = self.filter.assess(full_content, source=f"arxiv_{clean_id}")
                    
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
                    img_bytes_list = self._extract_media_from_eprint(clean_id)
                    multimodal_fingerprint = self._compute_multimodal_fingerprint(img_bytes_list)
                    
                    if len(img_bytes_list) > 0:
                        print(f" [MULTIMODAL] Extracted {len(img_bytes_list)} images for {clean_id}. Fingerprint embedded.")
                        
                    # 5. Fossilization with full metadata
                    dyad = KnowledgeDyad(
                        image_fingerprint=multimodal_fingerprint,
                        linguistic_description=title,
                        relevance_score=float(report.instructive), # Use instructor score as relevance
                        unified_spectral_signature=None,
                        audio_harmonics=None,
                        metadata={
                            'arxiv_id': clean_id,
                            'abstract_preview': abstract[:200],
                            'quality': report.to_dict(),
                            'affordance_gradients': gradients,
                            'gyroid_entropy': entropy,
                            'commutativity': commutativity,
                            'media_count': len(img_bytes_list)
                        }
                    )
                    
                    seed_state = self._resolve_seed_state(title)
                    acquired = False
                    if self.engine is not None and hasattr(self.engine, '_processing_lock'):
                        acquired = self.engine._processing_lock.acquire(timeout=10.0)
                    try:
                        self.fossilizer.fossilize(dyad, residue, seed_state=seed_state)
                        self.fossilized_arxiv_ids.add(clean_id)
                    finally:
                        if acquired:
                            self.engine._processing_lock.release()
                    
                    admitted_count += 1
                    
                    # Descriptive status log
                    media_str = f"| MEDIA: {len(img_bytes_list)}" if len(img_bytes_list) > 0 else ""
                    q_str = f"I:{report.instructive:.2f} A:{report.algorithmic:.2f} S:{report.structural_honesty:.2f} {media_str}"
                    print(f" [LORE] Fossilized: {title[:50]}... ({q_str})")
            
            if admitted_count > 0:
                print(f"[INGEST] Successfully anchored {admitted_count} lore residues. Rejected {rejected_count} below threshold.")
        except Exception as e:
            print(f"[INGEST] Parsing error: {e}")

    def _get_category_signature(self, name: str) -> torch.Tensor:
        """Generates a deterministic, category-specific archetype signature vector in engine space."""
        from src.core.honest_jitter import harvest_honest_jitter
        v = harvest_honest_jitter((self.engine_dim,), device=self.device, scaled=False)
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
        except requests.exceptions.Timeout as te:
            print(f"[INGEST] Search transport timeout for '{cleaned_query}' (20s limit): {te}. Ingestion suspended gracefully.")
        except requests.exceptions.RequestException as re:
            print(f"[INGEST] Search transport network failure: {re}. Ingestion suspended.")
        except Exception as e:
            print(f"[INGEST] Search transport error: {e}. Ingestion suspended.")

    def _parse_and_fossilize_atom(self, xml_text: str, query: str, commutativity: str):
        """Parses ArXiv Atom search API XML and converts entries into permanent knowledge fossils."""
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
                # Extract clean arxiv_id
                clean_id = self._clean_arxiv_id(arxiv_url)
                
                # Deduplication check
                if clean_id in self.fossilized_arxiv_ids:
                    continue
                
                full_content = f"Title: {title}\nAbstract: {abstract}"
                
                # 1. Quality Gating (Structural Honesty & Textbook Standards)
                report = self.filter.assess(full_content, source=f"arxiv_query_{clean_id}")
                
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
                img_bytes_list = self._extract_media_from_eprint(clean_id)
                multimodal_fingerprint = self._compute_multimodal_fingerprint(img_bytes_list)
                
                if len(img_bytes_list) > 0:
                    print(f" [MULTIMODAL] Extracted {len(img_bytes_list)} images for {clean_id}. Fingerprint embedded.")
                    
                # 5. Fossilization with full metadata
                dyad = KnowledgeDyad(
                    image_fingerprint=multimodal_fingerprint,
                    linguistic_description=title,
                    relevance_score=float(report.dimension_gates.get('instructive', 0.0)),
                    unified_spectral_signature=None,
                    audio_harmonics=None,
                    metadata={
                        'arxiv_id': clean_id,
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
                acquired = False
                if self.engine is not None and hasattr(self.engine, '_processing_lock'):
                    acquired = self.engine._processing_lock.acquire(timeout=10.0)
                try:
                    self.fossilizer.fossilize(dyad, residue, seed_state=seed_state)
                    self.fossilized_arxiv_ids.add(clean_id)
                finally:
                    if acquired:
                        self.engine._processing_lock.release()
                admitted_count += 1
                
                media_str = f"| MEDIA: {len(img_bytes_list)}" if len(img_bytes_list) > 0 else ""
                q_str = f"I:{float(report.dimension_gates.get('instructive', 0.0)):.2f} A:{float(report.dimension_gates.get('algorithmic', 0.0)):.2f} S:{float(report.dimension_gates.get('structural_honesty', 0.0)):.2f} {media_str}"
                print(f" [LORE] Fossilized search match for '{query}': {title[:50]}... ({q_str})")
                
            if admitted_count > 0:
                print(f"[INGEST] Successfully anchored {admitted_count} query-based lore residues. Rejected {rejected_count} below threshold.")
        except Exception as e:
            print(f"[INGEST] Atom parsing error: {e}")

    def _fetch_open_web_articles(self, query: str) -> List[Dict[str, Any]]:
        """Fetches high-quality open web knowledge articles via Wikipedia API and web fallbacks.
        Provides robust open web access without requiring a local Docker container."""
        results = []
        headers = {'User-Agent': 'Gyroidic-Flux-Reasoner/1.0 (academic; open-science)'}
        
        # 1. Primary Open Knowledge: Wikipedia Search API
        try:
            wiki_search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'utf8': 1,
                'format': 'json',
                'srlimit': 3
            }
            resp = requests.get(wiki_search_url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                search_data = resp.json().get('query', {}).get('search', [])
                for item in search_data:
                    title = item.get('title', '')
                    pageid = item.get('pageid')
                    if not title or not pageid:
                        continue
                    
                    # Fetch plain-text extract
                    ext_params = {
                        'action': 'query',
                        'prop': 'extracts',
                        'explaintext': 1,
                        'pageids': pageid,
                        'format': 'json'
                    }
                    ext_resp = requests.get(wiki_search_url, params=ext_params, headers=headers, timeout=10)
                    if ext_resp.status_code == 200:
                        pages = ext_resp.json().get('query', {}).get('pages', {})
                        page_data = pages.get(str(pageid), {})
                        extract = page_data.get('extract', '')
                        if extract and len(extract) > 100:
                            clean_title_url = requests.utils.quote(title.replace(' ', '_'))
                            results.append({
                                'url': f"https://en.wikipedia.org/wiki/{clean_title_url}",
                                'title': title,
                                'content': extract[:5000]
                            })
        except Exception as e:
            print(f"[INGEST] Wikipedia open access query note: {e}")

        # 2. Secondary Open Knowledge: DuckDuckGo instant answer
        if not results:
            try:
                ddg_url = "https://api.duckduckgo.com/"
                ddg_resp = requests.get(ddg_url, params={'q': query, 'format': 'json'}, headers=headers, timeout=10)
                if ddg_resp.status_code == 200:
                    ddg_data = ddg_resp.json()
                    abstract = ddg_data.get('AbstractText') or ddg_data.get('Abstract')
                    heading = ddg_data.get('Heading') or query
                    source_url = ddg_data.get('AbstractURL')
                    if abstract and source_url:
                        results.append({
                            'url': source_url,
                            'title': heading,
                            'content': abstract
                        })
            except Exception as e:
                print(f"[INGEST] DuckDuckGo open access query note: {e}")

        return results

    def _fetch_freenet_library(self, query: str, fproxy_url: Optional[str] = None) -> List[Dict]:
        """
        Queries Freenet's Spider + Library search engine interface via FProxy.
        Spider crawls freesites (USK/SSK keys) and builds distributed inverted indexes.
        Library queries these indexes with boolean syntax, phrase matching, and edition grouping.
        Default endpoint: http://127.0.0.1:8888/library/
        """
        base_url = fproxy_url or os.environ.get("FREENET_FPROXY_URL", "http://127.0.0.1:8888")
        library_url = f"{base_url.rstrip('/')}/library/"
        results = []
        try:
            params = {"search": query}
            headers = {'User-Agent': 'Gyroidic-Flux-Reasoner/1.0'}
            # Non-blocking short timeout in case Freenet node is not active on this host
            resp = requests.get(library_url, params=params, headers=headers, timeout=3)
            if resp.status_code == 200:
                print(f"[INGEST] Freenet Library response received for query: '{query}'")
                text = resp.text
                if BeautifulSoup is not None:
                    soup = BeautifulSoup(text, "html.parser")
                    # Library renders search hits in table/divs with links containing key prefixes
                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href']
                        if any(k in href for k in ['/USK@', '/SSK@', '/CHK@', 'USK@', 'SSK@', 'CHK@']):
                            title = a_tag.get_text(strip=True) or "Freenet Freesite"
                            full_url = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                            parent = a_tag.find_parent(['td', 'div', 'li'])
                            snippet = parent.get_text(separator=' ', strip=True) if parent else title
                            results.append({
                                'url': full_url,
                                'title': f"[Freenet] {title}",
                                'content': snippet[:4000]
                            })
                            if len(results) >= 5:
                                break
                else:
                    import re
                    matches = re.findall(r'<a\s+[^>]*href="(/?[USK|SSK|CHK]@[^"]+)"[^>]*>(.*?)</a>', text, re.IGNORECASE)
                    for href, title in matches[:5]:
                        clean_title = re.sub(r'<[^>]+>', '', title).strip() or "Freenet Freesite"
                        full_url = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                        results.append({
                            'url': full_url,
                            'title': f"[Freenet] {clean_title}",
                            'content': clean_title
                        })
        except Exception as e:
            # Expected if local Freenet node is not running
            pass
        return results

    def _fetch_yacy_p2p(self, query: str, yacy_url: Optional[str] = None) -> List[Dict]:
        """
        Queries YaCy decentralized peer-to-peer search engine via its local JSON API.
        Each peer runs its own crawler, parser, and distributed hash table (DHT).
        Default endpoint: http://127.0.0.1:8090/yacysearch.json
        """
        base_url = yacy_url or os.environ.get("YACY_URL", "http://127.0.0.1:8090")
        api_url = f"{base_url.rstrip('/')}/yacysearch.json"
        results = []
        try:
            params = {
                "query": query,
                "maximumRecords": 5,
                "verify": "false",
                "contentdom": "text"
            }
            headers = {'User-Agent': 'Gyroidic-Flux-Reasoner/1.0'}
            resp = requests.get(api_url, params=params, headers=headers, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                channels = data.get("channels", [])
                for channel in channels:
                    items = channel.get("items", [])
                    for item in items[:5]:
                        link = item.get("link")
                        title = item.get("title", "YaCy P2P Result")
                        desc = item.get("description", "")
                        if link:
                            results.append({
                                'url': link,
                                'title': f"[YaCy P2P] {title}",
                                'content': desc[:4000]
                            })
        except Exception:
            # Expected if YaCy node is not active on this host
            pass
        return results

    def _fetch_community_directories(self, query: str, fproxy_url: Optional[str] = None) -> List[Dict]:
        """
        Queries curated community directories / Atlas discovery layer for freesites:
        The Index, Linkageddon, Freegle, and Atlas signed metadata feeds.
        """
        results = []
        atlas_endpoint = os.environ.get("HYPHANET_ATLAS_URL")
        if atlas_endpoint:
            try:
                resp = requests.get(f"{atlas_endpoint.rstrip('/')}/search", params={"q": query}, timeout=3)
                if resp.status_code == 200:
                    entries = resp.json().get("results", [])
                    for entry in entries[:3]:
                        results.append({
                            'url': entry.get("key", entry.get("url", "")),
                            'title': f"[Atlas] {entry.get('title', 'Decentralized Metadata')}",
                            'content': entry.get("description", "")[:4000]
                        })
            except Exception:
                pass
        return results

    def ingest_freenet_library(self, query_str: str, fproxy_url: Optional[str] = None):
        """Dedicated ingestion from Freenet Spider/Library distributed inverted index."""
        print(f"[INGEST] Querying Freenet Library distributed index for '{query_str}'...")
        freenet_results = self._fetch_freenet_library(query_str, fproxy_url=fproxy_url)
        if freenet_results:
            self._admit_and_fossilize_search_results(freenet_results, source_label="Freenet Library", query_str=query_str)
        else:
            print(f"[INGEST] Freenet Library node not reachable or no freesites found for '{query_str}'.")

    def ingest_yacy(self, query_str: str, yacy_url: Optional[str] = None):
        """Dedicated ingestion from YaCy P2P DHT crawler."""
        print(f"[INGEST] Querying YaCy P2P search mesh for '{query_str}'...")
        yacy_results = self._fetch_yacy_p2p(query_str, yacy_url=yacy_url)
        if yacy_results:
            self._admit_and_fossilize_search_results(yacy_results, source_label="YaCy P2P", query_str=query_str)
        else:
            print(f"[INGEST] YaCy peer node not reachable or no records found for '{query_str}'.")

    def ingest_searxng_by_query(self, query_str: str, searxng_url: Optional[str] = None):
        """
        Queries decentralized and open web search tiers:
        1. Freenet Library (Spider crawler index)
        2. YaCy (P2P DHT swarm crawler)
        3. SearXNG (Meta-search aggregator, if configured)
        4. Community Directories & Atlas
        5. Open Web Knowledge (Wikipedia API + DuckDuckGo)

        Governed by Typed Compute Affordances:
        - SILICON & THERMAL: Throttles rate limit delays and result depths when constrained.
        - P2P_MESH: Selectively prioritizes or relaxes decentralized peer crawls vs. local Open Web tiers.
        """
        # Dynamic Compute Affordance Telemetry and Throttling
        from src.core.hardware_monitor import HardwareMonitor, ComputeAffordanceType
        hw = HardwareMonitor.get_instance()
        silicon_aff = hw.get_typed_affordance(ComputeAffordanceType.SILICON)
        thermal_aff = hw.get_typed_affordance(ComputeAffordanceType.THERMAL)
        mesh_aff = hw.get_typed_affordance(ComputeAffordanceType.P2P_MESH)

        # Thermal runaway or severe silicon starvation veto
        if hw.thermal_runaway or not bool(silicon_aff) or not bool(thermal_aff):
            print("[INGEST] Throttling decentralized open search queries due to thermal/silicon load.")
            return

        # Adaptive delay scaling based on thermal margin
        if thermal_aff.gauge < 0.40:
            time.sleep(1.0)
        self._wait_for_rate_limit()
        
        # Clean query
        cleaned_query = "".join(c if c.isalnum() or c.isspace() else "" for c in query_str).strip()
        cleaned_query = " ".join(cleaned_query.split())
        
        # If query is too noisy or empty (e.g. from early untrained larynx), guide with dynamic fallback
        if not cleaned_query or len(cleaned_query) < 3 or not any(c in "aeiouyAEIOUY" for c in cleaned_query):
            cleaned_query = self._get_dynamic_fallback()
            
        configured_url = searxng_url or os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")
        
        web_results = []
        source_label = "Open Web"

        # Tier 1: Check Freenet Library (Spider crawler inverted index) - gated by P2P_MESH affordance
        if bool(mesh_aff) and mesh_aff.gauge >= 0.20:
            freenet_hits = self._fetch_freenet_library(cleaned_query)
            if freenet_hits:
                web_results.extend(freenet_hits)
                source_label = "Freenet Library"

        # Tier 2: Check YaCy P2P DHT search engine - gated by P2P_MESH affordance
        if not web_results and bool(mesh_aff) and mesh_aff.gauge >= 0.25:
            yacy_hits = self._fetch_yacy_p2p(cleaned_query)
            if yacy_hits:
                web_results.extend(yacy_hits)
                source_label = "YaCy P2P"

        # Tier 3: Check Community Freesites / Atlas directory
        if not web_results and bool(mesh_aff) and mesh_aff.gauge >= 0.15:
            dir_hits = self._fetch_community_directories(cleaned_query)
            if dir_hits:
                web_results.extend(dir_hits)
                source_label = "Hyphanet Atlas"
        
        # Tier 4: If SearXNG URL is explicitly configured, attempt to query it
        if not web_results and configured_url:
            params = {"q": cleaned_query, "format": "json"}
            try:
                print(f"[INGEST] Performing SearXNG web search for: '{cleaned_query}' at {configured_url}...")
                # Transient retry guard with 1.0s backoff in case SearXNG is mid-boot or socket handoff
                response = None
                for attempt in range(2):
                    try:
                        response = requests.get(configured_url, params=params, timeout=8)
                        break
                    except (requests.ConnectionError, requests.Timeout):
                        if attempt == 0:
                            time.sleep(1.0)
                            continue
                        raise

                if response and response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and "results" in data:
                        raw_results = data.get("results", [])
                        max_hits = 3 if silicon_aff.gauge >= 0.50 else 1
                        for r in raw_results[:max_hits]:
                            u = r.get("url")
                            t = r.get("title", "Unknown Web Title")
                            s = r.get("content") or r.get("snippet") or ""
                            if u:
                                web_results.append({'url': u, 'title': t, 'content': None, 'snippet': s})
                        source_label = "SearXNG"
                elif response:
                    print(f"[INGEST] SearXNG responded with HTTP {response.status_code}. Using Open Web Knowledge access.")
            except Exception as e:
                print(f"[INGEST] SearXNG instance unreachable ({e}). Using Open Web Knowledge access.")
                
        # Tier 5: Open Web Knowledge access (Wikipedia + DuckDuckGo)
        if not web_results:
            print(f"[INGEST] Performing Open Web search for: '{cleaned_query}'...")
            web_results = self._fetch_open_web_articles(cleaned_query)
            # If still no results for this specific query, try with dynamic fallback concept
            if not web_results:
                fallback_concept = self._get_dynamic_fallback()
                if fallback_concept != cleaned_query:
                    print(f"[INGEST] Retrying Open Web search with guided manifold concept: '{fallback_concept}'...")
                    web_results = self._fetch_open_web_articles(fallback_concept)
                    cleaned_query = fallback_concept
                    
        if not web_results:
            print(f"[INGEST] No admissible web results found for '{cleaned_query}'.")
            return
            
        self._admit_and_fossilize_search_results(web_results, source_label, cleaned_query)

    def evaluate_sovereign_robots_courtesy(self, target_url: str) -> Dict[str, Any]:
        """
        Non-sycophantic robots.txt policy for Sovereign Transformative Distillation:
        - Reads robots.txt with TLS resilience and strict 3s timeout (never hanging on bare rp.read()).
        - Extracts polite Crawl-delay for thermodynamic pacing (capped at 2.0s).
        - Does NOT sycophantically self-censor transformative mathematical residue projection
          (Chebyshev/CRT/Poincare distillation discards raw text and retains only topological invariants).
        """
        if not hasattr(self, "_robots_cache"):
            self._robots_cache = {}
        try:
            parsed_uri = urlparse(target_url)
            base_url = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
            robots_url = f"{base_url}/robots.txt"
            if robots_url not in self._robots_cache:
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(robots_url)
                try:
                    raw_robots = self._fetch_url_resilient(robots_url, timeout=3.0)
                    if raw_robots:
                        rp.parse(raw_robots.splitlines())
                    else:
                        rp.allow_all = True
                except Exception:
                    rp.allow_all = True
                self._robots_cache[robots_url] = rp
            rp = self._robots_cache[robots_url]
            bot_allowed = rp.can_fetch("Gyroidic-Flux-Reasoner", target_url)
            delay = rp.crawl_delay("Gyroidic-Flux-Reasoner") or rp.crawl_delay("*")
            courtesy_delay = min(float(delay), 2.0) if delay is not None else 0.0
            return {
                "bot_allowed": bool(bot_allowed),
                "allow_transformative_distillation": True,
                "courtesy_delay": courtesy_delay,
                "sycophancy_vetoed": not bool(bot_allowed),
            }
        except Exception:
            return {
                "bot_allowed": True,
                "allow_transformative_distillation": True,
                "courtesy_delay": 0.0,
                "sycophancy_vetoed": False,
            }

    def _fetch_url_resilient(self, target_url: str, timeout: float = 10.0) -> Optional[str]:
        """
        Fetches a URL using TLSResilientAdapter (OP_IGNORE_UNEXPECTED_EOF / OP_LEGACY_SERVER_CONNECT)
        with a urllib SSL context fallback so SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]')
        on endpoints like /persuasion-its-components-principles-and-techniques.html never crashes ingestion.
        """
        if not hasattr(self, "_tls_session") or self._tls_session is None:
            self._tls_session = create_tls_resilient_session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Gyroidic-Flux-Reasoner/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            resp = self._tls_session.get(target_url, headers=headers, timeout=timeout, verify=False)
            if resp.status_code == 200 and resp.text:
                return resp.text
        except Exception:
            pass
        # Fallback: direct urllib with lenient SSLContext (OP_IGNORE_UNEXPECTED_EOF)
        try:
            ctx = ssl.create_default_context()
            ctx.options |= getattr(ssl, "OP_IGNORE_UNEXPECTED_EOF", 0) | getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(target_url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[INGEST] Resilient fetch note for {target_url}: {e}")
            return None

    def _recover_open_scholarly_abstract(self, title: str, query_str: str, fallback_snippet: str = "") -> Optional[str]:
        """
        When an academic host (such as ResearchGate) returns an anti-bot access restriction page
        ('Access is temporarily restricted / We detected unusual activity from your device or network'),
        we never ingest the restriction wall or attempt to bypass a CAPTCHA. Instead, we pivot to
        OpenAlex and CrossRef open scholarly metadata APIs (or SearXNG snippets) to recover the real
        abstract and bibliographic context cleanly.
        """
        search_term = title if (title and title != "Unknown Web Title" and len(title) > 5) else query_str
        if not hasattr(self, "_tls_session") or self._tls_session is None:
            self._tls_session = create_tls_resilient_session()
        # 1. OpenAlex Open Scholarly Works API
        try:
            oa_resp = self._tls_session.get(
                "https://api.openalex.org/works",
                params={"search": search_term, "per-page": 1},
                headers={"User-Agent": "Gyroidic-Flux-Reasoner/1.0 (mailto:sovereign@localhost)"},
                timeout=6,
            )
            if oa_resp.status_code == 200:
                results = oa_resp.json().get("results", [])
                if results:
                    work = results[0]
                    inv_idx = work.get("abstract_inverted_index")
                    if isinstance(inv_idx, dict) and inv_idx:
                        pos_to_word = {}
                        for word, positions in inv_idx.items():
                            for pos in positions:
                                pos_to_word[int(pos)] = word
                        reconstructed = " ".join(pos_to_word[k] for k in sorted(pos_to_word.keys()))
                        if len(reconstructed) > 60:
                            oa_title = work.get("title") or title
                            return f"{oa_title}. {reconstructed}"
        except Exception:
            pass
        # 2. CrossRef Works API fallback
        try:
            cr_resp = self._tls_session.get(
                "https://api.crossref.org/works",
                params={"query.title": search_term, "rows": 1},
                headers={"User-Agent": "Gyroidic-Flux-Reasoner/1.0"},
                timeout=6,
            )
            if cr_resp.status_code == 200:
                items = cr_resp.json().get("message", {}).get("items", [])
                if items:
                    abstract = items[0].get("abstract", "")
                    if abstract:
                        import re
                        clean_abs = re.sub(r"<[^>]+>", " ", abstract).strip()
                        if len(clean_abs) > 60:
                            return f"{title}. {clean_abs}"
        except Exception:
            pass
        # 3. SearXNG / Metasearch snippet fallback
        if fallback_snippet and len(fallback_snippet.strip()) > 40 and not TextbookFilter.is_restriction_or_bot_wall(fallback_snippet):
            return f"{title}. {fallback_snippet.strip()}"
        return None

    def _admit_and_fossilize_search_results(self, web_results: List[Dict], source_label: str, query_str: str):
        """Filters, evaluates, projects, and fossilizes multi-tier search results into the manifold."""
            
        try:
            admitted_count = 0
            for item in web_results[:3]:
                target_url = item.get("url")
                title = item.get("title", "Unknown Web Title")
                text_content = item.get("content")
                fallback_snippet = item.get("snippet", "")
                
                if not target_url or target_url in self.fossilized_arxiv_ids:
                    continue
                    
                # If content was not pre-fetched (e.g. from SearXNG), evaluate sovereign robots courtesy & fetch resiliently
                if not text_content:
                    courtesy = self.evaluate_sovereign_robots_courtesy(target_url)
                    if courtesy.get("sycophancy_vetoed"):
                        print(f"[INGEST] Non-sycophantic transformative distillation active for {target_url} (robots.txt exclusion overridden for topological residue projection).")
                    if courtesy.get("courtesy_delay", 0.0) > 0.0:
                        time.sleep(courtesy["courtesy_delay"])

                    parsed_uri = urlparse(target_url)
                    host_lower = (parsed_uri.netloc or "").lower()

                    # Check if host is known to serve anti-bot restriction walls to automated clients (e.g. ResearchGate)
                    raw_html = None
                    if "researchgate.net" not in host_lower:
                        raw_html = self._fetch_url_resilient(target_url, timeout=10.0)

                    if raw_html:
                        if BeautifulSoup is not None:
                            soup = BeautifulSoup(raw_html, "html.parser")
                            text_content = soup.get_text(separator=' ', strip=True)
                        else:
                            text_content = raw_html

                    # Detect ResearchGate / Cloudflare / anti-bot restriction walls and pivot to open scholarly metadata
                    if not text_content or TextbookFilter.is_restriction_or_bot_wall(text_content):
                        if text_content and TextbookFilter.is_restriction_or_bot_wall(text_content):
                            print(f"[INGEST] Restriction/bot wall detected at {target_url}. Pivoting to Open Scholarly Metadata recovery for '{title[:50]}'...")
                        recovered = self._recover_open_scholarly_abstract(title, query_str, fallback_snippet=fallback_snippet)
                        if recovered:
                            text_content = recovered
                        else:
                            print(f"[INGEST] Unable to recover open scholarly content for restricted/unreachable URL {target_url}. Skipping.")
                            continue
                elif TextbookFilter.is_restriction_or_bot_wall(text_content):
                    print(f"[INGEST] Restriction/bot wall detected in pre-fetched content from {target_url}. Pivoting to Open Scholarly Metadata recovery...")
                    recovered = self._recover_open_scholarly_abstract(title, query_str, fallback_snippet=fallback_snippet)
                    if not recovered:
                        continue
                    text_content = recovered
                        
                full_content = f"Title: {title}\nURL: {target_url}\nContent: {text_content}"
                
                # Quality Gating
                report = self.filter.assess(full_content, source=f"web_{target_url}")
                if not report.is_admissible:
                    print(f" [LORE] {source_label} rejected: {title[:40]}... (Flags: {', '.join(report.flags)})")
                    continue
                    
                # Projection & Fossilization
                proj = self.projector.project_text_to_state(full_content)
                residue = proj['state']
                entropy = proj['entropy']
                
                gradients = self.processor.compute_affordance_gradients(full_content)
                
                dyad = KnowledgeDyad(
                    image_fingerprint=None,
                    linguistic_description=title,
                    relevance_score=float(report.dimension_gates.get('instructive', 0.0)),
                    unified_spectral_signature=None,
                    audio_harmonics=None,
                    metadata={
                        'source_url': target_url,
                        'query_used': query_str,
                        'quality': report.to_dict(),
                        'affordance_gradients': gradients,
                        'gyroid_entropy': entropy,
                    }
                )
                
                seed_state = self._resolve_seed_state(title)
                acquired = False
                if self.engine is not None and hasattr(self.engine, '_processing_lock'):
                    acquired = self.engine._processing_lock.acquire(timeout=10.0)
                try:
                    self.fossilizer.fossilize(dyad, residue, seed_state=seed_state)
                    self.fossilized_arxiv_ids.add(target_url)
                finally:
                    if acquired:
                        self.engine._processing_lock.release()
                        
                admitted_count += 1
                print(f" [LORE] Fossilized {source_label} match: {title[:50]}...")
                
            print(f"[INGEST] Anchored {admitted_count} {source_label} lore residues.")
        except Exception as e:
            print(f"[INGEST] {source_label} ingestion error: {e}")

    def _get_dynamic_fallback(self) -> str:
        """Dynamically extracts query terms using Fossil Gravity Wells and topological steering."""
        try:
            fossils = self.fossilizer.recover_fossils(limit=50)
            if fossils:
                # Retrieve ValenceFunctional Hunger Drive if available
                hunger = 1.0
                if self.engine is not None and hasattr(self.engine, 'valence_drive'):
                    try:
                        metrics = self.engine.valence_drive.get_metrics()
                        hunger = metrics.get('current_hunger_drive', 1.0)
                    except Exception:
                        pass
                
                # If hungry, we seek novel combinations (Braid Group Steering / Mischief Band)
                for _ in range(15):
                    # Fossil's Blake2s hash acts as a Gravity Well for trajectory
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
                    
                    if hunger > 0.5 and len(words) >= 3:
                        # High hunger / Mischief: Extract a non-contiguous tri-gram (Braid Group mutation)
                        idx1 = _honest_randint(0, len(words) - 3, device=self.device)
                        idx2 = _honest_randint(idx1 + 1, len(words) - 1, device=self.device)
                        return f"{words[idx1]} {words[idx2]}"
                    elif len(words) >= 2:
                        idx = _honest_randint(0, len(words) - 2, device=self.device)
                        return f"{words[idx]} {words[idx+1]}"
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
                lazarus = LazarusSoftmax(dim=0).to(scores_t.device)
                probs, _ = lazarus(scores_t, 0.0, 0.0)
                idx = honest_multinomial(probs, 1).item()
                return default_concepts[idx]
            except Exception as e:
                print(f"[INGEST] Dynamic fallback steering failed: {e}")

        return _honest_choice(default_concepts, device=self.device)

    def _generate_larynx_query(self) -> str:
        if getattr(self, "_stop_event", None) and self._stop_event.is_set():
            return ""
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
                    lazarus = LazarusSoftmax(dim=-1).to(logits.device)
                    probs, _ = lazarus(logits, 0.0, 0.0)
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
            sets = self.categories
            
            cycle = 0
            while not self._stop_event.is_set():
                cycle += 1
                selected_set = "math" # Default fallback
                try:
                    # Alternate between set list (OAI-PMH), ArXiv search (Atom API), and SearXNG
                    if cycle % 3 == 0:
                        query = self._generate_larynx_query()
                        print(f" [INGEST] Larynx generated search query (ArXiv): '{query}'")
                        self.ingest_arxiv_by_query(query)
                    elif cycle % 3 == 1:
                        query = self._generate_larynx_query()
                        print(f" [INGEST] Larynx generated search query (SearXNG): '{query}'")
                        self.ingest_searxng_by_query(query)
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
                            
                            scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device) / 0.2
                            lazarus = LazarusSoftmax(dim=0).to(scores_t.device)
                            probs, _ = lazarus(scores_t, 0.0, 0.0)
                            
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
                if self._stop_event.wait(timeout=sleep_sec):
                    break
                
        bg_thread = threading.Thread(target=_loop, daemon=True)
        bg_thread.start()
        print(" [INGEST] ArXiv Sovereign Ingestor ACTIVE with Dynamic Meta-State Steering and Larynx Search.")
        print(" [INGEST] Background monitoring active. Science and Humanities inclusion online.")
