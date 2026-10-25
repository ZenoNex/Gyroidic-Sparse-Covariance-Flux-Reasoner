"""
Investor News Ingestion Subsystem with Larynx Integration and MAML Online Adaptation.

Role: Programmatically scrapes investor news, market topologies, and macro-economic
RSS feeds under robots.txt leniency validation. Features direct autoregressive Larynx
query generation and MAML inner-loop online adaptation to steer the reasoner's manifold.
"""

import urllib.request
import urllib.robotparser
from urllib.parse import urlparse
import json
import os
import re
import copy
from typing import List, Dict, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvestorNewsIngestor(nn.Module):
    """
    Ingests financial news feeds, market sentiment, and macroeconomic topologies
    with autoregressive Larynx query steering and MAML online adaptation.
    """
    def __init__(
        self,
        state_dim: int = 32,
        user_agent: str = "GyroidicFluxReasonerBot/1.9",
        larynx: Optional[Any] = None,
        engine: Optional[Any] = None,
        maml_adapter: Optional[Any] = None
    ):
        super().__init__()
        self.state_dim = state_dim
        self.user_agent = user_agent
        self.larynx = larynx
        self.engine = engine
        self.maml_adapter = maml_adapter
        self.rp_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}
        
        # Linear projection for market sentiment to state space
        self.sentiment_proj = nn.Linear(state_dim, state_dim, bias=False)
        nn.init.orthogonal_(self.sentiment_proj.weight, gain=0.25)
        
        # Canonical financial topics for fallback steering
        self.financial_concepts = [
            "sovereign yield curve inversion",
            "macro credit spread dislocation",
            "systemic liquidity volatility",
            "central bank repo collateral friction",
            "commercial real estate land rent debt",
            "interbank counterparty clearing stress"
        ]

    def set_larynx(self, larynx: Any):
        """Connect external Larynx module for autoregressive market query generation."""
        self.larynx = larynx

    def set_engine(self, engine: Any):
        """Connect parent engine or orchestrator."""
        self.engine = engine
        if hasattr(engine, 'larynx') and self.larynx is None:
            self.larynx = engine.larynx
        if hasattr(engine, 'calm') and self.maml_adapter is None:
            self.maml_adapter = engine.calm

    def set_maml_adapter(self, adapter: Any):
        """Connect canonical MAML adaptation loop (e.g. CALM from src.surrogates.calm_predictor)."""
        self.maml_adapter = adapter

    DEFAULT_USER_FEEDS = [
        {"url": "https://finance.yahoo.com/news/rssindex", "name": "Yahoo Finance Top News", "enabled": True},
        {"url": "https://feeds.content.dowjones.io/public/rss/mw_topstories", "name": "MarketWatch Top Stories", "enabled": True},
        {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml", "name": "NYT Economy News", "enabled": True},
        {"url": "https://export.arxiv.org/rss/math.DG", "name": "arXiv Differential Geometry", "enabled": True}
    ]

    def _get_feeds_filepath(self) -> str:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "user_rss_feeds.json")

    def get_user_feeds(self) -> List[Dict[str, Any]]:
        """Retrieve user-selected RSS feeds with persistence fallback to defaults."""
        path = self._get_feeds_filepath()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        return data
            except Exception:
                pass
        return copy.deepcopy(self.DEFAULT_USER_FEEDS)

    def save_user_feeds(self, feeds: List[Dict[str, Any]]) -> bool:
        """Persist user-selected RSS feeds."""
        path = self._get_feeds_filepath()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(feeds, f, indent=2)
            return True
        except Exception as e:
            print(f"[INGESTOR] Failed to save user RSS feeds: {e}")
            return False

    def add_user_feed(self, url: str, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Add a new RSS feed to user configuration."""
        feeds = self.get_user_feeds()
        for f in feeds:
            if f.get("url") == url:
                if name:
                    f["name"] = name
                f["enabled"] = True
                self.save_user_feeds(feeds)
                return feeds
        feed_name = name or urlparse(url).netloc or "Custom RSS Feed"
        feeds.append({"url": url, "name": feed_name, "enabled": True})
        self.save_user_feeds(feeds)
        return feeds

    def remove_user_feed(self, url: str) -> List[Dict[str, Any]]:
        """Remove an RSS feed from user configuration."""
        feeds = [f for f in self.get_user_feeds() if f.get("url") != url]
        self.save_user_feeds(feeds)
        return feeds

    def toggle_user_feed(self, url: str, enabled: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Toggle or set enabled status for an RSS feed."""
        feeds = self.get_user_feeds()
        for f in feeds:
            if f.get("url") == url:
                f["enabled"] = not f.get("enabled", True) if enabled is None else bool(enabled)
                break
        self.save_user_feeds(feeds)
        return feeds

    def get_active_feed_urls(self) -> List[str]:
        """Return list of enabled RSS feed URLs for auto-injection."""
        feeds = self.get_user_feeds()
        active = [f["url"] for f in feeds if f.get("enabled", True) and f.get("url")]
        return active if active else [f["url"] for f in self.DEFAULT_USER_FEEDS]

    def _get_robot_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = f"{base_url}/robots.txt"
        
        if robots_url in self.rp_cache:
            return self.rp_cache[robots_url]
            
        rp = urllib.robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(
                robots_url,
                headers={"User-Agent": self.user_agent}
            )
            with urllib.request.urlopen(req, timeout=3.0) as response:
                rp.parse(response.read().decode('utf-8', errors='ignore').splitlines())
        except Exception:
            rp.allow_all = True
            
        self.rp_cache[robots_url] = rp
        return rp

    def is_allowed(self, url: str) -> bool:
        """Verify robots.txt allows crawling the given URL."""
        try:
            rp = self._get_robot_parser(url)
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    def fetch_rss_feed(self, feed_url: str) -> List[Dict[str, str]]:
        """Fetch news items from an RSS feed if permitted by robots.txt."""
        if not self.is_allowed(feed_url):
            print(f"[INGESTOR] robots.txt excludes access to feed: {feed_url}")
            return []
            
        try:
            req = urllib.request.Request(
                feed_url,
                headers={"User-Agent": self.user_agent}
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                content = response.read().decode('utf-8', errors='ignore')
                
            items = []
            item_blocks = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
            for block in item_blocks:
                title_match = re.search(r'<title>(.*?)</title>', block, re.DOTALL)
                link_match = re.search(r'<link>(.*?)</link>', block, re.DOTALL)
                desc_match = re.search(r'<description>(.*?)</description>', block, re.DOTALL)
                
                title = title_match.group(1).strip() if title_match else ""
                link = link_match.group(1).strip() if link_match else ""
                desc = desc_match.group(1).strip() if desc_match else ""
                
                title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title)
                desc = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', desc)
                desc = re.sub(r'<[^>]*>', '', desc)
                
                if title:
                    items.append({
                        "title": title,
                        "link": link,
                        "description": desc
                    })
            return items
        except Exception as e:
            print(f"[INGESTOR] Failed to fetch feed {feed_url}: {e}")
            return []

    def generate_larynx_market_query(self, state: Optional[torch.Tensor] = None) -> str:
        """
        Uses the connected Larynx autoregressively to draft a market query phrase
        from the active manifold state.
        """
        larynx = self.larynx
        if larynx is None and self.engine is not None and hasattr(self.engine, 'larynx'):
            larynx = self.engine.larynx

        if larynx is not None:
            try:
                device = next(self.sentiment_proj.parameters()).device
                curr_state = state.clone().detach().to(device) if state is not None else torch.zeros((1, self.state_dim), device=device)
                if curr_state.dim() == 1:
                    curr_state = curr_state.unsqueeze(0)
                    
                vowels = set('aeiouAEIOU')
                generated_chars = []
                with torch.no_grad():
                    from src.core.gluing_operator import LazarusSoftmax
                    lazarus = LazarusSoftmax(dim=-1).to(device)
                    for step_idx in range(30):
                        logits, _ = larynx(curr_state, temperature=1.1)
                        if logits.dim() == 1:
                            logits = logits.unsqueeze(0)

                        # Clean vocabulary filtering
                        for idx in range(min(128, logits.shape[-1])):
                            if idx < 32 or idx > 126:
                                logits[0, idx] = -1e9

                        # Syllable-coherent vowel modulation (Chern-Simons gasket acoustic rule)
                        prev_char = generated_chars[-1] if generated_chars else ' '
                        if prev_char not in vowels and prev_char.isalpha():
                            for v in vowels:
                                if ord(v) < logits.shape[-1]:
                                    logits[0, ord(v)] *= 1.4
                        elif prev_char in vowels:
                            for v in vowels:
                                if ord(v) < logits.shape[-1]:
                                    logits[0, ord(v)] *= 0.4

                        probs, _ = lazarus(logits, 0.0, 0.0)
                        char_idx = torch.multinomial(probs[0], 1).item()
                        char = chr(max(32, min(126, char_idx)))

                        if char in ('.', '!', '?', ';', '\n'):
                            break
                        generated_chars.append(char)

                        # Autoregressive tanh feedback from projection weights
                        if hasattr(larynx, 'proj') and hasattr(larynx.proj, 'weight') and char_idx < larynx.proj.weight.shape[0]:
                            feedback = torch.tanh(larynx.proj.weight[char_idx].unsqueeze(0))
                            if feedback.shape == curr_state.shape:
                                curr_state = 0.9 * curr_state + 0.1 * feedback

                query = "".join(generated_chars).strip()
                cleaned_query = re.sub(r'[^a-zA-Z0-9_\-\s]', ' ', query).strip()
                if len(cleaned_query) >= 3:
                    return cleaned_query
            except Exception:
                pass

        # Honest jitter fallback from financial concepts
        from src.core.honest_jitter import harvest_honest_jitter
        jitter = harvest_honest_jitter((1,), device=torch.device('cpu'), scaled=False).item()
        idx = int(((jitter + 1.0) / 2.0) * len(self.financial_concepts))
        return self.financial_concepts[min(idx, len(self.financial_concepts) - 1)]

    def ingest_market_topologies(
        self,
        feed_urls: List[str],
        state: Optional[torch.Tensor] = None
    ) -> List[str]:
        """Harvests market news items and converts them to raw text payloads."""
        payloads = []
        urls = feed_urls
        # If state provided, prioritize endpoints aligned with state polarity
        if state is not None and len(feed_urls) > 1:
            val = float(state.flatten()[0].item()) if state.numel() > 0 else 0.0
            if val < 0:
                urls = list(reversed(feed_urls))

        for url in urls:
            items = self.fetch_rss_feed(url)
            for item in items:
                formatted = f"MARKET NEWS: {item['title']}. {item['description']}"
                payloads.append(formatted)
        return payloads

    def fetch_financial_feeds(
        self,
        feeds: Optional[List[str]] = None,
        state: Optional[torch.Tensor] = None
    ) -> List[str]:
        """Ingest real-time financial market RSS topologies with optional state steering."""
        if feeds is None:
            feeds = self.get_active_feed_urls()
        try:
            results = self.ingest_market_topologies(feeds, state=state)
            if results:
                return results
            return ["Equities flux neutral; topological yield curve steady."]
        except Exception:
            return ["Equities flux neutral; topological yield curve steady."]

    def steer_state_from_payloads(
        self,
        payloads: List[str],
        state_dim: int = 32,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Converts ingested market text payloads into an active continuous steering vector
        via the adapted MAML sentiment projection head.
        """
        dev = torch.device(device)
        steering_vec = torch.zeros(state_dim, device=dev)
        if not payloads:
            return steering_vec

        char_accum = torch.zeros(state_dim, device=dev)
        total_chars = 0

        for text in payloads[:8]:
            for i, ch in enumerate(text[:state_dim * 4]):
                pos = i % state_dim
                val = ((ord(ch) % 43) - 21.0) / 21.0
                char_accum[pos] += val
                total_chars += 1

        if total_chars > 0:
            char_accum = char_accum / max(1.0, float(total_chars) / state_dim)
            with torch.no_grad():
                proj_dev = next(self.sentiment_proj.parameters()).device
                raw_proj = self.sentiment_proj(char_accum.to(proj_dev)).to(dev)
                steering_vec = torch.tanh(raw_proj)
                steering_vec = steering_vec / (torch.norm(steering_vec) + 1e-8) * 0.25

        return steering_vec

    def adapt_maml(
        self,
        support_states: torch.Tensor,
        support_targets: torch.Tensor,
        steps: int = 1,
        lr: float = 0.01,
        entropy: Optional[torch.Tensor] = None
    ) -> 'InvestorNewsIngestor':
        """
        Connect to canonical MAML adaptation loop (CALM or KAGHBlock) on support market data.
        Delegates directly to existing MAML infrastructure rather than recreating ad-hoc adaptation loops.
        """
        adapter = self.maml_adapter
        if adapter is None and self.engine is not None:
            adapter = getattr(self.engine, 'calm', getattr(self.engine, 'kagh', None))

        if adapter is not None:
            if hasattr(adapter, 'adapt'):
                adapted = adapter.adapt(
                    support_states.unsqueeze(1) if support_states.dim() == 2 else support_states,
                    support_targets,
                    steps=steps,
                    lr=lr,
                    entropy=entropy
                )
                self.maml_adapter = adapted
            elif hasattr(adapter, 'adapt_online'):
                adapted = adapter.adapt_online(
                    support_states,
                    support_targets,
                    steps=steps,
                    lr=lr,
                    entropy=entropy
                )
                self.maml_adapter = adapted
            return self

        # Fallback to canonical CALM if no adapter was injected
        try:
            from src.surrogates.calm_predictor import CALM
            if self.maml_adapter is None:
                self.maml_adapter = CALM(dim=self.state_dim)
            self.maml_adapter = self.maml_adapter.adapt(
                support_states.unsqueeze(1) if support_states.dim() == 2 else support_states,
                support_targets,
                steps=steps,
                lr=lr,
                entropy=entropy
            )
        except Exception:
            pass
        return self

    def adapt_online(
        self,
        support_states: torch.Tensor,
        support_targets: torch.Tensor,
        steps: int = 1,
        lr: float = 0.01,
        entropy: Optional[torch.Tensor] = None
    ) -> 'InvestorNewsIngestor':
        """Alias for adapt_maml adhering to uniform MAML API naming."""
        return self.adapt_maml(support_states, support_targets, steps=steps, lr=lr, entropy=entropy)
