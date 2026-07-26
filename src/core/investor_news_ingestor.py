"""
Investor News Ingestion Subsystem with Robots.txt Leniency Validation.

Role: Programmatically scrapes investor news from lenient endpoints to harvest
topological features and market sentiments, avoiding lobotomizing constraints.
"""

import urllib.request
import urllib.robotparser
from urllib.parse import urlparse
import json
import os
import re
from typing import List, Dict, Any, Optional

class InvestorNewsIngestor:
    def __init__(self, user_agent: str = "GyroidicFluxReasonerBot/1.9"):
        self.user_agent = user_agent
        self.rp_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}

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
            # If robots.txt is missing or unreachable, assume lenient but cautious
            rp.allow_all = True
            
        self.rp_cache[robots_url] = rp
        return rp

    def is_allowed(self, url: str) -> bool:
        """Verify robots.txt allows crawling the given URL."""
        try:
            rp = self._get_robot_parser(url)
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True  # Fallback to true if parsing completely fails to ensure resilience

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
                
            # Regex parser for XML items to avoid heavy external dependencies
            items = []
            item_blocks = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
            for block in item_blocks:
                title_match = re.search(r'<title>(.*?)</title>', block, re.DOTALL)
                link_match = re.search(r'<link>(.*?)</link>', block, re.DOTALL)
                desc_match = re.search(r'<description>(.*?)</description>', block, re.DOTALL)
                
                title = title_match.group(1).strip() if title_match else ""
                link = link_match.group(1).strip() if link_match else ""
                desc = desc_match.group(1).strip() if desc_match else ""
                
                # Clean CDATA
                title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title)
                desc = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', desc)
                # Strip HTML tags
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

    def ingest_market_topologies(self, feed_urls: List[str]) -> List[str]:
        """Harvests market news items and converts them to raw text payloads."""
        payloads = []
        for url in feed_urls:
            print(f"[INGESTOR] Ingesting from market endpoint: {url} ...")
            items = self.fetch_rss_feed(url)
            for item in items:
                # Format news text to be ingested as topological information
                formatted = f"MARKET NEWS: {item['title']}. {item['description']}"
                payloads.append(formatted)
        return payloads
