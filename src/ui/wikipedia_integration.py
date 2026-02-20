#!/usr/bin/env python3
"""
Wikipedia Integration Module

Provides enhanced Wikipedia content extraction and processing capabilities
that integrate with the existing Gyroidic system. This module bridges the
gap between individual Wikipedia page fetching and WikiExtractor's processing.
"""

import requests
import re
import json
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote
import sys
import os

# Add WikiExtractor to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'wikipedia extracter', 'wikiextractor-3.0.7'))

try:
    from wikiextractor.clean import clean
    WIKIEXTRACTOR_AVAILABLE = True
except ImportError:
    WIKIEXTRACTOR_AVAILABLE = False
    print("[WIKI] WikiExtractor not available, using fallback cleaning")

class WikipediaIntegration:
    """
    Enhanced Wikipedia integration that combines API fetching with WikiExtractor processing.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GyroidicSystem/1.0 (Educational Research)'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Statistics
        self.stats = {
            'pages_processed': 0,
            'total_chars_extracted': 0,
            'total_chars_filtered': 0,
            'failed_requests': 0
        }
    
    def extract_title_from_url(self, url: str) -> str:
        """Extract Wikipedia page title from URL."""
        match = re.search(r'/wiki/([^#?]+)', url)
        if match:
            return unquote(match[1]).replace('_', ' ')
        return url
    
    def is_wikipedia_url(self, url: str) -> bool:
        """Check if URL is a Wikipedia URL."""
        return 'wikipedia.org/wiki/' in url
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract Wikipedia URLs from text."""
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        return [url for url in urls if self.is_wikipedia_url(url)]
    
    def rate_limit(self):
        """Apply rate limiting to avoid overwhelming Wikipedia API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def fetch_wikipedia_content(self, title: str) -> Optional[Dict]:
        """
        Fetch Wikipedia content using multiple API endpoints for comprehensive extraction.
        
        Args:
            title: Wikipedia page title
            
        Returns:
            Dictionary with page content or None if failed
        """
        self.rate_limit()
        
        try:
            # Method 1: Try Wikipedia REST API (summary)
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
            
            response = self.session.get(summary_url, timeout=10)
            if response.status_code == 200:
                summary_data = response.json()
                
                # Method 2: Get full content using MediaWiki API
                content_url = "https://en.wikipedia.org/w/api.php"
                content_params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': title,
                    'prop': 'extracts',
                    'exintro': False,  # Get full content, not just intro
                    'explaintext': True,  # Plain text, no HTML
                    'exsectionformat': 'plain'
                }
                
                self.rate_limit()
                content_response = self.session.get(content_url, params=content_params, timeout=15)
                
                if content_response.status_code == 200:
                    content_data = content_response.json()
                    pages = content_data.get('query', {}).get('pages', {})
                    
                    if pages:
                        page_data = next(iter(pages.values()))
                        full_extract = page_data.get('extract', '')
                        
                        # Combine summary and full content
                        result = {
                            'title': summary_data.get('title', title),
                            'extract': summary_data.get('extract', ''),
                            'full_content': full_extract,
                            'url': summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            'content_length': len(full_extract),
                            'method': 'api_combined'
                        }
                        
                        self.stats['pages_processed'] += 1
                        self.stats['total_chars_extracted'] += len(full_extract)
                        
                        return result
            
            # Fallback: Use only summary if full content fails
            if response.status_code == 200:
                data = response.json()
                extract = data.get('extract', '')
                
                result = {
                    'title': data.get('title', title),
                    'extract': extract,
                    'full_content': extract,  # Use extract as full content
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'content_length': len(extract),
                    'method': 'api_summary_only'
                }
                
                self.stats['pages_processed'] += 1
                self.stats['total_chars_extracted'] += len(extract)
                
                return result
            
            else:
                print(f"⚠️  Wikipedia API error for '{title}': {response.status_code}")
                self.stats['failed_requests'] += 1
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error fetching '{title}': {e}")
            self.stats['failed_requests'] += 1
            return None
        except Exception as e:
            print(f"⚠️  Unexpected error fetching '{title}': {e}")
            self.stats['failed_requests'] += 1
            return None
    
    def clean_wikipedia_content(self, content: str, title: str = "") -> str:
        """
        Clean Wikipedia content using WikiExtractor if available, otherwise use fallback.
        
        Args:
            content: Raw Wikipedia content
            title: Page title for context
            
        Returns:
            Cleaned content
        """
        if not content:
            return ""
        
        original_length = len(content)
        
        if WIKIEXTRACTOR_AVAILABLE:
            try:
                # Use WikiExtractor's cleaning function
                cleaned = clean(content)
                cleaned_length = len(cleaned)
                filtered_chars = original_length - cleaned_length
                self.stats['total_chars_filtered'] += filtered_chars
                return cleaned
            except Exception as e:
                print(f"⚠️  WikiExtractor cleaning failed for '{title}': {e}")
                # Fall through to fallback cleaning
        
        # Fallback cleaning (enhanced version of existing filter)
        cleaned = self._fallback_clean_content(content)
        cleaned_length = len(cleaned)
        filtered_chars = original_length - cleaned_length
        self.stats['total_chars_filtered'] += filtered_chars
        
        return cleaned
    
    def _fallback_clean_content(self, content: str) -> str:
        """
        Fallback content cleaning when WikiExtractor is not available.
        Enhanced version of the existing Wikipedia noise filtering.
        """
        # Step 1: Remove Wikipedia-specific markup and references
        
        # Remove reference brackets [1], [2], etc. but preserve math [x+y]
        # Protect mathematical expressions first
        math_patterns = [
            r'\[[\d\+\-\*\/\^\(\)\s,\.]+\]',  # [1+2], [0,1], [x^2]
            r'\[[A-Za-z]\s*[=\+\-\*\/]\s*[A-Za-z\d]+\]',  # [x=5], [a+b]
            r'\[\s*\d+\s*,\s*\d+\s*\]',  # [1,2], [0, 1]
            r'\[.*?(?:matrix|equation|formula|theorem|proof).*?\]',  # Mathematical contexts
        ]
        
        protected_spans = []
        for pattern in math_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                protected_spans.append((match.start(), match.end()))
        
        # Remove numeric references [1], [2], [123] if not protected
        def is_protected(start, end):
            for p_start, p_end in protected_spans:
                if start >= p_start and end <= p_end:
                    return True
            return False
        
        # Remove unprotected numeric references
        ref_pattern = r'\[\s*\d+\s*\]'
        matches = list(re.finditer(ref_pattern, content))
        for match in reversed(matches):
            if not is_protected(match.start(), match.end()):
                content = content[:match.start()] + content[match.end():]
        
        # Remove citation-style references
        citation_patterns = [
            r'\[citation needed\]',
            r'\[needs citation\]',
            r'\[source\?\]',
            r'\[clarification needed\]',
            r'\[when\?\]',
            r'\[who\?\]',
            r'\[where\?\]',
            r'\[dubious.*?\]',
            r'\[verify.*?\]',
            r'\[original research\?\]',
        ]
        
        for pattern in citation_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove author references like [Smith 2020], [Jones et al. 2019]
        author_ref_pattern = r'\[[A-Z][a-z]+(?:\s+et\s+al\.?)?\s+\d{4}[a-z]?\]'
        content = re.sub(author_ref_pattern, '', content)
        
        # Step 2: Clean up formatting
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove orphaned punctuation from removed references
        content = re.sub(r'\s*,\s*,', ',', content)  # Double commas
        content = re.sub(r'\s*\.\s*\.', '.', content)  # Double periods
        content = re.sub(r'\s+([,.;:])', r'\1', content)  # Space before punctuation
        
        # Ensure sentences don't run together
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)
        
        # Step 3: Remove common Wikipedia artifacts
        # Remove "edit" links and section markers
        content = re.sub(r'\[edit\]', '', content, flags=re.IGNORECASE)
        
        # Remove coordinate references
        content = re.sub(r'Coordinates:\s*\d+°[^.]*\.', '', content)
        
        # Remove "See also" and "References" section headers if they appear at the end
        content = re.sub(r'\n\s*(?:See also|References|External links|Further reading)\s*\n.*$', '', content, flags=re.IGNORECASE | re.DOTALL)
        
        return content.strip()
    
    def extract_key_concepts(self, title: str, content: str) -> List[str]:
        """
        Extract key concepts from Wikipedia page for association learning.
        
        Args:
            title: Page title
            content: Page content
            
        Returns:
            List of key concepts
        """
        concepts = []
        
        # Primary concept: the title itself
        concepts.append(title)
        
        # Extract concepts from title
        title_concepts = self._extract_concepts_from_title(title)
        concepts.extend(title_concepts)
        
        # Extract concepts from content (first paragraph)
        content_concepts = self._extract_concepts_from_content(content)
        concepts.extend(content_concepts)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept.lower() not in seen and len(concept) > 2:
                seen.add(concept.lower())
                unique_concepts.append(concept)
        
        return unique_concepts[:10]  # Limit to top 10 concepts
    
    def _extract_concepts_from_title(self, title: str) -> List[str]:
        """Extract concepts from page title."""
        concepts = []
        
        # Split by common separators
        parts = re.split(r'[,\-\(\):]', title)
        for part in parts:
            cleaned = part.strip()
            if len(cleaned) > 3:
                concepts.append(cleaned)
        
        # Extract meaningful words (longer than 4 characters)
        words = title.split()
        for word in words:
            if len(word) > 4 and not re.match(r'^(the|and|or|of|in|on|at|to|for|with|by)$', word, re.IGNORECASE):
                concepts.append(word)
        
        return concepts
    
    def _extract_concepts_from_content(self, content: str) -> List[str]:
        """Extract key concepts from content (first paragraph)."""
        concepts = []
        
        if not content:
            return concepts
        
        # Get first paragraph (up to first double newline or first 500 chars)
        first_para = content.split('\n\n')[0][:500]
        
        # Extract capitalized phrases (likely proper nouns/concepts)
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', first_para)
        concepts.extend(capitalized_phrases[:5])  # Top 5 capitalized phrases
        
        return concepts
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'pages_processed': 0,
            'total_chars_extracted': 0,
            'total_chars_filtered': 0,
            'failed_requests': 0
        }

# Global instance
wikipedia_integration = WikipediaIntegration()