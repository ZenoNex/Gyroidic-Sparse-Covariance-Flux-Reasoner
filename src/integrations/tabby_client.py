"""
Tabby ML Client — Connects to a local Tabby ML instance for code completion
and chat, following the OpenAI-compatible API format.

Tabby ML runs locally and provides:
  - /v1/completions — code completion
  - /v1/chat/completions — chat-style interaction
  - /v1/health — health check

No API token needed for local Tabby instances.
"""

import json
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TabbyConfig:
    """Configuration for connecting to a Tabby ML instance."""
    host: str = "localhost"
    port: int = 8080
    timeout: int = 30
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'base_url': self.base_url,
            'timeout': self.timeout,
        }


@dataclass  
class TabbyResponse:
    """Response from a Tabby ML API call."""
    success: bool
    text: str = ""
    model: str = ""
    usage: Dict[str, int] = None
    raw: Dict[str, Any] = None
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'success': self.success,
            'text': self.text,
            'model': self.model,
        }
        if self.usage:
            result['usage'] = self.usage
        if self.error:
            result['error'] = self.error
        return result


class TabbyClient:
    """
    Client for interacting with a local Tabby ML instance.
    Uses stdlib urllib only — no external dependencies.
    """
    
    def __init__(self, config: TabbyConfig = None):
        self.config = config or TabbyConfig()
        self._connected = False
        self._server_info = {}
    
    def _make_request(
        self,
        path: str,
        method: str = 'GET',
        data: Dict = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Tabby ML server."""
        url = f"{self.config.base_url}{path}"
        
        if data is not None:
            body = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=body,
                headers={'Content-Type': 'application/json'},
                method=method,
            )
        else:
            req = urllib.request.Request(url, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                response_data = resp.read().decode('utf-8')
                if response_data:
                    return json.loads(response_data)
                return {'status': resp.status}
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            raise ConnectionError(f"HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot reach Tabby at {url}: {e.reason}")
        except Exception as e:
            raise ConnectionError(f"Request failed: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the Tabby ML server.
        
        Returns:
            Dict with 'connected', 'message', and optional 'server_info'
        """
        try:
            # Try /v1/health first (standard Tabby endpoint)
            info = self._make_request('/v1/health')
            self._connected = True
            self._server_info = info
            return {
                'connected': True,
                'message': f'Connected to Tabby ML at {self.config.base_url}',
                'server_info': info,
            }
        except ConnectionError:
            pass
        
        try:
            # Fall back to root endpoint
            info = self._make_request('/')
            self._connected = True
            self._server_info = info
            return {
                'connected': True,
                'message': f'Connected to Tabby ML at {self.config.base_url}',
                'server_info': info,
            }
        except ConnectionError as e:
            self._connected = False
            return {
                'connected': False,
                'message': str(e),
                'server_info': {},
            }
    
    def complete(
        self,
        prompt: str,
        language: str = 'python',
        max_tokens: int = 256,
        temperature: float = 0.2,
        stop: List[str] = None,
    ) -> TabbyResponse:
        """
        Get a code completion from Tabby ML.
        
        Args:
            prompt: Code prefix to complete
            language: Programming language hint
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            
        Returns:
            TabbyResponse with the completion text
        """
        payload = {
            'prompt': prompt,
            'language': language,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        if stop:
            payload['stop'] = stop
        
        try:
            # First try Tabby-native completion endpoint
            result = self._make_request('/v1/completions', method='POST', data=payload)
            
            # Parse response (Tabby format)
            if 'choices' in result:
                choice = result['choices'][0]
                text = choice.get('text', choice.get('message', {}).get('content', ''))
            elif 'id' in result and 'choices' in result:
                text = result['choices'][0].get('text', '')
            else:
                text = result.get('text', str(result))
            
            return TabbyResponse(
                success=True,
                text=text,
                model=result.get('model', 'tabby'),
                usage=result.get('usage', {}),
                raw=result,
            )
        except ConnectionError as e:
            return TabbyResponse(success=False, error=str(e))
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> TabbyResponse:
        """
        Send a chat-style request to Tabby ML.
        
        Args:
            messages: List of {role: str, content: str} dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            TabbyResponse with the assistant reply
        """
        payload = {
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        
        try:
            result = self._make_request('/v1/chat/completions', method='POST', data=payload)
            
            if 'choices' in result:
                choice = result['choices'][0]
                text = choice.get('message', {}).get('content', choice.get('text', ''))
            else:
                text = result.get('text', str(result))
            
            return TabbyResponse(
                success=True,
                text=text,
                model=result.get('model', 'tabby'),
                usage=result.get('usage', {}),
                raw=result,
            )
        except ConnectionError as e:
            return TabbyResponse(success=False, error=str(e))
    
    def generate_training_sample(
        self,
        topic: str,
        style: str = 'textbook',
        language: str = 'python',
    ) -> TabbyResponse:
        """
        Generate a synthetic training sample by prompting Tabby ML.
        
        Uses structured prompts to create textbook-quality code examples
        for training the gyroidic system.
        
        Args:
            topic: The concept or algorithm to generate an example for
            style: 'textbook' (explained code), 'exercise' (problem+solution), 'qa' (Q&A)
            language: Programming language
        
        Returns:
            TabbyResponse with the generated sample
        """
        prompts = {
            'textbook': (
                f"Write a clear, self-contained {language} example that teaches the concept of {topic}. "
                f"Include docstrings, comments, and a brief explanation. "
                f"The code should be educational and runnable."
            ),
            'exercise': (
                f"Create a programming exercise about {topic} in {language}. "
                f"First state the problem, then provide the solution with explanation."
            ),
            'qa': (
                f"Question: Explain {topic} and provide a {language} implementation.\n"
                f"Answer:"
            ),
        }
        
        prompt = prompts.get(style, prompts['textbook'])
        
        messages = [
            {'role': 'system', 'content': 'You are a helpful programming instructor. Write clear, educational code.'},
            {'role': 'user', 'content': prompt},
        ]
        
        return self.chat(messages, max_tokens=1024, temperature=0.4)
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def server_info(self) -> Dict[str, Any]:
        return self._server_info
