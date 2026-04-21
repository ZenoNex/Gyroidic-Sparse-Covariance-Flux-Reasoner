"""
Google API Client Manager

Manages OAuth 2.0 credentials and service building for the Gyroidic Reasoner.
Handles token persistence and service initialization for GCS, BigQuery, and Drive.

Author: William Matthew Bryant
Created: April 2026
"""

import os
import json
import pickle
from typing import List, Optional, Any
from pathlib import Path

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource

class GoogleClientManager:
    """
    Manager for Google API credentials and client services.
    
    Handles the lifecycle of OAuth tokens and provides a consistent
    interface for building Google service clients.
    """
    
    def __init__(
        self,
        client_secrets_path: str,
        token_path: Optional[str] = None,
        scopes: Optional[List[str]] = None
    ):
        """
        Args:
            client_secrets_path: Path to the client_secret.json file.
            token_path: Path to store/load the cached token (e.g., token.pickle).
            scopes: List of required OAuth scopes.
        """
        self.client_secrets_path = Path(client_secrets_path)
        
        # Default token path in the same directory as secrets if not provided
        if token_path is None:
            self.token_path = self.client_secrets_path.parent / 'token.pickle'
        else:
            self.token_path = Path(token_path)
            
        # Default scopes if not provided (aligned with user request)
        if scopes is None:
            self.scopes = [
                'https://www.googleapis.com/auth/drive.readonly',
                'https://www.googleapis.com/auth/bigquery.readonly',
                'https://www.googleapis.com/auth/cloud-platform.read-only',
                'https://www.googleapis.com/auth/devstorage.read_only'
            ]
        else:
            self.scopes = scopes
            
        self.creds = None
        self._load_credentials()

    def _load_credentials(self):
        """Load credentials from disk or run the auth flow."""
        # The file token.pickle stores the user's access and refresh tokens
        if self.token_path.exists():
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)
                
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                print("🔄 Refreshing expired Google credentials...")
                self.creds.refresh(Request())
            else:
                if not self.client_secrets_path.exists():
                    raise FileNotFoundError(f"❌ Client secret not found at {self.client_secrets_path}")
                
                print("🌐 Initializing Google OAuth Flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.client_secrets_path), 
                    self.scopes
                )
                # This opens a local server for the consent flow
                self.creds = flow.run_local_server(port=0)
                
            # Save the credentials for the next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(self.creds, token)
                print(f"💾 Credentials saved to {self.token_path}")

    def build_service(self, service_name: str, version: str = 'v1') -> Resource:
        """
        Build a Google API service client.
        
        Args:
            service_name: Name of the service (e.g., 'drive', 'bigquery', 'storage').
            version: API version (e.g., 'v3' for Drive).
            
        Returns:
            service: Initialized Google discovery resource.
        """
        if not self.creds or not self.creds.valid:
            self._load_credentials()
            
        # Specific overrides for service-unique versioning/naming
        if service_name == 'drive':
            version = 'v3'
        elif service_name == 'bigquery':
            version = 'v2'
        elif service_name == 'storage':
            service_name = 'storage'
            version = 'v1'
                
        return build(service_name, version, credentials=self.creds)

    def get_token_status(self) -> dict:
        """Return diagnostic info about the current token state."""
        if not self.creds:
            return {'status': 'missing'}
        return {
            'status': 'valid' if self.creds.valid else 'expired',
            'has_refresh': bool(self.creds.refresh_token),
            'scopes': self.creds.scopes,
            'expiry': str(self.creds.expiry) if hasattr(self.creds, 'expiry') else 'unknown'
        }
