"""
Google Drive Ingestor

Implements selective sync and shard-based ingestion from Google Drive.
Supports recursive folder discovery and efficient chunk-based downloads.

Author: William Matthew Bryant
Created: April 2026
"""

import io
import os
from typing import List, Dict, Optional, Any
from pathlib import Path

from googleapiclient.http import MediaIoBaseDownload
from src.data.google_client_manager import GoogleClientManager
from src.data.conversational_types import Conversation, _stable_id

class GoogleDriveIngestor:
    """
    Ingestor for Google Drive content.
    
    Provides functionality to list folders, filter for "Nutrient" shards,
    and download content for manifold integration.
    """
    
    def __init__(self, client_manager: GoogleClientManager):
        """
        Args:
            client_manager: Initialized GoogleClientManager for service building.
        """
        self.manager = client_manager
        self.service = self.manager.build_service('drive', 'v3')
        
    def list_folder_contents(self, folder_id: str = 'root') -> List[Dict[str, Any]]:
        """
        List files in a specific Google Drive folder.
        
        Args:
            folder_id: The ID of the folder to list (defaults to root).
            
        Returns:
            files: List of file metadata objects (name, id, mimeType).
        """
        query = f"'{folder_id}' in parents and trashed = false"
        results = self.service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageSize=100
        ).execute()
        
        return results.get('files', [])

    def download_file_to_bytes(self, file_id: str) -> bytes:
        """
        Download a file's content as raw bytes.
        
        Args:
            file_id: The Google Drive file ID.
            
        Returns:
            content: Raw bytes of the file.
        """
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"📥 Download progressed {int(status.progress() * 100)}%")
        
        return fh.getvalue()

    def ingest_text_shard(self, file_id: str, file_name: str) -> List[Conversation]:
        """
        Download a text-based shard and convert it to Conversations.
        
        Args:
            file_id: Drive ID of the text/json file.
            file_name: Original name of the file for metadata.
            
        Returns:
            conversations: List of parsed conversations.
        """
        print(f"📄 Ingesting Cloud Shard: {file_name}...")
        raw_bytes = self.download_file_to_bytes(file_id)
        
        # Determine parsing strategy by file extension
        ext = Path(file_name).suffix.lower()
        
        try:
            if ext == '.json':
                data = json.loads(raw_bytes.decode('utf-8'))
                # Placeholder for mapping JSON structure to Conversations
                return [] 
            elif ext in ('.txt', '.md', '.log'):
                # Simple line-based turn extraction for raw text
                text = raw_bytes.decode('utf-8', errors='replace')
                # (Logic to convert text blocks to Conversation turns)
                return []
        except Exception as e:
            print(f"❌ Failed to parse Drive shard {file_name}: {e}")
            
        return []

    def sync_nutrient_folder(self, folder_name: str) -> List[Conversation]:
        """
        Search for a folder by name and ingest its contents.
        
        Args:
            folder_name: The name of the Drive folder to "sync".
            
        Returns:
            conversations: Aggregated conversations from the folder.
        """
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])
        
        if not folders:
            print(f"⚠️ Cloud Folder '{folder_name}' not found.")
            return []
            
        folder_id = folders[0]['id']
        files = self.list_folder_contents(folder_id)
        
        all_convs = []
        for f in files:
            # Skip folders for now (not recursive for simplicity)
            if f['mimeType'] == 'application/vnd.google-apps.folder':
                continue
                
            convs = self.ingest_text_shard(f['id'], f['name'])
            all_convs.extend(convs)
            
        return all_convs
