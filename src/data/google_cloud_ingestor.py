"""
Google Cloud Ingestor

Implements ingestion from Google Cloud Storage (GCS) and BigQuery.
Allows the Reasoner to pull tabular and object-based data from cloud manifolds.

Author: William Matthew Bryant
Created: April 2026
"""

from typing import List, Dict, Optional, Any
from src.data.google_client_manager import GoogleClientManager
from src.data.conversational_types import Conversation, ConversationTurn, _stable_id

class GoogleCloudIngestor:
    """
    Ingestor for Google Cloud Platform services.
    
    Provides specialized methods for BigQuery and GCS data extraction.
    """
    
    def __init__(self, client_manager: GoogleClientManager):
        """
        Args:
            client_manager: Initialized GoogleClientManager for service building.
        """
        self.manager = client_manager
        self.bq_service = self.manager.build_service('bigquery', 'v2')
        self.gcs_service = self.manager.build_service('storage', 'v1')
        
    def query_bigquery(self, project_id: str, query: str) -> List[Conversation]:
        """
        Execute a BigQuery query and convert results to Conversations.
        
        Args:
            project_id: The GCP project ID to bill the query to.
            query: The SQL query string.
            
        Returns:
            conversations: List of conversations derived from rows.
        """
        print(f"📊 Running Sovereign Query on BigQuery manifold...")
        body = {'query': query, 'useLegacySql': False}
        
        try:
            results = self.bq_service.jobs().query(
                projectId=project_id,
                body=body
            ).execute()
            
            rows = results.get('rows', [])
            fields = [f['name'] for f in results['schema']['fields']]
            
            conversations = []
            for row_data in rows:
                row = [v['v'] for v in row_data['f']]
                mapping = dict(zip(fields, row))
                
                # Heuristic mapping: look for 'text', 'body', or 'content'
                text = mapping.get('text') or mapping.get('body') or mapping.get('content') or str(mapping)
                speaker = mapping.get('user_id') or mapping.get('author') or 'bq_row'
                
                turns = [ConversationTurn(speaker_id=speaker, text=text)]
                
                conversations.append(Conversation(
                    conversation_id=_stable_id("bq", f"{project_id}_{hash(text)}"),
                    turns=turns,
                    context={'query': query, 'source_row': mapping},
                    source='google_bigquery'
                ))
            
            return conversations
        except Exception as e:
            print(f"❌ BigQuery Ingestion failed: {e}")
            return []

    def list_gcs_objects(self, bucket_name: str, prefix: str = '') -> List[str]:
        """
        List object names in a GCS bucket.
        """
        results = self.gcs_service.objects().list(
            bucket=bucket_name,
            prefix=prefix
        ).execute()
        
        return [obj['name'] for obj in results.get('items', [])]

    def ingest_gcs_blob(self, bucket_name: str, object_name: str) -> List[Conversation]:
        """
        Download a GCS blob and process its content.
        """
        print(f"☁️ Ingesting GCS Blob: {object_name}...")
        try:
            request = self.gcs_service.objects().get_media(
                bucket=bucket_name,
                object=object_name
            )
            content = request.execute().decode('utf-8', errors='replace')
            
            # Logic to parse the content based on suffix (similar to DriveIngestor)
            return []
        except Exception as e:
            print(f"❌ GCS Ingestion failed for {object_name}: {e}")
            return []
