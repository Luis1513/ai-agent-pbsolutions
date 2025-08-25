"""
Pinecone Uploader for Embeddings

This module uploads processed chunks with embeddings to Pinecone vector database.
It handles batch processing and provides progress tracking for large uploads.

The module handles:
    - Loading processed embeddings from JSON
    - Batch uploading to Pinecone index
    - Progress tracking and error handling
    - Connection management to Pinecone
"""

# ====================================================================================================== #
import json
from pathlib import Path
from typing import List, Dict, Any
from pinecone import Pinecone
from app.core.settings import settings
# ====================================================================================================== #



# ====================================================================================================== #
# Load Processed Embeddings from JSON file
# ====================================================================================================== #
def load_embeddings(embeddings_file: str = "data/embeddings_processed.json") -> List[Dict[str, Any]]:
    """
    Args: embeddings_file: Path to embeddings file
        
    Returns: List of chunks with embeddings ready for Pinecone
    """
    try:
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        print(f"Loaded {len(embeddings)} embeddings from {embeddings_file}")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return []
# ====================================================================================================== #



# ====================================================================================================== #
# Upload Embeddings to Pinecone
# ====================================================================================================== #
def upload_to_pinecone(embeddings: List[Dict[str, Any]], batch_size: int = 100) -> bool:
    """
    Args: embeddings: List of embeddings ready for Pinecone
          batch_size: Number of vectors to upload per batch
        
    Returns: bool: True if successful, False otherwise
    """
    try:
        # Initialize Pinecone client with API key from settings
        pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
        
        # Get the specified index from settings
        index = pinecone_client.Index(settings.pinecone_index)
        
        print(f"Connected to Pinecone index: {settings.pinecone_index}")
        
        # Upload embeddings in configurable batches
        total_uploaded = 0
        
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            
            try:
                # Prepare batch data for Pinecone upsert
                vectors = []
                for embedding in batch:
                    vector_data = {
                        'id': embedding['id'],
                        'values': embedding['values'],
                        'metadata': embedding['metadata']
                    }
                    vectors.append(vector_data)
                
                # Upload batch to Pinecone using upsert
                index.upsert(vectors=vectors)
                
                total_uploaded += len(batch)
                print(f"Uploaded batch {i//batch_size + 1}: {len(batch)} vectors (Total: {total_uploaded})")
                
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"Upload completed! Total vectors uploaded: {total_uploaded}")
        return True
        
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return False
# ====================================================================================================== #



# ====================================================================================================== #
# Main function to upload embeddings to Pinecone
# ====================================================================================================== #
def main():

    # Load embeddings from JSON file
    embeddings = load_embeddings()
    if not embeddings:
        print("No embeddings to upload")
        return
    
    # Upload all embeddings to Pinecone
    print(f"Starting upload of {len(embeddings)} vectors to Pinecone...")
    upload_success = upload_to_pinecone(embeddings)
    
    if upload_success:
        # Confirm successful upload completion
        print("Pinecone upload completed successfully!")
    else:
        print("Pinecone upload failed!")
# ====================================================================================================== #



# ====================================================================================================== #
# Main execution block
# ====================================================================================================== #
if __name__ == "__main__":
    main()
# ====================================================================================================== #