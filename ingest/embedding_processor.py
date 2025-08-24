"""
Embedding Processor for Chunks

This module processes text chunks and generates embeddings using OpenAI's embedding models.
It prepares the chunks with embeddings for storage in Pinecone vector database.

The module handles:
    - Loading processed chunks from JSON
    - Generating embeddings using OpenAI API
    - Formatting data for Pinecone storage
    - Saving processed embeddings to JSON
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from app.core.settings import settings



#*****************************************************************************************************#
def load_processed_chunks(chunks_file: str = "data/processed_chunks.json") -> List[Dict[str, Any]]:
    """
    Load processed chunks from JSON file.
    
    Args: chunks_file: Path to processed chunks file
        
    Returns: List of chunks with text and metadata
    """
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks
    except Exception as e:
        print(f"Error loading chunks: {e}")
        return []
#*****************************************************************************************************#



#*****************************************************************************************************#
def generate_embeddings_for_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for chunks using OpenAI.
    
    Args: chunks: List of chunks with text and metadata
        
    Returns: List of chunks with embeddings and metadata ready for Pinecone
    """
    # Initialize OpenAI client with API key from settings
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    chunks_with_embeddings = []
    
    # Process each chunk to generate embeddings
    for i, chunk in enumerate(chunks):
        try:
            # Generate embedding using OpenAI API
            embedding_response = openai_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=chunk['text']
            )
            
            # Extract embedding vector from response
            embedding_vector = embedding_response.data[0].embedding
            
            # Format chunk data for Pinecone storage
            chunk_data = {
                'id': chunk['chunk_id'],
                'values': embedding_vector,
                'metadata': {
                    'text': chunk['text'],
                    'sources': chunk['sources'],
                    'section': chunk['section']
                }
            }
            
            chunks_with_embeddings.append(chunk_data)
            
            # Show progress every 10 chunks
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks")
                
        except Exception as e:
            print(f"Error generating embedding for chunk {chunk['chunk_id']}: {e}")
            continue
    
    print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    return chunks_with_embeddings
#*****************************************************************************************************#



#*****************************************************************************************************#
def save_embeddings_to_json(chunks_with_embeddings: List[Dict[str, Any]], output_file: str = "data/embeddings_processed.json"):
    """
    Save chunks with embeddings to JSON file.
    
    Args:
        chunks_with_embeddings: List of chunks with embeddings
        output_file: Output file name
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)
        print(f"Embeddings saved to {output_file}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
#*****************************************************************************************************#



#*****************************************************************************************************#
def main():
    """
    Main function to process chunks and generate embeddings.
    """
    # Load processed chunks from JSON file
    chunks = load_processed_chunks()
    if not chunks:
        print("No chunks to process")
        return
    
    # Generate embeddings for all chunks
    chunks_with_embeddings = generate_embeddings_for_chunks(chunks)
    if chunks_with_embeddings:
        # Save processed embeddings ready for Pinecone
        save_embeddings_to_json(chunks_with_embeddings)
        print("Embedding generation completed successfully!")
    else:
        print("No embeddings were generated")
#*****************************************************************************************************#



#*****************************************************************************************************#
if __name__ == "__main__":
    main()
