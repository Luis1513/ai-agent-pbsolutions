"""
Smoke Check Script

This script performs a comprehensive smoke test to verify that the AI agent system
is properly configured and can communicate with external services (OpenAI and Pinecone).
It tests embedding generation, index creation/verification, and vector storage/retrieval.

Dependencies:
    - OpenAI API key configured
    - Pinecone API key configured
    - Required environment variables set in settings
"""

from app.core.settings import settings
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

#*****************************************************************************************************#
def check_openai_embeddings_and_get_dimension():
    """
    Test OpenAI embeddings API and retrieve vector dimension.
    
    Returns: tuple: (embedding_vector, vector_dimension)
        
    Raises: Exception: If OpenAI API call fails
    """
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    # Generate test embedding
    embedding_response = openai_client.embeddings.create(
        model=settings.openai_embedding_model,
        input="hello world"
    )
    
    # Extract embedding vector and dimension
    embedding_vector = embedding_response.data[0].embedding
    vector_dimension = len(embedding_vector)
    
    print(f"[OK] OpenAI embeddings: model={settings.openai_embedding_model}, dim={vector_dimension}")
    
    return embedding_vector, vector_dimension
#*****************************************************************************************************#



#*****************************************************************************************************#
def ensure_pinecone_index_exists(vector_dimension: int):
    """
    Ensure Pinecone index exists with correct configuration.
    
    Args: vector_dimension (int): Dimension of vectors to be stored
        
    Returns: Pinecone.Index: Configured Pinecone index instance
        
    Raises: Exception: If Pinecone operations fail
    """
    # Initialize Pinecone client
    pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    
    # Get list of existing indexes
    existing_indexes = {index["name"] for index in pinecone_client.list_indexes().indexes}
    
    # Create index if it doesn't exist
    if settings.pinecone_index not in existing_indexes:
        print(f"[i] Creating Pinecone index '{settings.pinecone_index}' (dim={vector_dimension})...")
        
        pinecone_client.create_index(
            name=settings.pinecone_index,
            dimension=vector_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud, 
                region=settings.pinecone_region
            ),
        )
    else:
        print(f"[OK] Pinecone index '{settings.pinecone_index}' exists.")
    
    # Return configured index
    return pinecone_client.Index(settings.pinecone_index)
#*****************************************************************************************************#



#*****************************************************************************************************#
def test_vector_storage_and_retrieval(pinecone_index, test_vector):
    """
    Test vector storage and retrieval functionality.
    
    Args:
        pinecone_index: Configured Pinecone index instance
        test_vector: Vector to store and query
        
    Raises: Exception: If vector operations fail
    """
    # Store test vector
    pinecone_index.upsert(
        vectors=[{
            "id": "smoke-test-vector", 
            "values": test_vector, 
            "metadata": {"source": "smoke_test"}
        }]
    )
    
    # Query for the stored vector
    query_result = pinecone_index.query(
        top_k=1, 
        vector=test_vector, 
        include_metadata=True
    )
    
    # Extract and display results
    top_match = query_result.matches[0]
    print(f"[OK] Pinecone query: id={top_match.id}, score={top_match.score:.4f}")
#*****************************************************************************************************#



#*****************************************************************************************************#
def main():
    """
    Main function to execute the complete smoke test.
    
    Tests:
        1. OpenAI embeddings generation
        2. Pinecone index configuration
        3. Vector storage and retrieval
    """
    try:
        # Test OpenAI embeddings and get vector dimension
        test_vector, vector_dimension = check_openai_embeddings_and_get_dimension()
        
        # Ensure Pinecone index is properly configured
        pinecone_index = ensure_pinecone_index_exists(vector_dimension)
        
        # Test vector storage and retrieval
        test_vector_storage_and_retrieval(pinecone_index, test_vector)
        
        print("[DONE] Smoke test completed.")
        
    except Exception as error:
        print(f"[ERROR] Smoke test failed: {str(error)}")
        raise
#*****************************************************************************************************#



#*****************************************************************************************************#
if __name__ == "__main__":
    main()
