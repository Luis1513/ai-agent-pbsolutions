"""
Complete Pipeline Runner

This module orchestrates the complete data processing pipeline from JSON files to Pinecone.
It coordinates document processing, embedding generation, and vector database upload.

The module handles:
    - JSON document processing and chunking
    - OpenAI embedding generation
    - Pinecone vector database upload
    - Progress tracking and timing metrics
    - Pipeline status reporting
"""

from pathlib import Path
from .document_processor import process_json_documents, save_chunks_to_json
from .embedding_processor import generate_embeddings_for_chunks, save_embeddings_to_json
from .pinecone_uploader import upload_to_pinecone



#*****************************************************************************************************#
def run_complete_pipeline(data_folder: str = "data", chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Run the complete pipeline from start to finish.
    
    Args: data_folder: Folder containing JSON files
           chunk_size: Size of each chunk
           chunk_overlap: Overlap between chunks
    """
    print("🚀 Starting complete pipeline...")
    print("=" * 50)
    
    # Step 1: Process JSON documents into chunks
    print("\n📄 STEP 1: Processing JSON documents into chunks...")
    
    chunks = process_json_documents(data_folder=data_folder, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print("❌ No chunks generated. Pipeline stopped.")
        return False
    
    # Save processed chunks to JSON file
    save_chunks_to_json(chunks, output_file="data/processed_chunks.json")
    print(f"✅ Chunks generated: {len(chunks)}")
    
    # Step 2: Generate embeddings for chunks
    print("\n🧠 STEP 2: Generating embeddings...")
    
    chunks_with_embeddings = generate_embeddings_for_chunks(chunks)
    if not chunks_with_embeddings:
        print("❌ No embeddings generated. Pipeline stopped.")
        return False
    
    print(f"✅ Embeddings generated: {len(chunks_with_embeddings)}")
    
    # Save embeddings to JSON file
    save_embeddings_to_json(chunks_with_embeddings, output_file="data/embeddings_processed.json")
    
    # Step 3: Upload to Pinecone
    print("\n☁️ STEP 3: Uploading to Pinecone...")
    
    upload_success = upload_to_pinecone(chunks_with_embeddings)
    if not upload_success:
        print("❌ Pinecone upload failed. Pipeline stopped.")
        return False
    
    print(f"✅ Pinecone upload completed")
    
    # Generate comprehensive pipeline summary
    print("\n" + "=" * 50)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📊 Summary:")
    print(f"   • Chunks created: {len(chunks)}")
    print(f"   • Embeddings generated: {len(chunks_with_embeddings)}")
    print(f"   • Vectors uploaded to Pinecone: {len(chunks_with_embeddings)}")
    print("=" * 50)
    
    return True
#*****************************************************************************************************#



#*****************************************************************************************************#
def main():
    """
    Main function to run the pipeline.
    """
    # Execute the complete pipeline with default settings
    success = run_complete_pipeline(data_folder="data")
    
    if success:
        print("   • Your knowledge base is now ready in Pinecone")
    else:
        print("\n❌ Pipeline failed. Check the logs above for details.")
#*****************************************************************************************************#


#*****************************************************************************************************#
if __name__ == "__main__":
    main()
