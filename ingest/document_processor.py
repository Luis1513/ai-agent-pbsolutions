"""
Document Processor for JSON Files

This module provides functionality for processing JSON documents and creating text chunks using LangChain's RecursiveCharacterTextSplitter. It reads JSON files from a specified
data folder, extracts text content, splits it into manageable chunks, and saves the processed chunks with metadata for further processing.

The module handles:
    - JSON file reading and parsing
    - Text chunking with configurable size and overlap
    - Metadata preservation and chunk identification
    - Error handling and logging
    - Output file generation

Usage:
    - Process documents: process_json_documents(data_folder, chunk_size, chunk_overlap)
    - Save chunks: save_chunks_to_json(chunks, output_file)
    - Standalone execution: python document_processor.py
"""

# ====================================================================================================== #
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ====================================================================================================== #



# ====================================================================================================== #
# Process JSON Documents
# ====================================================================================================== #
def process_json_documents(data_folder: str = "Data", chunk_size: int = 750, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    """
    This function reads all JSON files from the specified data folder, extracts text content from each file, and splits the text into chunks using LangChain's
    RecursiveCharacterTextSplitter. Each chunk maintains metadata including sources, section information, and a unique chunk identifier.
    
    Args:
        data_folder: Path to folder containing JSON files (default: "Data")
        chunk_size: Maximum size of each chunk in characters (default: 800)
        chunk_overlap: Overlap between consecutive chunks in characters (default: 150)
        
    Returns: List of dictionaries, where each dictionary represents a chunk
    """

    # Initialize text splitter with optimized chunk size and overlap
    # Chunk size 750: Optimal for maintaining complete context while avoiding abrupt cuts
    # Overlap 150: Ensures continuity between chunks without excessive redundancy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = []
    data_path = Path(data_folder)
        
    # Process each JSON file in the data folder
    for json_file in data_path.glob("*.json"):
        try:
            # Read and parse JSON file with UTF-8 encoding
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract required fields from JSON data
            if 'text' in data:
                text = data['text']
                sources = data['sources']
                section = data['section']
                
                # Split text into chunks using LangChain text splitter
                text_chunks = text_splitter.split_text(text)
                
                # Create chunk objects with metadata for each text chunk
                for i, chunk in enumerate(text_chunks):
                    chunk_data = {
                        'text': chunk,
                        'sources': sources,
                        'chunk_id': f"{section}-{i}",
                        'section': section,	
                    }
                    chunks.append(chunk_data)
                
                print(f"Processed {json_file.name}: {len(text_chunks)} chunks")
            else:
                print(f"No 'text' field found in {json_file.name}")
                
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    print(f"Total chunks created: {len(chunks)}")
    return chunks
# ====================================================================================================== #



# ====================================================================================================== #
# Save Chunks to JSON
# ====================================================================================================== #
def save_chunks_to_json(chunks: List[Dict[str, Any]], output_file: str = "data/processed_chunks.json"):
    """
    This function takes a list of processed chunks and saves them to a JSON file
    with proper formatting and encoding. 
    
    Args:
        chunks: List of processed chunks to save
        output_file: Path and filename for the output JSON file (default: "data/processed_chunks.json")
    """
    try:
        file_path = Path(output_file)
        
        # Write chunks to JSON file with proper formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"Chunks saved to {output_file}")
    except Exception as e:
        print(f"Error saving chunks: {e}")
# ====================================================================================================== #


# ====================================================================================================== #
# Main Execution Block
# ====================================================================================================== #
if __name__ == "__main__":
    """
    Main execution block for standalone document processing.
    """
    # Process documents and save chunks
    chunks = process_json_documents()
    if chunks:
        save_chunks_to_json(chunks, output_file="data/processed_chunks.json")
# ====================================================================================================== #
