"""
FastAPI Main Application

This module contains the main FastAPI application for the PB RAG (Retrieval-Augmented Generation) system.
It provides health check endpoints and serves as the entry point for the API server.

Dependencies:
    - FastAPI framework
    - Application settings configuration
    - Environment variables properly configured
"""

from fastapi import FastAPI
from app.core.settings import settings

#*****************************************************************************************************#
# Initialize FastAPI application
app = FastAPI(
    title="PB RAG", 
    version="0.1.0",
    description="Retrieval-Augmented Generation API for document processing and AI interactions"
)
#*****************************************************************************************************#



#*****************************************************************************************************#
@app.get("/healthz")
def health_check():
    """
    Health check endpoint to verify system status and configuration.
    
    Returns:
        dict: Health status information including:
            - status: System status (always "ok" if endpoint is reachable)
            - env: Current environment (development, production, etc.)
            - openai_model: Configured OpenAI model for embeddings
            - pinecone_index: Configured Pinecone index name
            
    Raises:
        HTTPException: If health check fails
    """
    health_status = {
        "status": "ok",
        "env": settings.env,
        "openai_model": settings.openai_model,
        "pinecone_index": settings.pinecone_index,
    }
    
    return health_status
#*****************************************************************************************************#



#*****************************************************************************************************#
@app.on_event("startup")
def startup_event():
    """
    Application startup event handler.
    
    Performs initialization tasks when the FastAPI application starts.
    """
    print(f"[OK] PB RAG API started: env={settings.env}, model={settings.openai_model}")
#*****************************************************************************************************#



#*****************************************************************************************************#
@app.on_event("shutdown")
def shutdown_event():
    """
    Application shutdown event handler.
    
    Performs cleanup tasks when the FastAPI application shuts down.
    """
    print("[INFO] PB RAG API shutting down...")
#*****************************************************************************************************#
