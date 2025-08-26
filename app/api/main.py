"""
FastAPI Main Application

This module contains the main FastAPI application for the PB RAG (Retrieval-Augmented Generation) system.
It provides health check endpoints and serves as the entry point for the API server.

Dependencies:
    - FastAPI framework
    - Application settings configuration
    - Environment variables properly configured
"""
# ====================================================================================================== #
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.core.settings import settings
from app.graph.agent_graph import create_agent_graph, AgentState
# ====================================================================================================== #



# ====================================================================================================== #
# Request/Response models
# ====================================================================================================== #
class QuestionRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float

# Initialize FastAPI application
app = FastAPI(
    title="PB RAG", 
    version="0.1.0",
    description="Retrieval-Augmented Generation API for document processing and AI interactions"
)

# Add CORS middleware for public API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for public API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ====================================================================================================== #



# ====================================================================================================== #
@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: QuestionRequest):
    """
    Main RAG endpoint that processes user questions using the LangGraph agent.
    
    Args: request: QuestionRequest containing the user's question
        
    Returns: RAGResponse with answer, sources, and confidence
    """
    try:
        # Validate question
        if not request.question or len(request.question.strip()) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Question must be at least 3 characters long"
            )
        
        # Create agent instance
        agent = create_agent_graph()
        
        # Prepare input for the agent
        agent_input: AgentState = {
            "question": request.question.strip(),
            "validated_question": None,
            "relevant_chunks": [],
            "generated_response": None,
            "confidence": None,
            "final_response": None,
            "status": "started"
        }
        
        # Execute the agent workflow
        result = agent.invoke(agent_input)
        
        # Extract the final response
        final_response = result.get("final_response", {})
        
        if not final_response:
            raise HTTPException(
                status_code=500, 
                detail="Agent failed to generate response"
            )
        
        # Return formatted response
        return RAGResponse(
            answer=final_response.get("answer", ""),
            sources=final_response.get("sources", []),
            confidence=final_response.get("confidence", 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )
# ====================================================================================================== #



# ====================================================================================================== #
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
            
    Raises: HTTPException: If health check fails
    """
    try:
        health_status = {
            "status": "ok",
            "env": settings.env,
            "openai_model": settings.openai_model,
            "pinecone_index": settings.pinecone_index,
            "timestamp": "2024-01-01T00:00:00Z"  # Add timestamp for monitoring
        }
        
        return health_status
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )
# ====================================================================================================== #



# ====================================================================================================== #
@app.get("/")
def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "PB RAG API - Retrieval-Augmented Generation System",
        "version": "0.1.0",
        "endpoints": {
            "health": "/healthz",
            "ask": "/ask",
            "docs": "/docs"
        },
        "status": "operational"
    }
# ====================================================================================================== #



# ====================================================================================================== #
@app.on_event("startup")
def startup_event():
    """
    Application startup event handler.
    
    Performs initialization tasks when the FastAPI application starts.
    """
    print(f"[OK] PB RAG API started: env={settings.env}, model={settings.openai_model}")
# ====================================================================================================== #



# ====================================================================================================== #
@app.on_event("shutdown")
def shutdown_event():
    """
    Application shutdown event handler.
    
    Performs cleanup tasks when the FastAPI application shuts down.
    """
    print("[INFO] PB RAG API shutting down...")
# ====================================================================================================== #
