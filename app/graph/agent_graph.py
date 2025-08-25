"""
LangGraph Agent for RAG System

This module implements a simple RAG agent using LangGraph with 4 nodes:
1. Input Node - Receives and validates user questions
2. Retrieval Node - Searches relevant information in Pinecone
3. Generation Node - Generates responses using LLM
4. Output Node - Formats final response
"""

from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from app.core.settings import settings
from openai import OpenAI
from pinecone import Pinecone
from typing import List

# ====================================================================================================== #
# STATE DEFINITION - Using TypedDict for type safety
# ====================================================================================================== #
class AgentState(TypedDict):
    question: str
    validated_question: Optional[str]
    relevant_chunks: List[Dict[str, Any]]
    generated_response: Optional[str]
    confidence: Optional[float]
    final_response: Optional[Dict[str, Any]]
    status: str
# ====================================================================================================== #



# ====================================================================================================== #
# Confidence calibration
# ====================================================================================================== #
def compute_confidence_from_scores(
    similarity_scores: List[float],
    expected_min_similarity: float = 0.20,
    expected_max_similarity: float = 0.70,
    score_threshold: float = 0.20
) -> float:
    """
    Convert retrieval cosine similarities into a user-friendly confidence in [0, 1].

    Approach:
    - Keep only the Top-3 scores (strongest evidence).
    - Weighted average (0.6, 0.3, 0.1) to emphasize Top-1.
    - Linearly rescale that value into [0, 1] using two anchors:
        expected_min_similarity -> 0.0  (low confidence)
        expected_max_similarity -> 1.0  (high confidence)
    """
    if not similarity_scores:
        return 0.0

    # Filter out very low scores
    filtered_scores = [s for s in similarity_scores if s >= score_threshold]
    if not filtered_scores:
        return 0.0
    
    # Use top-3 or less if there are few results
    top_scores = sorted(filtered_scores, reverse=True)[:3]
    
    # Weights that sum 1.0
    weights = [0.6, 0.3, 0.1][:len(top_scores)]
    weighted_avg = sum(w * s for w, s in zip(weights, top_scores)) / sum(weights)
    
    # More robust normalization
    confidence = (weighted_avg - expected_min_similarity) / (expected_max_similarity - expected_min_similarity)
    return max(0.0, min(1.0, confidence))
# ====================================================================================================== #



# ====================================================================================================== #
# NODE 1: INPUT NODE - Receives and validates user questions
# ====================================================================================================== #
def input_node(state: AgentState) -> AgentState:
    
    # Extract question from state
    question = state.get("question", "")
    
    # Simple validation
    if not question or len(question.strip()) < 3:
        raise ValueError("Question must be at least 3 characters long")
    
    # Clean and store question
    cleaned_question = question.strip()
    
    print(f"✅ Input Node: Question validated: '{cleaned_question}'")
    
    # Return updated state
    return {
        **state,
        "validated_question": cleaned_question,
        "status": "input_validated"
    }
# ====================================================================================================== #



# ====================================================================================================== #
# NODE 2: RETRIEVAL NODE - Searches relevant information in Pinecone
# ====================================================================================================== #
def retrieval_node(state: AgentState) -> AgentState:
    
    question = state.get("validated_question", "")
    
    print(f"🔍 Retrieval Node: Searching for: '{question}'")
    
    try:
        # Step 1: Generate embedding for the question
        print("   📡 Generating embedding for question...")
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        embedding_response = openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=question
        )
        
        question_embedding = embedding_response.data[0].embedding
        print(f"   ✅ Embedding generated (dimension: {len(question_embedding)})")
        
        # Step 2: Search in Pinecone with rerank
        print("   🔍 Searching in Pinecone with rerank...")
        pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
        index = pinecone_client.Index(settings.pinecone_index)
        
        # Step 2a: Initial search to get more candidates
        search_results = index.query(
            vector=question_embedding,
            top_k=10,  # Get more candidates for rerank
            include_metadata=True,
            include_values=False,
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 10,
                "rank_fields": ["text"]
            }
        )
        
        # Step 2: Process reranked results (more relevant)
        relevant_chunks = []
        for match in search_results.matches:
            chunk_data = {
                "text": match.metadata.get("text", ""),
                "source": match.metadata.get("sources", "unknown"),
                "section": match.metadata.get("section", "unknown"),
                "score": match.score,  # This score is already reranked
                "chunk_id": match.id
            }
            relevant_chunks.append(chunk_data)
        
        print(f"   ✅ Found {len(relevant_chunks)} relevant chunks")
        
        # Log all results for debugging
        if relevant_chunks:
            print(f"   📊 All retrieved chunks:")
            for i, chunk in enumerate(relevant_chunks[:5]):  # Show top 5
                print(f"      {i+1}. Score: {chunk['score']:.3f} | Section: {chunk['section']} | Text: {chunk['text'][:80]}...")
            
            top_chunk = relevant_chunks[0]
            print(f"   🏆 Top result: {top_chunk['text'][:100]}... (score: {top_chunk['score']:.3f})")
        
        # Return updated state
        return {
            **state,
            "relevant_chunks": relevant_chunks,
            "status": "retrieval_completed"
        }
        
    except Exception as e:
        print(f"   ❌ Error in retrieval: {str(e)}")
        # Return empty chunks on error, but continue the pipeline
        return {
            **state,
            "relevant_chunks": [],
            "status": "retrieval_failed",
            "error": str(e)
        }
# ====================================================================================================== #



# ====================================================================================================== #
# NODE 3: GENERATION NODE - Generates responses using LLM
# ====================================================================================================== #
def generation_node(state: AgentState) -> AgentState:

    question = state.get("validated_question", "")
    chunks = state.get("relevant_chunks", [])
    
    try:
        # Check if we have chunks to work with
        if not chunks:
            print("   ⚠️ No relevant chunks found, generating generic response")
            generic_response = f"No encontré información específica sobre '{question}' en mi base de conocimiento."
            return {
                **state,
                "generated_response": generic_response,
                "status": "generation_completed",
                "confidence": 0.0
            }
        
        # ---------------------- Build context from chunks ----------------------
        print("   📚 Building context from relevant chunks...")
        # Sort by score (highest first) so Top-1/Top-3 are truly the best evidence
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)

        # Prepare human-readable context for the LLM
        context_sections: list[str] = []
        for idx, chunk in enumerate(sorted_chunks, start=1):
            chunk_text: str = chunk.get("text", "")
            chunk_score: float = float(chunk.get("score", 0.0))
            context_sections.append(f"Source {idx} (relevance: {chunk_score:.3f}):\n{chunk_text}")

        # ---------------------- Calibrated confidence -------------------------
        similarity_scores: list[float] = [float(c.get("score", 0.0)) for c in sorted_chunks]
        calibrated_confidence: float = compute_confidence_from_scores(
            similarity_scores=similarity_scores,
            expected_min_similarity=0.20,  # tweak after measuring on your eval set
            expected_max_similarity=0.70
        )
        print(f"   📊 Scores (Top-10): {[round(s, 3) for s in similarity_scores[:10]]}")
        print(f"   🎯 Calibrated confidence: {calibrated_confidence:.3f}")
        
        # ---------------------- Create intelligent prompt ----------------------
        system_prompt = """Eres un asistente experto en Punta Blanca Solutions. 
        Tu tarea es responder preguntas basándote en la información proporcionada.
        
        INSTRUCCIONES CRÍTICAS:
        1. Responde SOLO basándote en la información proporcionada
        2. Si la información es insuficiente, di claramente "No tengo información suficiente sobre [aspecto específico]
        3. Si hay información en múltiples fuentes, combínala para dar una respuesta completa
        4. Cita las fuentes más relevantes (con mayor score)
        5. Responde en español profesional
        6. Sé preciso y específico
        
        Contexto disponible:"""
        
        user_prompt = f"""
        Pregunta del usuario: {question}

        Información relevante (ordenada por relevancia):
        {'\n\n'.join(context_sections)}
        """
        
        # ---------------------- Generate response with OpenAI ----------------------
        print("   Calling OpenAI API...")
        openai_client = OpenAI(api_key=settings.openai_api_key)
        
        response = openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Low temperature for consistent, factual responses
            max_tokens=500
        )
        
        generated_response = response.choices[0].message.content.strip()
        
        print(f"   ✅ Response generated successfully")
        print(f"   📊 Average confidence: {calibrated_confidence:.3f}")
        
        # ---------------------- Return updated state ----------------------
        return {
            **state,
            "generated_response": generated_response,
            "confidence": calibrated_confidence,
            "status": "generation_completed"
        }
        
    except Exception as e:
        print(f"   ❌ Error in generation: {str(e)}")
        # Return fallback response on error
        fallback_response = f"Lo siento, tuve un problema generando la respuesta para: '{question}'. Por favor, intenta de nuevo."
        
        return {
            **state,
            "generated_response": fallback_response,
            "confidence": 0.0,
            "status": "generation_failed",
            "error": str(e)
        }
# ====================================================================================================== #



# ====================================================================================================== #
# NODE 4: OUTPUT NODE - Formats final response
# ====================================================================================================== #
def output_node(state: AgentState) -> AgentState:

    question = state.get("validated_question", "")
    response = state.get("generated_response", "")
    chunks = state.get("relevant_chunks", [])
    
    print(f"📤 Output Node: Formatting response")
    
    # Format response according to API specification
    confidence = state.get("confidence", 0.0)
    
    # Debug: Print what we're getting
    print(f"   📊 Debug - Confidence from state: {confidence}")
    print(f"   📊 Debug - Raw sources from chunks: {[chunk.get('source', 'unknown') for chunk in chunks]}")
    
    # Eliminate duplicates in sources using set
    unique_sources = list(set([chunk.get("source", "unknown") for chunk in chunks]))
    print(f"   📊 Debug - Unique sources after deduplication: {unique_sources}")
    
    formatted_response = {
        "answer": response,
        "sources": unique_sources,
        "confidence": confidence
    }
    
    print(f"   📤 Final formatted response: {formatted_response}")
    print(f"✅ Output Node: Response formatted successfully")
    
    # Return final state
    return {
        **state,
        "final_response": formatted_response,
        "status": "completed"
    }
# ====================================================================================================== #



# ====================================================================================================== #
# GRAPH DEFINITION - Connect all nodes
# ====================================================================================================== #
def create_agent_graph():
    
    # Create workflow with typed state
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("input", input_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("output", output_node)
    
    # Define the flow: input → retrieval → generation → output
    workflow.set_entry_point("input")
    workflow.add_edge("input", "retrieval")
    workflow.add_edge("retrieval", "generation")
    workflow.add_edge("generation", "output")
    workflow.add_edge("output", END)
    
    # Compile the graph
    compiled_workflow = workflow.compile()
    
    print("✅ LangGraph agent created successfully!")
    return compiled_workflow
# ====================================================================================================== #



# ====================================================================================================== #
# MAIN FUNCTION - For testing
# ====================================================================================================== #
def main():
    print("🧪 Testing LangGraph Agent...")
    
    # Create the agent
    agent = create_agent_graph()
    
    # Test with sample input (must match AgentState structure)
    test_input: AgentState = {
        "question": "¿Quiénes son los fundadores de Punta Blanca y cuáles son sus roles?",
        "validated_question": None,
        "relevant_chunks": [],
        "generated_response": None,
        "confidence": None,
        "final_response": None,
        "status": "started"
    }
    
    print(f"\n📝 Test Input: {test_input['question']}")
    print("=" * 50)
    
    # Run the agent
    result = agent.invoke(test_input)
    
    print("\n" + "=" * 50)
    print("🎉 AGENT EXECUTION COMPLETED!")
    print("=" * 50)
    print(f"📊 Final Status: {result.get('status', 'unknown')}")
    print(f"💬 Response: {result.get('final_response', {}).get('answer', 'No response')}")
    print(f"🔗 Sources: {result.get('final_response', {}).get('sources', [])}")
    print(f"🎯 Confidence: {result.get('final_response', {}).get('confidence', 0)}")
# ====================================================================================================== #



# ====================================================================================================== #
if __name__ == "__main__":
    main()
# ====================================================================================================== #