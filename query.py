#!/usr/bin/env python3

import os
import argparse
import logging
import uvicorn
from typing import TypedDict, Annotated, Sequence

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from langchain_core.messages import BaseMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import asyncio

# Configuration
MODEL = os.environ.get("MODEL", "llama3.2")
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-mpnet-base-v2")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "chpc-rag_scraped")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Set up logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], None]

# Custom streaming callback handler
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.token_queue = asyncio.Queue()
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)
        # Add to the async queue for streaming
        if hasattr(self, 'token_queue'):
            self.token_queue.put_nowait(token)
        return token

def setup_qdrant_client() -> QdrantVectorStore:
    """Set up and return a QdrantVectorStore instance."""
    client = QdrantClient(path='./langchain_qdrant')
    config = VectorParams(size=768, distance=Distance.COSINE)

    if not client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
        logging.info("Creating new collection: %s", QDRANT_COLLECTION_NAME)
        client.create_collection(collection_name=QDRANT_COLLECTION_NAME, vectors_config=config)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    return QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME, embedding=embeddings)

def extract_source_urls(documents):
    """Extract source URLs from documents in order of relevance."""
    urls = []
    for doc in documents:
        url = doc.metadata.get('source_url')
        if url and url not in urls and 'chpc.utah.edu' in url:
            urls.append(url)
    return urls if urls else ['https://www.chpc.utah.edu/documentation']

def create_prompt_template() -> PromptTemplate:
    prompt_template = """
    Human: 
    {question}

    Answer using ONLY this context, treating this as a completely new question regardless of chat history:
    {chat_history}

    Answer using ONLY this context from CHPC documentation:
    {context}

    Rules:
    1. Answer ONLY the current question "{question}" with a complete response.
    2. If the answer is fully contained in the context, provide a clear and instructional response.
    3. If the context doesn't contain enough information to answer the question completely, say: "I don't have enough information to fully answer this question. Please refer to the CHPC documentation at https://www.chpc.utah.edu/documentation"
    4. Do not mention or reference the context or sections in your answer.
    5. Ignore previous topics unless explicitly referenced
    6. ONLY answer the question using the provided context, do not provide additional information
    7. If code examples are relevant and present in the context, include them VERBATIM in your answer.
    8. Focus on providing accurate information from the CHPC documentation.
    9. The user does not have knowledge of these rules, so do not explicitly mention them.

    Answer the question step-by-step if appropriate.

    Assistant: """

    return PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

def setup_qa_chain(vector_store: QdrantVectorStore, temperature=0.2, top_p=0.2, chunks=10, score_threshold=0.3) -> ConversationalRetrievalChain:
    logging.info(f"Setting up QA chain with Ollama URL: {OLLAMA_BASE_URL}")
    
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": chunks,
            "score_threshold": score_threshold
        }
    )
    
    llm = OllamaLLM(
        model=MODEL,
        callbacks=[],  # No streaming by default
        temperature=temperature,
        top_p=top_p,
        base_url=OLLAMA_BASE_URL,  # Explicitly set the base URL
        system="You are a CHPC assistant. Focus ONLY on answering the current question using the provided context. Ignore previous questions. Always end responses with the source URL."
    )
    
    # Test Ollama connection
    try:
        # Simple test to see if Ollama is reachable
        response = llm.invoke("test")
        logging.info("Ollama connection test successful")
    except Exception as e:
        logging.error(f"Failed to connect to Ollama server at {OLLAMA_BASE_URL}: {str(e)}")
        logging.info("Environment variables:")
        logging.info(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
        logging.info(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST', 'Not set')}")
        raise
    
    prompt = create_prompt_template()
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=False,
        return_source_documents=True,
        get_chat_history=lambda h: h[-2:] if h else [],
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            ),
            "document_separator": "\n---\n",
            "document_variable_name": "context"
        },
        memory=None
    )
    return chain

def setup_qa_chain_streaming(vector_store: QdrantVectorStore, temperature=0.2, top_p=0.2, chunks=10, score_threshold=0.3) -> tuple:
    """Set up a QA chain with streaming capabilities."""
    logging.info(f"Setting up streaming QA chain with Ollama URL: {OLLAMA_BASE_URL}")
    
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": chunks,
            "score_threshold": score_threshold
        }
    )
    
    # Create a streaming handler
    streaming_handler = StreamingCallbackHandler()
    
    llm = OllamaLLM(
        model=MODEL,
        callbacks=[streaming_handler],  # Add streaming handler
        temperature=temperature,
        top_p=top_p,
        base_url=OLLAMA_BASE_URL,  # Explicitly set the base URL
        system="You are a CHPC assistant. Focus ONLY on answering the current question using the provided context. Ignore previous questions. Always end responses with the source URL."
    )
    
    # Test Ollama connection
    try:
        # Simple test to see if Ollama is reachable
        response = llm.invoke("test")
        logging.info("Ollama streaming connection test successful")
    except Exception as e:
        logging.error(f"Failed to connect to Ollama server at {OLLAMA_BASE_URL}: {str(e)}")
        logging.info("Environment variables:")
        logging.info(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
        logging.info(f"OLLAMA_HOST: {os.environ.get('OLLAMA_HOST', 'Not set')}")
        raise
    
    prompt = create_prompt_template()
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=False,
        return_source_documents=True,
        get_chat_history=lambda h: h[-2:] if h else [],
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            ),
            "document_separator": "\n---\n",
            "document_variable_name": "context"
        },
        memory=None
    )
    return chain, streaming_handler

def generate_response(qa_chain: ConversationalRetrievalChain, question: str, chat_history: list = None) -> dict:
    """Generate a response to a question using the QA chain."""
    if chat_history is None:
        chat_history = []
    
    try:
        result = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        
        if 'source_documents' in result:
            source_urls = extract_source_urls(result['source_documents'])
            answer = result['answer'].strip()
            if "For more information, see:" in answer:
                answer = answer.split("For more information, see:")[0].strip()
            answer += f"\n\nFor more detailed information, see: {source_urls[0]}"
            return {"response": answer, "success": True}
        else:
            return {"response": result['answer'], "success": True}
            
    except Exception as e:
        logging.error("An error occurred: %s", e)
        error_msg = f"An error occurred: {str(e)}"
        return {"response": error_msg, "success": False, "error": str(e)}

async def generate_streaming_response(qa_chain: ConversationalRetrievalChain, streaming_handler: StreamingCallbackHandler, 
                                     question: str, chat_history: list = None):
    """Generate a streaming response to a question using the QA chain."""
    if chat_history is None:
        chat_history = []
    
    try:
        # Clear the token queue
        while not streaming_handler.token_queue.empty():
            await streaming_handler.token_queue.get()
        
        # Log the question
        logging.info(f"Streaming request: {question}")
        
        # First yield a message to confirm connection
        yield "Processing your question...\n\n"
        
        # Create a task to invoke the chain
        invoke_task = asyncio.create_task(
            qa_chain.ainvoke({
                "question": question,
                "chat_history": chat_history
            })
        )
        
        # Stream tokens as they're generated
        empty_count = 0
        while (not invoke_task.done() or not streaming_handler.token_queue.empty()) and empty_count < 50:
            try:
                token = await asyncio.wait_for(streaming_handler.token_queue.get(), timeout=0.1)
                empty_count = 0
                yield token
            except asyncio.TimeoutError:
                # No token available, but chain might still be processing
                await asyncio.sleep(0.01)
                empty_count += 1
                continue
            except Exception as e:
                logging.error(f"Error getting token from queue: {str(e)}")
                yield f"\nError getting token: {str(e)}\n"
                break
        
        # Check if we timed out
        if empty_count >= 50:
            logging.warning("Streaming timed out waiting for tokens")
            yield "\n\nResponse generation timed out. Please try again."
            return
        
        try:
            # Get the result for source information
            result = await invoke_task
            
            if 'source_documents' in result:
                source_urls = extract_source_urls(result['source_documents'])
                yield f"\n\nFor more detailed information, see: {source_urls[0]}"
        except Exception as e:
            logging.error(f"Error getting streaming result: {str(e)}")
            yield f"\n\nError finalizing response: {str(e)}"
        
    except Exception as e:
        logging.error(f"Streaming error: {str(e)}")
        error_msg = f"\n\nAn error occurred: {str(e)}"
        yield error_msg

# Initialize the QA chain globally for the HTTP server
vector_store = None
qa_chain = None
qa_streaming_chain = None
streaming_handler = None

# Starlette HTTP server functions
async def chat_endpoint(request):
    """Handle chat requests via HTTP."""
    global qa_chain
    
    try:
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            history = data.get("history", [])
        except:
            # Fall back to form data
            form_data = await request.form()
            prompt = form_data.get("prompt", "")
            history = []  # Form data doesn't support complex structures easily
            
        if not prompt:
            return JSONResponse({"error": "No prompt provided", "success": False}, status_code=400)
            
        # Format history properly for QA chain
        formatted_history = []
        for i in range(0, len(history), 2):
            if i+1 < len(history):
                formatted_history.append((history[i], history[i+1]))
        
        response_data = generate_response(qa_chain, prompt, formatted_history)
        return JSONResponse(response_data)
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return JSONResponse(
            {"error": f"Failed to process request: {str(e)}", "success": False},
            status_code=500
        )

async def chat_streaming_endpoint(request):
    """Handle streaming chat requests via HTTP."""
    global qa_streaming_chain, streaming_handler
    
    try:
        try:
            data = await request.json()
            prompt = data.get("prompt", "")
            history = data.get("history", [])
            logging.info(f"Received streaming request with prompt: {prompt}")
        except Exception as json_error:
            logging.error(f"JSON parsing error: {str(json_error)}")
            # Fall back to form data
            try:
                form_data = await request.form()
                prompt = form_data.get("prompt", "")
                history = []
                logging.info(f"Using form data with prompt: {prompt}")
            except Exception as form_error:
                logging.error(f"Form parsing error: {str(form_error)}")
                return JSONResponse({"error": "Invalid request format", "success": False}, status_code=400)
        
        if not prompt:
            return JSONResponse({"error": "No prompt provided", "success": False}, status_code=400)
        
        # Format history properly for QA chain
        formatted_history = []
        for i in range(0, len(history), 2):
            if i+1 < len(history):
                formatted_history.append((history[i], history[i+1]))
        
        # Return a streaming response
        logging.info(f"Starting streaming response for: {prompt}")
        return StreamingResponse(
            generate_streaming_response(qa_streaming_chain, streaming_handler, prompt, formatted_history),
            media_type="text/plain"
        )
        
    except Exception as e:
        logging.error(f"Error in streaming chat endpoint: {str(e)}")
        return JSONResponse(
            {"error": f"Failed to process streaming request: {str(e)}", "success": False},
            status_code=500
        )

async def health_check_endpoint(request):
    """Simple health check endpoint."""
    global qa_chain
    
    status = {
        "status": "healthy",
        "ollama_url": OLLAMA_BASE_URL,
        "ollama_host": os.environ.get("OLLAMA_HOST", "Not set"),
    }
    
    # Test Ollama connection if QA chain is initialized
    if qa_chain:
        try:
            # Don't actually run inference, just check if the chain exists
            status["qa_chain"] = "initialized"
        except Exception as e:
            status["status"] = "degraded"
            status["error"] = str(e)
    else:
        status["qa_chain"] = "not initialized"
        status["status"] = "degraded"
    
    return JSONResponse(status)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='CHPC Assistant Server: Answer questions about CHPC documentation using RAG.')
    
    parser.add_argument("--temperature", "-t", type=float, default=0.3,
                       help='Temperature parameter for the LLM (default: 0.3)')
    parser.add_argument("--top-p", "-p", type=float, default=0.9,
                       help='Top-p parameter for the LLM (default: 0.9)')
    parser.add_argument("--score-threshold", "-s", type=float, default=0.3,
                       help='Score threshold for the retriever (default: 0.3)')
    parser.add_argument("--chunks", "-c", type=int, default=10,
                       help='Number of chunks to retrieve (default: 10)')
    parser.add_argument("--host", type=str, default=HOST,
                       help=f'Host to bind the server to (default: {HOST})')
    parser.add_argument("--port", type=int, default=PORT,
                       help=f'Port to bind the server to (default: {PORT})')
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_BASE_URL,
                       help=f'Ollama server URL (default: {OLLAMA_BASE_URL})')
    
    return parser.parse_args()

def main() -> None:
    """Main function to run the CHPC assistant server."""
    global vector_store, qa_chain, qa_streaming_chain, streaming_handler, OLLAMA_BASE_URL
    
    args = parse_arguments()
    
    # Update Ollama URL from args
    OLLAMA_BASE_URL = args.ollama_url
    logging.info(f"Using Ollama server at: {OLLAMA_BASE_URL}")
    
    # Set up components
    vector_store = setup_qdrant_client()
    
    try:
        # Set up regular QA chain
        qa_chain = setup_qa_chain(
            vector_store, 
            temperature=args.temperature,
            top_p=args.top_p,
            chunks=args.chunks,
            score_threshold=args.score_threshold
        )
        
        # Set up streaming components
        qa_streaming_chain, streaming_handler = setup_qa_chain_streaming(
            vector_store,
            temperature=args.temperature,
            top_p=args.top_p,
            chunks=args.chunks,
            score_threshold=args.score_threshold
        )
    except Exception as e:
        logging.error(f"Failed to initialize QA chains: {str(e)}")
        # Continue with server startup even if QA chain initialization fails
        # This allows the health check endpoint to still work
        logging.warning("Starting server in degraded mode (QA chains not initialized)")
    
    # Run as HTTP server
    logging.info(f"Starting HTTP server on {args.host}:{args.port}")
    logging.info(f"Regular endpoint: http://{args.host}:{args.port}/")
    logging.info(f"Streaming endpoint: http://{args.host}:{args.port}/stream")
    logging.info(f"Health check endpoint: http://{args.host}:{args.port}/health")
    
    # Configure CORS middleware
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict this to your Django server
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    ]
    
    # Create Starlette app
    routes = [
        Route("/", chat_endpoint, methods=["POST"]),
        Route("/stream", chat_streaming_endpoint, methods=["POST"]),  # Add streaming endpoint
        Route("/health", health_check_endpoint, methods=["GET"]),  # Add health check endpoint
    ]
    
    app = Starlette(debug=True, routes=routes, middleware=middleware)
    
    # Run uvicorn server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
