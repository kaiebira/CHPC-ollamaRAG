#!/usr/bin/env python3
"""
CHPC Assistant Server using RAG with Ollama and Qdrant.

This script sets up and runs a Starlette web server that provides API endpoints
for interacting with a Retrieval-Augmented Generation (RAG) system.
It uses LangChain to orchestrate the process, Ollama for the LLM,
Qdrant for the vector store, and HuggingFace sentence transformers for embeddings.
Includes question condensing to handle conversational context better.
"""

# --- Standard Library Imports ---
import os
import argparse
import logging
import asyncio
import json
from typing import TypedDict, Sequence, List, Optional, Tuple, AsyncGenerator

# --- Third-Party Imports ---
import uvicorn
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Use Async client for setup checks, but from_existing_collection handles client internally
from qdrant_client import AsyncQdrantClient, QdrantClient # Keep sync for initial check if needed
from qdrant_client.http.models import Distance, VectorParams
import qdrant_client.http.exceptions as qdrant_exceptions

# --- Configuration using Pydantic Settings ---

class Settings(BaseSettings):
    """
    Manages application settings using environment variables, .env files,
    and command-line arguments.
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    model: str = Field(default="gemma3:4b", description="Name of the Ollama model to use.")
    embeddings_model_name: str = Field(default="all-mpnet-base-v2", description="Name of the sentence transformer model for embeddings.")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Base URL for the Ollama server.")
    # System prompt can be simpler if question condensing handles history context
    system_prompt: str = Field(
        default=(
            "You are a helpful CHPC assistant. Answer the user's question based *only* on the provided context documents. "
            "If the context doesn't provide an answer, state that clearly and suggest referring to the main CHPC documentation."
        ),
        description="System prompt for the LLM (used in the final answer generation)."
    )
    qdrant_path: str = Field(default="./langchain_qdrant", description="Path to the local Qdrant database directory.")
    qdrant_collection_name: str = Field(default="chpc-rag", description="Name of the Qdrant collection.")
    qdrant_vector_size: int = Field(default=768, description="Dimension size of the vectors (depends on embedding model).")
    qdrant_distance: Distance = Field(default=Distance.COSINE, description="Distance metric for Qdrant (e.g., Cosine, Dot, Euclid).")
    retriever_search_type: str = Field(default="mmr", description="Type of search for the retriever ('similarity' or 'mmr'). MMR helps diversity.")
    retriever_k: int = Field(default=5, description="Number of documents to retrieve and pass to the LLM.")
    retriever_fetch_k: int = Field(default=10, description="Number of documents to fetch initially for MMR calculation.")
    retriever_score_threshold: float = Field(default=0.3, description="Minimum relevance score for retrieved documents (0-1).")
    temperature: float = Field(default=0.2, description="LLM temperature for sampling (0-1). Lower is more deterministic.")
    top_p: float = Field(default=0.8, description="LLM top-p nucleus sampling (0-1).")
    # Add Ollama context window size setting - Updated default to 128k
    num_ctx: int = Field(default=131072, description="Context window size for the Ollama LLM.")
    host: str = Field(default="0.0.0.0", description="Host address to bind the server to.")
    port: int = Field(default=8000, description="Port number to bind the server to.")
    log_level: str = Field(default="INFO", description="Logging level (e.g., DEBUG, INFO, WARNING).")

# --- Logging Setup ---
def setup_logging(log_level: str):
    """
    Configures application-wide logging.
    """
    log_level_upper = log_level.upper()
    logging.basicConfig(
        format="{asctime} - {levelname:<8} - {name}:{funcName}:{lineno} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level_upper, logging.INFO)
    )
    logging.getLogger().setLevel(getattr(logging, log_level_upper, logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# --- Application State ---
class AppState(TypedDict):
    """Structure definition for the global application state."""
    settings: Settings
    vector_store: QdrantVectorStore
    qa_chain: ConversationalRetrievalChain
    qa_streaming_chain: ConversationalRetrievalChain
    streaming_handler: "StreamingCallbackHandler"

app_state: Optional[AppState] = None

# --- Streaming Callback Handler ---
class StreamingCallbackHandler(BaseCallbackHandler):
    """
    LangChain Callback handler for streaming LLM responses asynchronously.
    """
    def __init__(self):
        self.token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.tokens: List[str] = []

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
        await self.token_queue.put(token)

    async def on_llm_end(self, response, **kwargs) -> None:
        await self.token_queue.put(None)

    async def on_llm_error(self, error: BaseException, **kwargs) -> None:
        logger.error(f"LLM error during streaming: {error}", exc_info=True)
        await self.token_queue.put(f"\nLLM Error: {str(error)}\n")
        await self.token_queue.put(None)

# --- Helper Functions ---

async def setup_qdrant_vector_store(settings: Settings) -> QdrantVectorStore:
    """
    Sets up the QdrantVectorStore instance.
    """
    logger.info(f"Checking Qdrant collection status at path: {settings.qdrant_path}")
    try:
        # Use sync client for initial check/create
        sync_client = QdrantClient(path=settings.qdrant_path)
        collections_response = sync_client.get_collections()
        collection_names = [col.name for col in collections_response.collections]

        if settings.qdrant_collection_name not in collection_names:
            logger.warning(f"Collection '{settings.qdrant_collection_name}' not found. Creating...")
            sync_client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(size=settings.qdrant_vector_size, distance=settings.qdrant_distance)
            )
            logger.info(f"Collection '{settings.qdrant_collection_name}' created successfully.")
        else:
             logger.info(f"Using existing Qdrant collection: {settings.qdrant_collection_name}")
        sync_client.close()

    except qdrant_exceptions.UnexpectedResponse as e:
        logger.error(f"Failed initial check/create with Qdrant: {e}")
        raise RuntimeError(f"Could not connect to or setup vector database: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Qdrant setup check: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected Qdrant setup error: {e}")

    logger.info(f"Loading embedding model: {settings.embeddings_model_name}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model_name)
    except Exception as e:
        logger.error(f"Failed to load embedding model '{settings.embeddings_model_name}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to load embedding model: {e}")

    logger.info(f"Initializing QdrantVectorStore for collection '{settings.qdrant_collection_name}' using from_existing_collection...")
    try:
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            path=settings.qdrant_path,
            collection_name=settings.qdrant_collection_name,
        )
        logger.info("QdrantVectorStore initialized successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize QdrantVectorStore with from_existing_collection: {e}", exc_info=True)
        if isinstance(e, TypeError) and 'embedding' in str(e):
             logger.error("Potential issue: The 'embedding' argument might not be compatible in this context/version.")
        raise RuntimeError(f"Failed to initialize QdrantVectorStore: {e}")


def extract_source_urls(documents: List[Document], max_urls: int = 3) -> List[str]:
    """
    Extracts unique, relevant source URLs from retrieved documents.
    """
    urls: List[str] = []
    seen_urls: set[str] = set()
    for doc in documents:
        url = doc.metadata.get('source_url')
        if url and url not in seen_urls and 'chpc.utah.edu' in url:
            urls.append(url)
            seen_urls.add(url)
            if len(urls) >= max_urls:
                break
    if not urls:
        return ['https://www.chpc.utah.edu/documentation']
    return urls

def create_prompt_template() -> PromptTemplate:
    """
    Creates the main prompt template for the final answer generation step.
    This prompt uses the (potentially condensed) question and the retrieved context.
    """
    prompt_template_str = """
Human: Answer the following question based *only* on the provided context:
{question}

Context:
{context}

Rules:
1. Base your answer *solely* on the provided context documents.
2. If the context fully answers the question, provide a clear and helpful response.
3. If the context does not contain enough information to answer the question, state that clearly, for example: "Based on the provided documentation excerpts, I cannot answer that question. You may find more information at https://www.chpc.utah.edu/documentation/."
4. Do NOT mention the context itself or these rules in your answer.
5. If code examples are relevant and present in the context, include them VERBATIM, often using Markdown code blocks.

Assistant:"""

    return PromptTemplate(
        template=prompt_template_str,
        input_variables=["context", "question"] # History is handled by the condensing step
    )

# --- Question Condensing Components ---
CONDENSE_QUESTION_PROMPT_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question that captures the necessary context from the chat history.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)

def format_chat_history_for_condensing(chat_history: List[Tuple[str, str]]) -> str:
    """
    Formats chat history (list of tuples) into a string for the condense prompt.
    """
    buffer = ""
    for human_msg, ai_msg in chat_history:
        buffer += f"Human: {human_msg}\nAssistant: {ai_msg}\n"
    return buffer.strip()
# --- End Question Condensing Components ---


async def setup_qa_chain(
    settings: Settings,
    vector_store: QdrantVectorStore,
    streaming: bool = False
) -> Tuple[ConversationalRetrievalChain, Optional[StreamingCallbackHandler]]:
    """
    Sets up the LangChain ConversationalRetrievalChain (asynchronously).
    Includes configuration for question condensing using chat history.
    """
    operation_type = "streaming" if streaming else "non-streaming"
    logger.info(f"Setting up {operation_type} QA chain with Ollama URL: {settings.ollama_base_url}")
    logger.info(f"Using context window size (num_ctx): {settings.num_ctx}") # Log context size

    retriever_search_kwargs = {
        "k": settings.retriever_k,
        "score_threshold": settings.retriever_score_threshold
    }
    if settings.retriever_search_type == "mmr":
        retriever_search_kwargs["fetch_k"] = settings.retriever_fetch_k
        logger.info(f"Using MMR retriever: k={settings.retriever_k}, fetch_k={settings.retriever_fetch_k}, score_threshold={settings.retriever_score_threshold}")
    else:
         logger.info(f"Using Similarity retriever: k={settings.retriever_k}, score_threshold={settings.retriever_score_threshold}")

    retriever = vector_store.as_retriever(
        search_type=settings.retriever_search_type,
        search_kwargs=retriever_search_kwargs
    )

    callbacks = []
    streaming_handler = None
    if streaming:
        streaming_handler = StreamingCallbackHandler()
        callbacks = [streaming_handler]

    try:
        # LLM for the final answer generation step
        llm = OllamaLLM(
            model=settings.model,
            callbacks=callbacks,
            temperature=settings.temperature,
            top_p=settings.top_p,
            base_url=settings.ollama_base_url,
            system=settings.system_prompt,
            num_ctx=settings.num_ctx, # <-- Set context window size
        )
        # LLM specifically for the question condensing step
        condensing_llm = OllamaLLM(
            model=settings.model,
            temperature=0,
            base_url=settings.ollama_base_url,
            system="You are an AI assistant. Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question.",
            num_ctx=settings.num_ctx, # <-- Set context window size here too
        )

        logger.info("Testing Ollama connection (for main LLM)...")
        test_response = await llm.ainvoke("Respond with exactly 'OK'")
        if "OK" not in test_response:
             logger.warning(f"Ollama test response unexpected: {test_response}")
        logger.info("Ollama connection test successful.")

    except Exception as e:
        logger.error(f"Failed to connect or communicate with Ollama server at {settings.ollama_base_url}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to connect to Ollama LLM: {e}")

    combine_docs_prompt = create_prompt_template()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        condense_question_llm=condensing_llm,
        return_source_documents=True,
        get_chat_history=format_chat_history_for_condensing,
        verbose=(settings.log_level.upper() == "DEBUG"),
        combine_docs_chain_kwargs={
            "prompt": combine_docs_prompt,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            ),
            "document_separator": "\n---\n",
            "document_variable_name": "context"
        }
    )

    return chain, streaming_handler

# --- API Request/Response Models ---

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user's current message/question.")
    history: List[str] = Field(default=[], description="Flat list representing the conversation history (user, AI, user, AI...).")

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    success: bool = True

class ErrorResponse(BaseModel):
    detail: str
    success: bool = False

# --- API Endpoints ---

async def health_check_endpoint(request):
    global app_state
    if not app_state or not app_state.get('settings'):
         return JSONResponse({"status": "uninitialized", "detail": "Application state not ready"}, status_code=503)

    settings = app_state['settings']
    status = {
        "status": "healthy",
        "ollama_url": settings.ollama_base_url,
        "qdrant_collection": settings.qdrant_collection_name,
        "model": settings.model,
        "embedding_model": settings.embeddings_model_name,
        "num_ctx": settings.num_ctx, # Report context size
    }
    if not app_state.get('qa_chain') or not app_state.get('qa_streaming_chain'):
        status["status"] = "degraded"
        status["detail"] = "QA chains not fully initialized."
        return JSONResponse(status, status_code=503)
    return JSONResponse(status)

async def chat_endpoint(request):
    global app_state
    if not app_state or not app_state.get('qa_chain'):
        logger.error("Chat endpoint called but QA chain is not initialized.")
        raise HTTPException(status_code=503, detail="QA service not initialized")

    try:
        data = await request.json()
        chat_request = ChatRequest(**data)
    except json.JSONDecodeError:
        logger.warning("Received invalid JSON in request body.")
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.warning(f"Request body validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid request body: {e}")

    formatted_history: List[Tuple[str, str]] = []
    for i in range(0, len(chat_request.history), 2):
        if i + 1 < len(chat_request.history):
            formatted_history.append((chat_request.history[i], chat_request.history[i+1]))
        else:
            logger.warning("Chat history has an odd number of messages. Ignoring the last one.")

    try:
        logger.info(f"Invoking non-streaming chain for prompt: '{chat_request.prompt[:50]}...'")
        qa_chain = app_state['qa_chain']
        result = await qa_chain.ainvoke({
            "question": chat_request.prompt,
            "chat_history": formatted_history
        })
        logger.info("Non-streaming chain invocation complete.")

        answer = result.get('answer', "Sorry, I couldn't generate a response.").strip()
        source_docs = result.get('source_documents', [])
        source_urls = extract_source_urls(source_docs)
        response_data = ChatResponse(response=answer, sources=source_urls)
        return JSONResponse(response_data.model_dump())

    except Exception as e:
        logger.exception(f"Error during non-streaming chat processing for prompt '{chat_request.prompt[:50]}...'")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")

async def chat_streaming_endpoint(request):
    global app_state
    if not app_state or not app_state.get('qa_streaming_chain') or not app_state.get('streaming_handler'):
        logger.error("Streaming endpoint called but streaming QA chain or handler is not initialized.")
        raise HTTPException(status_code=503, detail="Streaming QA service not initialized")

    try:
        data = await request.json()
        chat_request = ChatRequest(**data)
    except json.JSONDecodeError:
        logger.warning("Received invalid JSON in streaming request body.")
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.warning(f"Streaming request body validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid request body: {e}")

    formatted_history: List[Tuple[str, str]] = []
    for i in range(0, len(chat_request.history), 2):
        if i + 1 < len(chat_request.history):
            formatted_history.append((chat_request.history[i], chat_request.history[i+1]))
        else:
             logger.warning("Streaming chat history has an odd number of messages. Ignoring the last one.")

    streaming_handler = app_state['streaming_handler']
    qa_streaming_chain = app_state['qa_streaming_chain']

    # Reset handler state
    streaming_handler.tokens = []
    while not streaming_handler.token_queue.empty():
        try:
            streaming_handler.token_queue.get_nowait()
            streaming_handler.token_queue.task_done()
        except asyncio.QueueEmpty:
            break

    async def invoke_chain_and_get_sources() -> List[Document]:
        try:
            logger.info(f"Invoking streaming chain for prompt: '{chat_request.prompt[:50]}...'")
            result = await qa_streaming_chain.ainvoke({
                "question": chat_request.prompt,
                "chat_history": formatted_history
            })
            logger.info("Streaming chain invocation task complete.")
            return result.get('source_documents', [])
        except Exception as e:
            logger.exception("Error invoking streaming chain")
            await streaming_handler.on_llm_error(e)
            return []

    invoke_task = asyncio.create_task(invoke_chain_and_get_sources())

    async def stream_generator() -> AsyncGenerator[bytes, None]:
        source_docs_result: List[Document] = []
        stream_interrupted = False
        try:
            while True:
                token = await streaming_handler.token_queue.get()
                if token is None:
                    streaming_handler.token_queue.task_done()
                    logger.debug("Received stream end sentinel (None).")
                    break
                yield token.encode('utf-8')
                streaming_handler.token_queue.task_done()

            logger.debug("Waiting for invoke_task to complete...")
            source_docs_result = await invoke_task
            logger.debug(f"invoke_task completed, received {len(source_docs_result)} source documents.")

        except asyncio.CancelledError:
             logger.warning("Stream generator task cancelled.")
             stream_interrupted = True
             if not invoke_task.done():
                 invoke_task.cancel()
        except Exception as e:
            logger.exception("Error occurred in stream generator loop.")
            stream_interrupted = True
            yield f"\n\nError during streaming response generation: {str(e)}".encode('utf-8')
            if not invoke_task.done():
                 invoke_task.cancel()
        finally:
            if not stream_interrupted and source_docs_result:
                source_urls = extract_source_urls(source_docs_result)
                if source_urls:
                    try:
                        sources_text = "\n\nSources:\n" + "\n".join(f"- {url}" for url in source_urls)
                        yield sources_text.encode('utf-8')
                        logger.info("Appended source URLs to the stream.")
                    except Exception as e:
                         logger.error(f"Error encoding/yielding source URLs: {e}")
            elif stream_interrupted:
                 logger.warning("Stream was interrupted, skipping source URL appending.")
            logger.info("Streaming response generation finished.")

    return StreamingResponse(stream_generator(), media_type="text/plain")

# --- Argument Parsing & Main Execution ---

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CHPC Assistant Server: Answer questions about CHPC documentation using RAG.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    settings_defaults = Settings().model_dump()
    parser.add_argument("--temperature", type=float, help='LLM temperature.')
    parser.add_argument("--top-p", type=float, help='LLM top-p.')
    parser.add_argument("--score-threshold", type=float, help='Retriever score threshold.')
    parser.add_argument("--chunks", type=int, help='Number of chunks to retrieve (retriever_k).')
    # Add argument for num_ctx
    parser.add_argument("--num-ctx", type=int, help='Ollama context window size.')
    parser.add_argument("--host", type=str, help='Server host address.')
    parser.add_argument("--port", type=int, help='Server port number.')
    parser.add_argument("--ollama-url", type=str, help='Ollama server base URL.')
    parser.add_argument("--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level.')
    parser.add_argument("--model", type=str, help='Ollama model name.')
    parser.add_argument("--qdrant-path", type=str, help='Path to local Qdrant data directory.')
    parser.add_argument("--collection", type=str, help='Qdrant collection name.')
    # Set defaults using the potentially updated Settings class defaults
    parser.set_defaults(**settings_defaults)
    return parser.parse_args()

async def main() -> None:
    """
    Main asynchronous function to initialize application components and start the server.
    """
    global app_state

    args = parse_arguments()
    # Prioritize command-line args over environment/defaults
    arg_overrides = {k: v for k, v in vars(args).items() if v is not None}
    if 'chunks' in arg_overrides:
        arg_overrides['retriever_k'] = arg_overrides.pop('chunks')
    if 'collection' in arg_overrides:
         arg_overrides['qdrant_collection_name'] = arg_overrides.pop('collection')
    if 'ollama_url' in arg_overrides:
        arg_overrides['ollama_base_url'] = arg_overrides.pop('ollama_url')
    # Add mapping for num_ctx if provided
    if 'num_ctx' in arg_overrides:
        arg_overrides['num_ctx'] = arg_overrides.pop('num_ctx')


    settings = Settings(**arg_overrides)
    setup_logging(settings.log_level)
    logger.info("Starting CHPC Assistant Server...")
    logger.info(f"Effective Settings: {settings.model_dump_json(indent=2)}")

    try:
        vector_store = await setup_qdrant_vector_store(settings)
        qa_chain, _ = await setup_qa_chain(settings, vector_store, streaming=False)
        qa_streaming_chain, streaming_handler = await setup_qa_chain(settings, vector_store, streaming=True)

        if not qa_chain or not qa_streaming_chain or not streaming_handler:
             raise RuntimeError("Failed to initialize one or more QA chains or the streaming handler.")

        app_state = {
            "settings": settings,
            "vector_store": vector_store,
            "qa_chain": qa_chain,
            "qa_streaming_chain": qa_streaming_chain,
            "streaming_handler": streaming_handler
        }
        logger.info("Application components initialized successfully.")

    except (RuntimeError, Exception) as e:
        logger.critical(f"FATAL: Failed to initialize application components: {e}", exc_info=True)
        logger.warning("Running in DEGRADED mode due to initialization failure. Only /health might work.")
        app_state = {"settings": settings}

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    ]

    routes = [
        Route("/", chat_endpoint, methods=["POST"], name="chat_non_streaming"),
        Route("/stream", chat_streaming_endpoint, methods=["POST"], name="chat_streaming"),
        Route("/health", health_check_endpoint, methods=["GET"], name="health_check"),
    ]

    app = Starlette(
        debug=(settings.log_level.upper() == "DEBUG"),
        routes=routes,
        middleware=middleware
    )

    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        log_config=None # Use Uvicorn's default logging config
    )
    server = uvicorn.Server(config)

    logger.info(f"Starting Uvicorn server on http://{settings.host}:{settings.port}")
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.critical(f"Server failed to start or crashed: {e}", exc_info=True)


