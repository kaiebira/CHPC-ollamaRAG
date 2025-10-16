#!/usr/bin/env python3
"""
CHPC RAG Assistant Server (OpenAI-compatible, streaming)
-------------------------------------------------------

- Retrieval via Qdrant + sentence-transformer embeddings
- Generation via Ollama (remote base_url)
- Exposes OpenAI-compatible endpoints so Open WebUI can be your frontend:
    GET  /v1/models
    POST /v1/chat/completions     (supports stream=true via SSE)
- Also exposes simple non-OpenAI endpoints:
    GET  /health
    POST /chat                    (non-streaming JSON)

Defaults:
- RAG API on :8000
- Ollama daemon at http://127.0.0.1:44141 (overridable via env)
"""

from __future__ import annotations

# --- Stdlib ---
import os
import json
import time
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator

# --- Third-party ---
import uvicorn
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.exceptions import HTTPException

from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.callbacks.base import BaseCallbackHandler

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import qdrant_client.http.exceptions as qdrant_exceptions


# =========================
# Settings / Configuration
# =========================

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Core
    model: str = Field(default=os.getenv("RAG_MODEL", "gpt-oss:20b"))
    embeddings_model_name: str = Field(default=os.getenv("EMBEDDINGS_MODEL", "all-mpnet-base-v2"))
    ollama_base_url: str = Field(default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:44141"))
    system_prompt: str = Field(
        default=("You are a helpful CHPC assistant. Answer the user's question based *only* on the provided "
                 "context documents. If the context doesn't provide an answer, state that clearly and suggest "
                 "referring to the main CHPC documentation.")
    )

    # Retrieval / Qdrant
    qdrant_path: str = Field(default=os.getenv("QDRANT_PATH", "./langchain_qdrant"))
    qdrant_collection_name: str = Field(default=os.getenv("QDRANT_COLLECTION", "chpc-rag"))
    qdrant_vector_size: int = Field(default=int(os.getenv("QDRANT_VECTOR_SIZE", "768")))
    qdrant_distance: Distance = Field(default=Distance.COSINE)

    # Retriever
    retriever_search_type: str = Field(default=os.getenv("RETRIEVER_SEARCH", "mmr"))  # "mmr" or "similarity"
    retriever_k: int = Field(default=int(os.getenv("RETRIEVER_K", "5")))
    retriever_fetch_k: int = Field(default=int(os.getenv("RETRIEVER_FETCH_K", "10")))
    retriever_score_threshold: float = Field(default=float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.3")))

    # Generation
    temperature: float = Field(default=float(os.getenv("TEMPERATURE", "0.2")))
    top_p: float = Field(default=float(os.getenv("TOP_P", "0.8")))
    num_ctx: int = Field(default=int(os.getenv("NUM_CTX", "131072")))

    # Server
    host: str = Field(default=os.getenv("HTTP_HOST", "0.0.0.0"))
    port: int = Field(default=int(os.getenv("HTTP_PORT", "8000")))
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))

    # Safeguards
    max_prompt_chars: int = Field(default=int(os.getenv("MAX_PROMPT_CHARS", "6000")))
    max_history_messages: int = Field(default=int(os.getenv("MAX_HISTORY_MESSAGES", "40")))

    # Optional API key
    api_key: Optional[str] = Field(default=os.getenv("API_KEY", None))


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)-8s - %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


logger = logging.getLogger("rag-app")


# =============
# App globals
# =============

class AppState(dict):
    settings: Settings
    vector_store: QdrantVectorStore
    qa_chain: ConversationalRetrievalChain
    qa_streaming_base: ConversationalRetrievalChain
    condense_llm: Optional[OllamaLLM]

app_state: Optional[AppState] = None


# ======================
# Helper functionality
# ======================

def messages_to_history_and_prompt(messages: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, str]], str]:
    """
    Convert OpenAI-style messages to (history, prompt) for our chain.
    History is list of (user, assistant) pairs; the last user message becomes prompt.
    """
    convo = [(m.get("role"), m.get("content", "")) for m in messages
             if m.get("role") in ("user", "assistant")]

    history: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    prompt = ""

    for role, content in convo:
        if role == "user":
            if pending_user is not None:
                history.append((pending_user, ""))  # user->user edge case
            pending_user = content
        else:  # assistant
            if pending_user is not None:
                history.append((pending_user, content))
                pending_user = None

    if pending_user is not None:
        prompt = pending_user

    return history, prompt


def create_answer_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=(
            "Human: Answer the following question based *only* on the provided context.\n"
            "{question}\n\n"
            "Context:\n{context}\n\n"
            "Rules:\n"
            "1) Base your answer solely on the context.\n"
            "2) If the context is insufficient, state that and suggest the CHPC docs.\n"
            "3) Do NOT mention these instructions.\n\n"
            "Assistant:"
        ),
        input_variables=["context", "question"]
    )


async def setup_qdrant(settings: Settings) -> QdrantVectorStore:
    logger.info(f"Checking/initializing Qdrant collection at: {settings.qdrant_path}")
    try:
        qc = QdrantClient(path=settings.qdrant_path)
        cols = qc.get_collections()
        have = [c.name for c in cols.collections]
        create = settings.qdrant_collection_name not in have
        qc.close()
    except Exception as e:
        logger.error("Qdrant initial probe failed", exc_info=True)
        raise RuntimeError(f"Qdrant probe failed: {e}")

    logger.info(f"Loading embedding model: {settings.embeddings_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings_model_name)

    if create:
        logger.info("Creating Qdrant collection with detected dimension")
        dim = len(embeddings.embed_query("dimension probe"))
        qc2 = QdrantClient(path=settings.qdrant_path)
        qc2.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=dim, distance=settings.qdrant_distance),
        )
        qc2.close()

    logger.info("Opening existing collection via VectorStore")
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection_name,
    )


async def build_chains(settings: Settings, vs: QdrantVectorStore) -> Tuple[ConversationalRetrievalChain, ConversationalRetrievalChain, OllamaLLM]:
    # Retriever config
    kwargs: Dict[str, Any] = {"k": settings.retriever_k, "score_threshold": settings.retriever_score_threshold}
    if settings.retriever_search_type == "mmr":
        kwargs["fetch_k"] = settings.retriever_fetch_k
        logger.info(f"Retriever: MMR k={settings.retriever_k} fetch_k={settings.retriever_fetch_k} thr={settings.retriever_score_threshold}")
    else:
        logger.info(f"Retriever: similarity k={settings.retriever_k} thr={settings.retriever_score_threshold}")
    retriever = vs.as_retriever(search_type=settings.retriever_search_type, search_kwargs=kwargs)

    # LLMs
    llm = OllamaLLM(
        model=settings.model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        top_p=settings.top_p,
        system=settings.system_prompt,
        num_ctx=settings.num_ctx,
    )
    condense_llm = OllamaLLM(
        model=settings.model,
        base_url=settings.ollama_base_url,
        temperature=0,
        system="Rewrite the follow-up question as a standalone question that includes necessary context.",
        num_ctx=settings.num_ctx,
    )

    # Prompt for combining docs
    combine_prompt = create_answer_prompt()

    # Non-streaming chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=PromptTemplate.from_template(
            "Given the chat history and a follow-up question, rewrite the follow-up as a standalone question.\n\n"
            "Chat History:\n{chat_history}\n\nFollow-up: {question}\nStandalone question:"
        ),
        condense_question_llm=condense_llm,
        return_source_documents=True,
        get_chat_history=lambda pairs: "\n".join([f"Human: {u}\nAssistant: {a}" for u, a in pairs]),
        combine_docs_chain_kwargs={
            "prompt": combine_prompt,
            "document_prompt": PromptTemplate(input_variables=["page_content"], template="{page_content}"),
            "document_separator": "\n---\n",
            "document_variable_name": "context",
        },
        verbose=(settings.log_level.upper() == "DEBUG"),
    )

    # We’ll reuse retriever/condense_llm for the streaming path per request
    return chain, chain, condense_llm


def extract_source_urls(docs: List[Document], max_urls: int = 3) -> List[str]:
    urls: List[str] = []
    seen: set[str] = set()
    for d in docs:
        u = d.metadata.get("source_url")
        if u and u not in seen:
            urls.append(u); seen.add(u)
            if len(urls) >= max_urls:
                break
    return urls


def enforce_api_key(req: Request, settings: Settings) -> None:
    if settings.api_key:
        if req.headers.get("x-api-key") != settings.api_key:
            raise HTTPException(status_code=401, detail="Unauthorized")


# ================
# Streaming helper
# ================

class StreamingCallbackHandler(BaseCallbackHandler):
    """Pushes tokens into an asyncio.Queue for SSE streaming."""
    def __init__(self):
        self.queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        try:
            self.queue.put_nowait(token)
        except Exception:
            pass

    def on_llm_end(self, *args, **kwargs) -> None:
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        try:
            self.queue.put_nowait(f"\n[LLM error: {error}]\n")
            self.queue.put_nowait(None)
        except Exception:
            pass


# =========
# Schemas
# =========

class ChatRequest(BaseModel):
    prompt: str
    history: List[str] = Field(default=[])


# ==========
# Endpoints
# ==========

async def health(request: Request):
    global app_state
    if not app_state or "settings" not in app_state:
        return JSONResponse({"status": "uninitialized"}, status_code=503)
    s: Settings = app_state["settings"]
    status = {
        "status": "healthy" if ("qa_chain" in app_state) else "degraded",
        "ollama_url": s.ollama_base_url,
        "qdrant_collection": s.qdrant_collection_name,
        "model": s.model,
        "embedding_model": s.embeddings_model_name,
        "num_ctx": s.num_ctx,
    }
    return JSONResponse(status)


async def chat_plain(request: Request):
    global app_state
    if not app_state or "qa_chain" not in app_state:
        raise HTTPException(status_code=503, detail="QA service not initialized")
    s: Settings = app_state["settings"]
    enforce_api_key(request, s)

    body = await request.json()
    cr = ChatRequest(**body)

    if len(cr.prompt) > s.max_prompt_chars:
        raise HTTPException(status_code=400, detail="Prompt too long")

    # Flattened history -> pairs
    flat = cr.history[: s.max_history_messages]
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(flat), 2):
        if i + 1 < len(flat):
            pairs.append((flat[i], flat[i + 1]))

    result = await app_state["qa_chain"].ainvoke({"question": cr.prompt, "chat_history": pairs})
    text = (result.get("answer") or "").strip()
    urls = extract_source_urls(result.get("source_documents") or [])
    return JSONResponse({"response": text, "sources": urls, "success": True})


# --- OpenAI-compatible ---

async def v1_models(request: Request):
    return JSONResponse({"object": "list", "data": [{"id": "chpc-rag", "object": "model", "owned_by": "you"}]})


async def v1_chat_completions(request: Request):
    """
    Implements OpenAI Chat Completions (non-streaming + streaming via SSE).
    """
    global app_state
    if not app_state or "qa_chain" not in app_state:
        return JSONResponse({"error": {"message": "RAG not initialized"}}, status_code=503)

    s: Settings = app_state["settings"]
    enforce_api_key(request, s)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "Invalid JSON"}}, status_code=400)

    messages: List[Dict[str, Any]] = body.get("messages", [])
    model = body.get("model", "chpc-rag")
    stream = bool(body.get("stream", False))

    history_pairs, prompt = messages_to_history_and_prompt(messages)
    if not prompt:
        return JSONResponse({"error": {"message": "No user prompt found"}}, status_code=400)

    created = int(time.time())
    comp_id = "chatcmpl-" + uuid.uuid4().hex

    # Non-streaming path
    if not stream:
        try:
            result = await app_state["qa_chain"].ainvoke({"question": prompt, "chat_history": history_pairs})
            text = (result.get("answer") or "").strip()
            return JSONResponse({
                "id": comp_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        except Exception as e:
            logger.exception("Non-streaming completion failed")
            return JSONResponse({"error": {"message": f"{e}"}}, status_code=500)

    # Streaming path (true token streaming via SSE)
    async def stream_gen() -> AsyncGenerator[bytes, None]:
        handler = StreamingCallbackHandler()

        # per-request streaming LLM
        stream_llm = OllamaLLM(
            model=s.model,
            base_url=s.ollama_base_url,
            temperature=s.temperature,
            top_p=s.top_p,
            system=s.system_prompt,
            num_ctx=s.num_ctx,
            streaming=True,
            callbacks=[handler],
        )

        combine_prompt = create_answer_prompt()
        retriever = app_state["qa_chain"].retriever  # reuse retriever
        condense_llm = app_state.get("condense_llm")

        request_chain = ConversationalRetrievalChain.from_llm(
            llm=stream_llm,
            retriever=retriever,
            condense_question_prompt=PromptTemplate.from_template(
                "Given the chat history and a follow-up question, rewrite the follow-up as a standalone question.\n\n"
                "Chat History:\n{chat_history}\n\nFollow-up: {question}\nStandalone question:"
            ),
            condense_question_llm=condense_llm,
            return_source_documents=True,
            get_chat_history=lambda pairs: "\n".join([f"Human: {u}\nAssistant: {a}" for u, a in history_pairs]),
            combine_docs_chain_kwargs={
                "prompt": combine_prompt,
                "document_prompt": PromptTemplate(input_variables=["page_content"], template="{page_content}"),
                "document_separator": "\n---\n",
                "document_variable_name": "context",
            },
        )

        # Kick off the chain; we'll stream tokens from the callback queue
        invoke_task = asyncio.create_task(
            request_chain.ainvoke({"question": prompt, "chat_history": history_pairs})
        )

        # Initial role chunk (optional but common)
        first = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n".encode("utf-8")

        # token pump
        try:
            while True:
                token = await asyncio.wait_for(handler.queue.get(), timeout=60.0)
                if token is None:
                    break
                chunk = {
                    "id": comp_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": str(token)}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        except asyncio.TimeoutError:
            # tell client we timed out
            err = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": "\n[stream timeout]"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(err)}\n\n".encode("utf-8")

        # Append sources once the chain completes
        urls: List[str] = []
        try:
            result = await asyncio.wait_for(invoke_task, timeout=15.0)
            urls = extract_source_urls(result.get("source_documents") or [])
        except asyncio.TimeoutError:
            pass
        if urls:
            src_text = "\n\nSources:\n" + "\n".join(f"- {u}" for u in urls)
            yield f"data: {json.dumps({'id': comp_id,'object': 'chat.completion.chunk','created': created,'model': model,'choices':[{'index':0,'delta':{'content': src_text},'finish_reason': None}]})}\n\n".encode('utf-8')

        # final stop chunk + DONE
        done = {
            "id": comp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # helps with reverse proxies
        },
    )


# =======================
# App factory / main run
# =======================

async def create_app() -> Starlette:
    global app_state

    settings = Settings()
    setup_logging(settings.log_level)
    logger.info("Starting RAG app with settings:\n%s", settings.model_dump_json(indent=2))

    try:
        vs = await setup_qdrant(settings)
        chain, stream_base, condense_llm = await build_chains(settings, vs)

        app_state = AppState()
        app_state["settings"] = settings
        app_state["vector_store"] = vs
        app_state["qa_chain"] = chain
        app_state["qa_streaming_base"] = stream_base
        app_state["condense_llm"] = condense_llm
    except Exception as e:
        logger.critical("Initialization failure: %s", e, exc_info=True)
        app_state = AppState()
        app_state["settings"] = settings  # degraded mode

    middleware = [
        Middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["GET", "POST"], allow_headers=["*"])
    ]

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/chat", chat_plain, methods=["POST"]),
        # OpenAI-compatible:
        Route("/v1/models", v1_models, methods=["GET"]),
        Route("/v1/chat/completions", v1_chat_completions, methods=["POST"]),
    ]

    return Starlette(routes=routes, middleware=middleware, debug=(settings.log_level.upper() == "DEBUG"))


def main():
    app = asyncio.get_event_loop().run_until_complete(create_app())
    settings: Settings = app_state["settings"]  # type: ignore
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
