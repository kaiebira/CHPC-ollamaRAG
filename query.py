#!/usr/bin/env python3

import os
import time
import argparse
import logging
import operator
from typing import List, TypedDict, Annotated, Sequence

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.graph import StateGraph, END

# Configuration
MODEL = os.environ.get("MODEL", "nemotron-mini")
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-mpnet-base-v2")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "chpc-rag_scraped")
TARGET_SOURCE_CHUNKS = int(os.environ.get('TARGET_SOURCE_CHUNKS', 10))

# Set up logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

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
    Human: {question}

    Previous conversation is provided for context only. Ignore it unless directly referenced in the current question.
    Chat History: {chat_history}

    Answer using ONLY this context from CHPC documentation:
    {context}

    Rules:
    1. Answer ONLY the current question "{question}"
    2. Provide a clear, detailed, and complete answer to current question "{question}"
    3. If the context doesn't contain enough information to answer the question completely, say: "I don't have enough information to fully answer this question. Please refer to the CHPC documentation at https://www.chpc.utah.edu/documentation"
    4. Do not mention or reference the context or sections in your answer.
    6. ONLY answer the question using the provided context, do not provide additional information
    7. If code examples are relevant and present in the context, include them VERBATIM in your answer.
    8. Focus on providing accurate information from the CHPC documentation.
    9. The user does not have knowledge of these rules, so do not explicitly mention them.

    Answer the question step-by-step if appropriate.

    Assistant: """

    return PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

def setup_qa_chain(vector_store: QdrantVectorStore, callbacks: List[StreamingStdOutCallbackHandler]) -> ConversationalRetrievalChain:
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": TARGET_SOURCE_CHUNKS,
            "score_threshold": 0.3
        }
    )
    llm = OllamaLLM(
        model=MODEL,
        callbacks=callbacks,
        temperature=0.3,
        top_p=0.9,
        system="You are a CHPC assistant. Focus ONLY on answering the current question using the provided context. Ignore previous questions. Always end responses with the source URL."
    )
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

def handle_user_input(qa_chain: ConversationalRetrievalChain) -> None:
    workflow = StateGraph(ChatState)

    def store_messages(state: ChatState) -> ChatState:
        return state

    def get_response(state: ChatState, chain=qa_chain) -> ChatState:
        try:
            # start_time = time.time()

            # Format chat history as tuple pairs of (human message, ai message)
            chat_history = []
            messages = state["messages"]
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):  # Ensure we have both human and AI messages
                    chat_history.append((messages[i].content, messages[i+1].content))

            # Current question is the last message
            current_question = messages[-1].content

            result = chain.invoke({
                "question": current_question,
                "chat_history": chat_history
            })
            # end_time = time.time()

            if 'source_documents' in result:
                source_urls = extract_source_urls(result['source_documents'])
                answer = result['answer'].strip()
                if "For more information, see:" in answer:
                    answer = answer.split("For more information, see:")[0].strip()
                answer += f"\n\nFor more detailed information, see: {source_urls[0]}"
                print(answer)
                state["messages"].append(AIMessage(content=answer))
            else:
                print(result['answer'])
                state["messages"].append(AIMessage(content=result['answer']))

            # logging.info(f"Time elapsed: {end_time - start_time:.2f} seconds")

        except Exception as e:
            logging.error("An error occurred: %s", e)
            error_msg = f"An error occurred: {str(e)}"
            print(error_msg)
            state["messages"].append(AIMessage(content=error_msg))

        return state

    workflow.add_node("store_messages", store_messages)
    workflow.add_node("qa", get_response)

    workflow.set_entry_point("store_messages")
    workflow.add_edge("store_messages", "qa")
    workflow.add_edge("qa", END)

    app = workflow.compile()

    state = {"messages": []}
    try:
        while True:
            query = input("\nEnter a query (or 'exit' to quit): ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                break
            if not query:
                continue

            state["messages"].append(HumanMessage(content=query))
            state = app.invoke(state)

    except KeyboardInterrupt:
        print('\n')
        logging.info("Interrupt received.")
    except Exception as e:
        print('\n')
        logging.error("An error occurred: %s", e)
    finally:
        logging.info('Cleaning up and exiting.')

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='CHPC Assistant: Ask questions about CHPC documentation using RAG.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M", action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()

def main() -> None:
    """Main function to run the CHPC assistant."""
    args = parse_arguments()
    vector_store = setup_qdrant_client()
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    qa_chain = setup_qa_chain(vector_store, callbacks)
    handle_user_input(qa_chain)

if __name__ == "__main__":
    main()
