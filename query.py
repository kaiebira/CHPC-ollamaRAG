#!/usr/bin/env python3

import os
import time
import argparse
import logging
from typing import List

from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Configuration
MODEL = os.environ.get("MODEL", "llama3.2")
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-mpnet-base-v2")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "chpc-rag_scraped")
TARGET_SOURCE_CHUNKS = int(os.environ.get('TARGET_SOURCE_CHUNKS', 6))

# Set up logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

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

def format_docs(docs):
    """Format documents with explicit source URL sections."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source_url = doc.metadata.get('source_url', 'https://www.chpc.utah.edu/documentation')
        formatted_docs.append(f"""
SECTION {i+1}
SOURCE URL: {source_url}
CONTENT:
{doc.page_content}
""")
    return "\n".join(formatted_docs)

def create_prompt_template() -> PromptTemplate:
    """Create and return a PromptTemplate for the AI assistant."""
    prompt_template = """
    Human: {question}

    You are an AI assistant for CHPC at the University of Utah. Answer the user's question using ONLY the information provided in the context below. Do not use any external knowledge.

    Context:
    {context}

    Instructions:
    1. If the answer is fully contained in the context, provide a clear and detailed response.
    2. If the context doesn't contain enough information to answer the question completely, say: "I don't have enough information to fully answer this question. Please refer to the CHPC documentation at https://www.chpc.utah.edu/documentation"
    3. Do not mention or reference the context or sections in your answer.
    4. If code examples are relevant and present in the context, include them in your answer.
    5. Focus on providing accurate information from the CHPC documentation.

    Answer the question step-by-step if appropriate.

    Assistant: """

    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def setup_qa_chain(vector_store: QdrantVectorStore, callbacks: List[StreamingStdOutCallbackHandler]) -> ConversationalRetrievalChain:
    """Set up and return a ConversationalRetrievalChain instance."""
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
        system="You must always end responses with the exact source URL from the most relevant context section."
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=2
    )
    
    prompt = create_prompt_template()

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            ),
            "document_separator": "\n---\n",
            "document_variable_name": "context"
        }
    )

def handle_user_input(qa_chain: ConversationalRetrievalChain) -> None:
    """Handle user input and generate responses using the QA chain."""
    try:
        while True:
            query = input("\nEnter a query (or 'exit' to quit): ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                break
            if not query:
                continue

            start_time = time.time()
            res = qa_chain.invoke({"question": query})
            end_time = time.time()

            # Extract valid CHPC URLs from source documents
            if 'source_documents' in res:
                source_urls = extract_source_urls(res['source_documents'])
                
                # Log URLs for debugging
                logging.debug(f"Retrieved source URLs: {source_urls}")
                
                # Get the answer and ensure it ends with correct URL
                answer = res['answer'].strip()
                
                # Remove any existing "For more information" line
                if "For more information, see:" in answer:
                    answer = answer.split("For more information, see:")[0].strip()
                
                # Append the correct URL
                answer += f"\n\nFor more detailed information, see: {source_urls[0]}"
                print(answer)
            else:
                print(res['answer'])
            
            logging.info(f"Time elapsed: {end_time - start_time:.2f} seconds")

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
        description='privateGPT: Ask questions to your documents without an internet connection, using the power of LLMs.')
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
