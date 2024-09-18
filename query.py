#!/usr/bin/env python3
import os
import time
import argparse
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Configuration
MODEL = os.environ.get("MODEL", "mistral-nemo:latest")
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-mpnet-base-v2")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "chpc-rag")
TARGET_SOURCE_CHUNKS = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

# Set up logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

def setup_qdrant_client():
    client = QdrantClient(path='./langchain_qdrant')
    config = VectorParams(size=768, distance=Distance.COSINE)

    if not client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
        logging.info("Creating new collection: %s", QDRANT_COLLECTION_NAME)
        client.create_collection(collection_name=QDRANT_COLLECTION_NAME, vectors_config=config)

    return QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME,
                             embedding=HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME))


def create_prompt_template():
    prompt_template = """
    Human: {question}

    You are an AI assistant for CHPC at the University of Utah. Answer the user's question using ONLY the information provided in the context below. Do not use any external knowledge.

    Context:
    {context}

    Instructions:
    1. If the answer is fully contained in the context, provide a clear and detailed response.
    2. If the context doesn't contain enough information to answer the question completely, say: "I don't have enough information to fully answer this question. For more details, please refer to the CHPC documentation at https://www.chpc.utah.edu/documentation"
    3. Do not mention or refer to the context in your answer.
    4. If code examples are relevant and present in the context, include them in your answer.

    Answer the question step-by-step if appropriate.

    Assistant: """

    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def main():
    args = parse_arguments()
    vector_store = setup_qdrant_client()
    retriever = vector_store.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS, "score_threshold": 0.3})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    llm = Ollama(
        model=MODEL,
        callbacks=callbacks,
        temperature=0.3,  # Lower temperature for more focused outputs
        top_p=0.9,  # Adjust top_p for more deterministic responses
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer", k=2)
    prompt = create_prompt_template()

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    try:
        while True:
            query = input("\nEnter a query (or 'Ctrl+C' to quit): ") 
            if query.lower() in ["exit()", "quit", "q"]:
                break
            if query.strip() == "":
                continue

            start_time = time.time()  # Record the start time
            res = qa.invoke({"question": query})
            end_time = time.time()  # Record the end time

            answer = res['answer']
            print(answer)

            elapsed_time = end_time - start_time  # Calculate elapsed time
            logging.info(f"Time elapsed: {elapsed_time:.2f} seconds")  # Print elapsed time

    except KeyboardInterrupt:
        logging.info("Interrupt received.")
    except Exception as e:
        logging.error("An error occurred: %s", e)
    finally:
        logging.info('\n')
        logging.info("Cleaning up and exiting.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='privateGPT: Ask questions to your documents without an internet connection, using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M", action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()


if __name__ == "__main__":
    main()
