import logging
import os
import glob
import nltk
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

# LangChain imports
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Set up logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

# Constants
SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
EMBEDDINGS_MODEL_NAME = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-mpnet-base-v2')
QDRANT_COLLECTION_NAME = os.environ.get('QDRANT_COLLECTION_NAME', 'chpc-rag')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 500

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def load_single_document(file_path: str) -> List[Document]:
    """
    Load a single document from a file path.

    Args:
        file_path (str): The file path to load.

    Returns:
        List[Document]: A list of documents loaded from the file.
    """
    if os.path.getsize(file_path) != 0:
        filename, ext = os.path.splitext(file_path)
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            try:
                loader = loader_class(file_path, **loader_args)
                if loader:
                    return loader.load()
            except Exception as e:
                print('\n')
                logging.warning(f"Corrupted file {file_path}. Ignoring it. Error: {e}")
        else:
            print('\n')
            logging.warning(f"Unsupported file {file_path}. Ignoring it.")
    else:
        print('\n')
        logging.warning(f"Empty file {file_path}. Ignoring it.")
    return []

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Load all documents from a source directory, ignoring specified files.

    Args:
        source_dir (str): The source directory to load from.
        ignored_files (List[str], optional): A list of files to ignore. Defaults to [].

    Returns:
        List[Document]: A list of documents loaded from the source directory.
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=8) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                if docs:
                    results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents, split them into chunks, and return the chunks.

    Args:
        ignored_files (List[str], optional): A list of files to ignore. Defaults to [].

    Returns:
        List[Document]: A list of chunks.
    """
    logging.info(f"Loading documents from /{SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY, ignored_files)
    if not documents:
        logging.info("No new documents to load.")
        exit(0)
    logging.info(f"Loaded {len(documents)} new documents from {SOURCE_DIRECTORY}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(texts)} chunks of text (max. {CHUNK_SIZE} tokens each)")
    return texts

def main():
    # Set up Qdrant client
    client = QdrantClient(path='./langchain_qdrant')
    config = VectorParams(
        size=768,  # vector size 768 for sentence-transformers/all-mpnet-base-v2
        distance=Distance.COSINE  # similarity distance
    )

    # Set up embeddings model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    # Check if collection exists
    if client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
        # Grab existing collection
        logging.info('Found existing collection: ' + QDRANT_COLLECTION_NAME)
        vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME,
                                         embedding=embeddings)

        # Get existing document sources
        existing_docs = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=None,
            limit=10000,  # Adjust this if you have more documents
            with_payload=True,
            with_vectors=False,
        )[0]
        existing_sources = [doc.payload.get('source') for doc in existing_docs if doc.payload]

        # Process new documents
        chunks = process_documents(ignored_files=existing_sources)
        if chunks:
            logging.info(f"Creating embeddings for {len(chunks)} new chunks. May take some minutes...")
            vector_store.add_documents(chunks)
        else:
            logging.info("No new documents to process.")
    else:
        # Create new collection
        logging.info("Creating new collection: " + QDRANT_COLLECTION_NAME)
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=config
        )

        # Process documents and create vector store
        chunks = process_documents()
        logging.info(f"Creating embeddings for {len(chunks)} documents. May take some minutes...")
        vector_store = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            url="localhost",
            collection_name=QDRANT_COLLECTION_NAME,
        )

    logging.info(f"Ingestion complete.")

if __name__ == "__main__":
    main()
