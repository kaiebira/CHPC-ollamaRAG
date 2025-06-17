import logging
import os
import glob
from typing import List, Tuple, Type, Optional
from multiprocessing import Pool
import re
from tqdm import tqdm

import nltk
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    CSVLoader, EverNoteLoader, PyMuPDFLoader, TextLoader, UnstructuredEPubLoader,
    BSHTMLLoader, UnstructuredMarkdownLoader, UnstructuredODTLoader,
    UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
)
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Constants
SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', 'source_documents/chpc_utah')
EMBEDDINGS_MODEL_NAME = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-mpnet-base-v2')
QDRANT_COLLECTION_NAME = os.environ.get('QDRANT_COLLECTION_NAME', 'chpc-rag')
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 500

# Configure logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define loader mapping
LOADER_MAPPING: dict[str, Tuple[Type, dict]] = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (BSHTMLLoader, {"get_text_separator": " "}), # Use BSHTMLLoader for HTML
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def extract_source_url(content: str) -> Optional[str]:
    """Extract source URL from HTML comments at the start of the file."""
    url_pattern = r'<!--\s*Original URL:\s*(https?://[^\s>]+)\s*-->'
    match = re.search(url_pattern, content[:500])  # Only search start of file
    return match.group(1) if match else None

def load_single_document(file_path: str) -> List[Document]:
    """Load a single document from a file path."""
    if os.path.getsize(file_path) == 0:
        logging.warning(f"Empty file {file_path}. Ignoring it.")
        return []

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in LOADER_MAPPING:
        logging.warning(f"Unsupported file extension {ext} for {file_path}. Ignoring it.")
        return []

    loader_class, loader_args = LOADER_MAPPING[ext]
    try:
        # For HTML files, extract source URL
        source_url = None
        if ext == '.html':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            source_url = extract_source_url(content)

        # Load the document using the appropriate loader
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()

        # Add metadata to each document
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            # BSHTMLLoader puts source in metadata by default, but ensure file_path is there
            doc.metadata['file_path'] = file_path
            if source_url:
                doc.metadata['source_url'] = source_url
            # Keep track of original source for deduplication
            # Use source_url if available, otherwise use file_path
            doc.metadata['source'] = source_url if source_url else file_path

        return documents

    except Exception as e:
        logging.warning(f"Error loading file {file_path}: {e}")
        return []


def get_existing_sources(client: QdrantClient) -> List[str]:
    """Get existing document sources from the Qdrant collection."""
    existing_docs = client.scroll(
        collection_name=QDRANT_COLLECTION_NAME,
        scroll_filter=None,
        limit=10000, # Adjust limit as needed, or implement pagination
        with_payload=True,
        with_vectors=False,
    )[0]
    # Ensure payload exists before trying to access 'source'
    return [doc.payload.get('source') for doc in existing_docs if doc.payload and doc.payload.get('source')]


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """Load all documents from a source directory, ignoring specified files."""
    all_files = [
        f for ext in LOADER_MAPPING
        for f in glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        if f not in ignored_files
    ]

    with Pool(processes=8) as pool:
        results = []
        with tqdm(total=len(all_files), desc='Loading new documents', ncols=80) as pbar:
            for docs in pool.imap_unordered(load_single_document, all_files):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """Load documents, split them into chunks, and return the chunks."""
    logging.info(f"Loading documents from /{SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY, ignored_files)

    if not documents:
        logging.info("No new documents to load.")
        return []

    logging.info(f"Loaded {len(documents)} new documents from {SOURCE_DIRECTORY}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(chunks)} chunks of text (max. {CHUNK_SIZE} tokens each)")
    return chunks


def setup_qdrant_client() -> QdrantClient:
    """Set up and return a Qdrant client."""
    return QdrantClient(path='./langchain_qdrant')


def setup_vector_store(client: QdrantClient, embeddings: HuggingFaceEmbeddings) -> QdrantVectorStore:
    """Set up and return a QdrantVectorStore."""
    if client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
        logging.info(f'Found existing collection: {QDRANT_COLLECTION_NAME}')
        return QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME, embedding=embeddings)

    logging.info(f"Creating new collection: {QDRANT_COLLECTION_NAME}")
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    return QdrantVectorStore.from_documents(
        [],
        embeddings,
        url="localhost",
        collection_name=QDRANT_COLLECTION_NAME,
    )


def main():
    client = setup_qdrant_client()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
    vector_store = setup_vector_store(client, embeddings)

    # NOTE: This current implementation fetches all existing source identifiers
    # and skips processing files whose source identifier is already in Qdrant.
    # This means if a file is updated, it WILL NOT be re-ingested unless
    # the Qdrant collection is cleared first or a more sophisticated update
    # mechanism (checking modification times/hashes) is implemented.
    existing_sources = get_existing_sources(client)
    logging.info(f"Found {len(existing_sources)} existing sources in Qdrant. These will be skipped.")
    chunks = process_documents(ignored_files=existing_sources)

    if chunks:
        logging.info(f"Creating embeddings for {len(chunks)} new chunks. This may take some time...")
        vector_store.add_documents(chunks)
        logging.info("Ingestion complete.")
    else:
        logging.info("No new documents to process.")

if __name__ == "__main__":
    main()
