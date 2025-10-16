import logging
import os
import glob
import argparse
import hashlib
import time
from typing import List, Tuple, Type, Optional, Dict, Any
from multiprocessing import Pool, cpu_count
import re
from tqdm import tqdm

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
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

############################
# Defaults / Environment
############################
ENV_SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', 'source_documents/chpc_utah')
ENV_EMBEDDINGS_MODEL_NAME = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-mpnet-base-v2')
ENV_QDRANT_COLLECTION_NAME = os.environ.get('QDRANT_COLLECTION_NAME', 'chpc-rag')
DEFAULT_CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1500'))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '500'))
DEFAULT_SCROLL_PAGE = int(os.environ.get('QDRANT_SCROLL_LIMIT', '1000'))
DEFAULT_BATCH_SIZE = int(os.environ.get('INGEST_BATCH_SIZE', '256'))
DEFAULT_POOL_SIZE = int(os.environ.get('INGEST_POOL_SIZE', '0'))  # 0 -> auto

# Configure logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO
)

# Disable tokenizers parallelism ( huggingface )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define loader mapping
LOADER_MAPPING: dict[str, Tuple[Type, dict]] = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (BSHTMLLoader, {"get_text_separator": " "}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def extract_source_url(content: str) -> Optional[str]:
    """Extract source URL from HTML comments near the file start."""
    url_pattern = r'<!--\s*Original URL:\s*(https?://[^\s>]+)\s*-->'
    head = content[:1000]
    match = re.search(url_pattern, head)
    return match.group(1) if match else None


def file_hash(path: str, block_size: int = 65536) -> str:
    """Compute SHA256 hash of a file (streaming)."""
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            sha.update(data)
    return sha.hexdigest()


def get_pool_size(requested: int) -> int:
    if requested and requested > 0:
        return requested
    # Leave one core free if possible
    cores = cpu_count()
    return max(1, cores - 1)

def load_single_document(args: Tuple[str, Dict[str, Any]]) -> List[Document]:
    """Load a single document from a file path. (Executed in worker processes.)

    Args tuple: (file_path, base_metadata)
    """
    file_path, base_meta = args
    try:
        if os.path.getsize(file_path) == 0:
            logging.warning(f"Empty file {file_path}. Skipping.")
            return []

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in LOADER_MAPPING:
            logging.warning(f"Unsupported extension {ext} for {file_path}. Skipping.")
            return []

        loader_class, loader_args = LOADER_MAPPING[ext]
        source_url = None
        if ext == '.html':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    head = f.read(4000)
                source_url = extract_source_url(head)
            except Exception:
                pass

        loader = loader_class(file_path, **loader_args)
        documents = loader.load()
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata.update(base_meta)
            if source_url:
                doc.metadata['source_url'] = source_url
            doc.metadata['source'] = source_url if source_url else base_meta['file_path']
        return documents
    except Exception as e:
        logging.warning(f"Error loading file {file_path}: {e}")
        return []


def scroll_all_points(client: QdrantClient, collection: str, page: int) -> List[Any]:
    records_all = []
    offset = None
    while True:
        try:
            result = client.scroll(
                collection_name=collection,
                scroll_filter=None,
                limit=page,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
        except TypeError:
            # Older client versions may not support 'offset' kw when None
            result = client.scroll(
                collection_name=collection,
                scroll_filter=None,
                limit=page,
                with_payload=True,
                with_vectors=False,
            )
        records = result[0]
        records_all.extend(records)
        next_offset = result[1] if len(result) > 1 else None
        if not records or next_offset is None or len(records) < page:
            break
        offset = next_offset
    return records_all


def get_existing_index(client: QdrantClient, collection: str, page: int) -> Dict[str, Dict[str, Any]]:
    """Build an index: source -> representative payload (first occurrence)."""
    if not client.collection_exists(collection_name=collection):
        return {}
    points = scroll_all_points(client, collection, page)
    index: Dict[str, Dict[str, Any]] = {}
    for p in points:
        payload = getattr(p, 'payload', None) or {}
        source = payload.get('source')
        if source and source not in index:
            index[source] = payload
    return index


def discover_files(source_dir: str) -> List[str]:
    return [
        f for ext in LOADER_MAPPING
        for f in glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
    ]


def plan_ingestion(files: List[str], existing_index: Dict[str, Dict[str, Any]], force: bool) -> Tuple[List[str], List[str], List[str]]:
    """Return (new_files, modified_files, skipped_files)."""
    new_files, modified_files, skipped = [], [], []
    for fp in files:
        try:
            size = os.path.getsize(fp)
            mtime = int(os.path.getmtime(fp))
        except OSError:
            continue
        ext = os.path.splitext(fp)[1].lower()
        if ext not in LOADER_MAPPING:
            skipped.append(fp)
            continue
        # Determine source identifier (may be URL for html after load; here use path as preliminary)
        source_id = fp  # updated to URL later if present
        existing = existing_index.get(source_id)
        if force or not existing:
            # We'll verify hash only if not forced and existing present.
            if not existing:
                new_files.append(fp)
            else:
                modified_files.append(fp)
            continue
        # If existing has a stored hash we can compare
        stored_hash = existing.get('doc_content_sha256')
        if stored_hash:
            try:
                current_hash = file_hash(fp)
            except Exception:
                skipped.append(fp)
                continue
            if current_hash != stored_hash:
                modified_files.append(fp)
            else:
                skipped.append(fp)
        else:
            # No hash previously -> treat as unchanged for now
            skipped.append(fp)
    return new_files, modified_files, skipped


def load_documents(selected_files: List[str], pool_size: int) -> List[Document]:
    tasks = []
    for fp in selected_files:
        try:
            meta = {
                'file_path': fp,
                'file_size': os.path.getsize(fp),
                'file_mtime': int(os.path.getmtime(fp)),
                'doc_content_sha256': file_hash(fp)
            }
        except OSError:
            continue
        tasks.append((fp, meta))

    if not tasks:
        return []

    results: List[Document] = []
    with Pool(processes=pool_size) as pool:
        with tqdm(total=len(tasks), desc='Loading documents', ncols=90) as pbar:
            for docs in pool.imap_unordered(load_single_document, tasks):
                results.extend(docs)
                pbar.update()
    return results


def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    # Add chunk indices metadata
    # Group by document-level id (hash)
    by_doc: Dict[str, List[Document]] = {}
    for c in chunks:
        doc_id = c.metadata.get('doc_content_sha256') or c.metadata.get('file_path')
        by_doc.setdefault(doc_id, []).append(c)
    for doc_id, parts in by_doc.items():
        total = len(parts)
        for idx, part in enumerate(parts):
            part.metadata['chunk_index'] = idx
            part.metadata['chunk_count'] = total
            part.metadata['chunk_id'] = f"{doc_id}:{idx}"
    return chunks


def setup_qdrant_client() -> QdrantClient:
    return QdrantClient(path='./langchain_qdrant')


def get_embedding_dimension(embeddings: HuggingFaceEmbeddings) -> int:
    # Try library-specific API, fallback to probing
    try:
        if hasattr(embeddings, 'client') and hasattr(embeddings.client, 'get_sentence_embedding_dimension'):
            return embeddings.client.get_sentence_embedding_dimension()  # sentence-transformers
    except Exception:
        pass
    try:
        test_vec = embeddings.embed_query("dimension probe")
        return len(test_vec)
    except Exception as e:
        raise RuntimeError(f"Cannot determine embedding dimension: {e}")


def setup_vector_store(client: QdrantClient, embeddings: HuggingFaceEmbeddings, collection: str) -> QdrantVectorStore:
    if client.collection_exists(collection_name=collection):
        logging.info(f"Using existing collection: {collection}")
        return QdrantVectorStore(client=client, collection_name=collection, embedding=embeddings)
    dim = get_embedding_dimension(embeddings)
    logging.info(f"Creating collection '{collection}' (dim={dim})")
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    return QdrantVectorStore.from_documents([], embeddings, url="localhost", collection_name=collection)


def delete_source_points(client: QdrantClient, collection: str, source: str):
    try:
        flt = Filter(must=[FieldCondition(key='source', match=MatchValue(value=source))])
        client.delete(collection_name=collection, points_selector=flt)
    except Exception as e:
        logging.warning(f"Failed to delete old points for source={source}: {e}")


def ingest_chunks(vector_store: QdrantVectorStore, chunks: List[Document], batch_size: int):
    if not chunks:
        return
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        vector_store.add_documents(batch)
        logging.info(f"Embedded batch {start // batch_size + 1} ({end - start} docs) [{end}/{total}]")


def summarize(new_count: int, modified_count: int, skipped_count: int, chunk_total: int, duration: float):
    logging.info(
        f"Summary: new={new_count} modified={modified_count} skipped={skipped_count} chunks={chunk_total} time={duration:.1f}s"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    p.add_argument('--source-dir', default=ENV_SOURCE_DIRECTORY)
    p.add_argument('--model', default=ENV_EMBEDDINGS_MODEL_NAME)
    p.add_argument('--collection', default=ENV_QDRANT_COLLECTION_NAME)
    p.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument('--chunk-overlap', type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument('--scroll-page', type=int, default=DEFAULT_SCROLL_PAGE, help='Qdrant scroll page size')
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Embedding batch size (logical)')
    p.add_argument('--pool-size', type=int, default=DEFAULT_POOL_SIZE, help='File loader process pool size')
    p.add_argument('--force-reindex', action='store_true', help='Reindex all files regardless of existing hashes')
    p.add_argument('--dry-run', action='store_true', help='Plan & report only; no embeddings upload')
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    client = setup_qdrant_client()
    embeddings = HuggingFaceEmbeddings(model_name=args.model)
    vector_store = setup_vector_store(client, embeddings, args.collection)

    existing_index = get_existing_index(client, args.collection, args.scroll_page)
    logging.info(f"Existing unique sources in collection: {len(existing_index)}")

    all_files = discover_files(args.source_dir)
    logging.info(f"Discovered {len(all_files)} candidate files in {args.source_dir}")
    new_files, modified_files, skipped_files = plan_ingestion(all_files, existing_index, args.force_reindex)
    logging.info(f"Plan -> new: {len(new_files)}, modified: {len(modified_files)}, unchanged: {len(skipped_files)}")

    selected_files = new_files + modified_files
    if not selected_files:
        summarize(0, 0, len(skipped_files), 0, time.time() - t0)
        return

    pool_sz = get_pool_size(args.pool_size)
    logging.info(f"Loading {len(selected_files)} files (pool={pool_sz})")
    documents = load_documents(selected_files, pool_sz)
    logging.info(f"Loaded {len(documents)} raw document segments (pre-split)")

    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    logging.info(f"Split into {len(chunks)} chunks of text (max. {args.chunk_size} characters each)")

    # Delete old points for modified sources before re-adding
    if modified_files and not args.dry_run:
        for fp in modified_files:
            delete_source_points(client, args.collection, fp)

    if args.dry_run:
        logging.info("Dry run complete; no embeddings ingested.")
        summarize(len(new_files), len(modified_files), len(skipped_files), len(chunks), time.time() - t0)
        return

    logging.info(f"Embedding & uploading {len(chunks)} chunks (batch={args.batch_size})")
    ingest_chunks(vector_store, chunks, args.batch_size)
    summarize(len(new_files), len(modified_files), len(skipped_files), len(chunks), time.time() - t0)
    logging.info("Ingestion complete.")

if __name__ == "__main__":
    main()
