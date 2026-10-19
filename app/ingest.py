import os
import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    DATA_DIR,
    QDRANT_PATH,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    get_collection_name,
)

def clean_text(text):
    # Some PDFs extract one token per line; collapse all whitespace so each
    # chunk holds real content and table rows stay together
    return re.sub(r"\s+", " ", text).strip()

def load_pdf_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append((i + 1, text))
    return pages

# --- Strategy 1: Fixed-size ---
def chunk_fixed(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Strategy 2: Recursive ---
def chunk_recursive(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)

# --- Strategy 3: Semantic ---
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]

def chunk_semantic(text, model, similarity_threshold=0.5):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
        )
        if sim >= similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def get_chunker(strategy, model=None):
    if strategy == "fixed":
        return lambda text: chunk_fixed(text)
    elif strategy == "recursive":
        return lambda text: chunk_recursive(text)
    elif strategy == "semantic":
        return lambda text: chunk_semantic(text, model)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def build_chunks_from_pdf(pdf_path, source_name, chunker):
    pages = load_pdf_pages(pdf_path)
    all_chunks = []
    for page_num, page_text in pages:
        for chunk in chunker(page_text):
            if chunk.strip():
                all_chunks.append({
                    "text": chunk,
                    "source": source_name,
                    "page": page_num,
                })
    return all_chunks

def wipe_collection(client, name):
    """Delete every point in a collection (delete_collection leaves data behind here)."""
    ids = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.extend(p.id for p in points)
        if offset is None:
            break
    if ids:
        client.delete(collection_name=name, points_selector=PointIdsList(points=ids))

def ingest_all_pdfs(strategy="fixed"):
    print(f"Strategy: {strategy}")
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chunker = get_chunker(strategy, model=model)

    client = QdrantClient(path=QDRANT_PATH)
    collection_name = get_collection_name(strategy)

    if client.collection_exists(collection_name):
        wipe_collection(client, collection_name)
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}")
        return

    point_id = 0
    all_points = []

    for pdf_file in pdf_files:
        pdf_path = DATA_DIR / pdf_file
        print(f"Processing {pdf_file}...")
        chunks = build_chunks_from_pdf(pdf_path, source_name=pdf_file, chunker=chunker)
        print(f"  -> {len(chunks)} chunks")

        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True)

        for chunk, embedding in zip(chunks, embeddings):
            all_points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                    },
                )
            )
            point_id += 1

    client.upsert(collection_name=collection_name, points=all_points)
    print(f"Ingested {point_id} total chunks into '{collection_name}'.")

    stored = client.count(collection_name=collection_name).count
    if stored != point_id:
        print(f"WARNING: expected {point_id} points but collection holds {stored}. Stale data?")

if __name__ == "__main__":
    import sys
    strategy = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    ingest_all_pdfs(strategy)