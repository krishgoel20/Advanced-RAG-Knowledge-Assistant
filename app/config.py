import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
QDRANT_PATH = str(BASE_DIR / "qdrant_data")

# --- Qdrant ---
QDRANT_COLLECTION_NAME = "rag_knowledge_assistant"
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2's output size

# --- Embedding model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 1000       # characters per chunk (fixed-size & recursive)
CHUNK_OVERLAP = 100     # characters of overlap between chunks

CHUNKING_STRATEGIES = ["fixed", "recursive", "semantic"]

def get_collection_name(strategy: str) -> str:
    return f"{QDRANT_COLLECTION_NAME}_{strategy}"

# --- Retrieval ---
TOP_K = 5              # number of chunks to retrieve for Version 1

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"