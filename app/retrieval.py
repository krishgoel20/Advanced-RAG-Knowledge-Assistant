from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from app.config import (
    QDRANT_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K,
    get_collection_name,
)

_model = None
_client = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client

def retrieve(query, strategy="fixed", top_k=TOP_K):
    """Embed the query, search Qdrant, return list of chunk dicts with scores."""
    model = get_model()
    client = get_client()

    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=get_collection_name(strategy),
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "id": hit.id,
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "page": hit.payload["page"],
            "score": hit.score,
        }
        for hit in results
    ]

if __name__ == "__main__":
    import sys
    query = input("Enter a test query: ")
    strategies = ["fixed", "recursive", "semantic"]

    for strategy in strategies:
        print(f"\n{'='*60}\nSTRATEGY: {strategy}\n{'='*60}")
        results = retrieve(query, strategy=strategy)
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
            print(f"Source: {r['source']} | Page: {r['page']}")
            print(r["text"][:300])