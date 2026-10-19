import re
from rank_bm25 import BM25Okapi

from app.config import get_collection_name
from app.retrieval import retrieve as vector_retrieve, get_client

_bm25_index = None
_bm25_corpus = None  # list of chunk dicts, same order as the BM25 index

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def load_bm25_index(strategy="fixed"):
    """Scrolls every point out of Qdrant once and builds an in-memory BM25 index."""
    global _bm25_index, _bm25_corpus

    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus

    print("Building BM25 index (one-time)...")
    client = get_client()
    collection_name = get_collection_name(strategy)

    corpus = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        for p in points:
            corpus.append({
                "id": p.id,
                "text": p.payload["text"],
                "source": p.payload["source"],
                "page": p.payload["page"],
            })
        if offset is None:
            break

    tokenized_corpus = [tokenize(c["text"]) for c in corpus]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_corpus = corpus
    print(f"BM25 index built on {len(corpus)} chunks.")
    return _bm25_index, _bm25_corpus

def bm25_search(query, strategy="fixed", top_k=20):
    index, corpus = load_bm25_index(strategy)
    scores = index.get_scores(tokenize(query))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [corpus[i] for i in ranked]

def reciprocal_rank_fusion(result_lists, k=60):
    """
    result_lists: list of ranked lists (each a list of chunk dicts with 'id'),
    already ordered best-to-worst. Returns chunk dicts sorted by fused score.
    """
    fused_scores = {}
    chunk_lookup = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            chunk_id = chunk["id"]
            chunk_lookup[chunk_id] = chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [
        {**chunk_lookup[cid], "fused_score": round(fused_scores[cid], 5)}
        for cid in ranked_ids
    ]

def hybrid_retrieve(query, strategy="fixed", top_k=5, candidate_k=20):
    vector_results = vector_retrieve(query, strategy=strategy, top_k=candidate_k)
    keyword_results = bm25_search(query, strategy=strategy, top_k=candidate_k)

    fused = reciprocal_rank_fusion([vector_results, keyword_results])
    return fused[:top_k]

if __name__ == "__main__":
    query = input("Enter a test query: ")

    print(f"\n{'='*60}\nVECTOR-ONLY\n{'='*60}")
    for i, r in enumerate(vector_retrieve(query, top_k=5), 1):
        print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"Source: {r['source']} | Page: {r['page']}")
        print(r["text"][:300])

    print(f"\n{'='*60}\nHYBRID (BM25 + Vector, RRF fused)\n{'='*60}")
    for i, r in enumerate(hybrid_retrieve(query, top_k=5), 1):
        print(f"\n--- Result {i} (fused score: {r['fused_score']}) ---")
        print(f"Source: {r['source']} | Page: {r['page']}")
        print(r["text"][:300])