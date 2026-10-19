from sentence_transformers import CrossEncoder

_reranker = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading reranker model...")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def rerank(query, chunks, top_k=5):
    """chunks: list of chunk dicts (must have 'text'). Returns top_k reranked."""
    if not chunks:
        return []

    model = get_reranker()
    pairs = [[query, c["text"]] for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]

if __name__ == "__main__":
    from app.hybrid_search import hybrid_retrieve

    query = input("Enter a test query: ")
    candidates = hybrid_retrieve(query, top_k=20, candidate_k=20)

    print(f"\n{'='*60}\nBEFORE RERANKING (hybrid top 5 of 20)\n{'='*60}")
    for i, c in enumerate(candidates[:5], 1):
        print(f"\n--- {i} (fused: {c['fused_score']}) ---")
        print(f"Source: {c['source']} | Page: {c['page']}")
        print(c["text"][:200])

    reranked = rerank(query, candidates, top_k=5)
    print(f"\n{'='*60}\nAFTER RERANKING (cross-encoder top 5)\n{'='*60}")
    for i, c in enumerate(reranked, 1):
        print(f"\n--- {i} (rerank: {c['rerank_score']:.4f}) ---")
        print(f"Source: {c['source']} | Page: {c['page']}")
        print(c["text"][:200])