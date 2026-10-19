import sys

from app.hybrid_search import hybrid_retrieve
from app.reranker import rerank

query = " ".join(sys.argv[1:])
pool = hybrid_retrieve(query, top_k=20, candidate_k=20)

for i, c in enumerate(rerank(query, [dict(c) for c in pool], top_k=5), 1):
    print(f"\n--- [{i}] rerank {c['rerank_score']:.2f} | {c['source']} p{c['page']} ---")
    print(c["text"])