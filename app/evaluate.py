import json
import re
import sys

from app.config import BASE_DIR
from app.retrieval import retrieve as vector_retrieve
from app.hybrid_search import hybrid_retrieve, load_bm25_index
from app.reranker import rerank
from app.llm import generate_answer, validate_citations

K = 5
POOL = 20
EVAL_PATH = BASE_DIR / "eval_set.json"

def norm(text):
    """Lowercase and strip all whitespace so PDF spacing quirks don't break matching."""
    return re.sub(r"\s+", "", text).lower()

def is_hit(text, expected):
    t = norm(text)
    return any(norm(e) in t for e in expected)

def first_hit_rank(chunks, expected):
    for rank, chunk in enumerate(chunks, 1):
        if is_hit(chunk["text"], expected):
            return rank
    return None

def hit_rate(ranks):
    return sum(1 for r in ranks if r) / len(ranks)

def mrr(ranks):
    return sum(1 / r for r in ranks if r) / len(ranks)

def fmt(rank):
    return str(rank) if rank else "-"

def evaluate(with_llm=False):
    with open(EVAL_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    _, corpus = load_bm25_index()

    ranks = {"vector": [], "hybrid": [], "vector+rerank": [], "hybrid+rerank": []}
    pool_hits = []
    llm_results = []
    skipped = []

    print(f"\n{'question':<62} {'vec':<4} {'hyb':<4} {'rr':<4} pool")
    for case in cases:
        q, expected = case["question"], case["expected"]

        # Guard: skip test cases whose expected phrase isn't in the corpus at all
        if not any(is_hit(c["text"], expected) for c in corpus):
            skipped.append(q)
            continue

        vec = vector_retrieve(q, top_k=K)
        pool = hybrid_retrieve(q, top_k=POOL, candidate_k=POOL)
        hyb = pool[:K]
        rer = rerank(q, [dict(c) for c in pool], top_k=K)
        
        vec_pool = vector_retrieve(q, top_k=POOL)
        vec_rer = rerank(q, [dict(c) for c in vec_pool], top_k=K)
        ranks["vector+rerank"].append(first_hit_rank(vec_rer, expected))
        r_vec = first_hit_rank(vec, expected)
        r_hyb = first_hit_rank(hyb, expected)
        r_rer = first_hit_rank(rer, expected)
        in_pool = first_hit_rank(pool, expected) is not None

        ranks["vector"].append(r_vec)
        ranks["hybrid"].append(r_hyb)
        ranks["hybrid+rerank"].append(r_rer)
        pool_hits.append(in_pool)

        print(f"{q[:60]:<62} {fmt(r_vec):<4} {fmt(r_hyb):<4} {fmt(r_rer):<4} {'Y' if in_pool else 'N'}")

        if with_llm:
            answer = generate_answer(q, rer)
            warnings = validate_citations(answer, len(rer))
            answer_expected = case.get("answer_expected", expected)
            answer_ok = any(norm(e) in norm(answer) for e in answer_expected)
            llm_results.append((not warnings, answer_ok))
            print(f"    warnings: {warnings or 'none'} | has expected phrase: {answer_ok}")
            print(f"    answer: {answer[:300]!r}")

    n = len(pool_hits)
    if n == 0:
        print("No usable test cases.")
        return

    print(f"\nEvaluated {n} questions ({len(skipped)} skipped)")
    for name, rs in ranks.items():
        print(f"{name:<15} hit@{K}: {hit_rate(rs):.2f}   MRR: {mrr(rs):.2f}")
    print(f"answer in hybrid top-{POOL} pool: {sum(pool_hits)}/{n}")

    if with_llm:
        clean = sum(1 for c, _ in llm_results if c)
        correct = sum(1 for _, a in llm_results if a)
        print(f"citations valid: {clean}/{n}   answer contains expected phrase: {correct}/{n}")

    for q in skipped:
        print(f"SKIPPED (expected phrase not in any chunk): {q}")

if __name__ == "__main__":
    evaluate(with_llm="--llm" in sys.argv)