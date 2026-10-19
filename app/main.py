from fastapi import FastAPI
from pydantic import BaseModel

from app.hybrid_search import hybrid_retrieve
from app.reranker import rerank
from app.llm import generate_answer, validate_citations, extract_cited_ids

app = FastAPI(title="RAG Knowledge Assistant")

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    warnings: list[str] = []

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Knowledge Assistant is running"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    candidates = hybrid_retrieve(request.query, top_k=20, candidate_k=20)
    chunks = rerank(request.query, candidates, top_k=5)
    answer = generate_answer(request.query, chunks)

    warnings = validate_citations(answer, len(chunks))
    cited_ids = set(extract_cited_ids(answer))

    sources = [
        {
            "id": i + 1,
            "source": c["source"],
            "page": c["page"],
            "label": f"Source: {c['source']}, Page: {c['page']}",
            "score": round(c["rerank_score"], 4),
        }
        for i, c in enumerate(chunks)
        if not cited_ids or (i + 1) in cited_ids
    ]

    return AskResponse(answer=answer, sources=sources, warnings=warnings)