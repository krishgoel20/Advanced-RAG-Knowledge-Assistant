# Advanced RAG Knowledge Assistant 🧠
 
A question-answering API over PDF documents, built beyond the basic "PDF chatbot": hybrid retrieval, cross-encoder reranking, page-level citations, and an evaluation harness that measured what each stage actually contributed.
 
Test corpus: Tesla's FY2022 Form 10-K and FY2023 annual report (the PDFs are not included in this repo).
 
---
 
## What it does
 
**For users asking questions:**
- Ask a question in plain English about the loaded PDFs — *"What was Tesla's total revenue in 2023?"*
- Get a short answer grounded only in the retrieved excerpts, with inline citations like `[1][2]`
- See exactly where each citation came from: file name and page number
- Get `warnings` back if the answer cites a source that doesn't exist or has no citations at all
**For developers and maintainers:**
- Ingest PDFs with three interchangeable chunking strategies (fixed, recursive, semantic), each stored in its own collection for side-by-side comparison
- Run an evaluation script that scores vector, hybrid, and reranked retrieval on hit@5 and MRR
- Inspect the exact chunks the LLM received for any query with a debug tool
---
 
## What makes it different
 
Most "chat with your PDF" demos embed fixed-size chunks, take the top-k by vector similarity, and trust the answer. This project replaces each of those assumptions with something measured:
 
| Stage | Basic PDF chatbot | This project |
|---|---|---|
| **Chunking** | One fixed strategy, never compared | Fixed, recursive and semantic implemented, then compared; chunk size tested at 500 vs 1000 |
| **Retrieval** | Vector similarity only | Vector search + BM25 keyword search, fused with Reciprocal Rank Fusion |
| **Ranking** | Order of embedding similarity | Cross-encoder reranks 20 candidates down to 5 |
| **Citations** | None, or the model names sources freely | Numbered excerpts, `[n]` markers, file + page per source, only cited sources returned |
| **Citation checking** | None | Output normalization plus validation, with `warnings` in the response |
| **Evaluation** | Eyeballing a few answers | hit@5, MRR, pool recall, and an ablation of every stage |
| **Debugging** | Guesswork | Tool that prints the exact chunks the LLM saw |
 
The honest headline result: reranking drives most of the ranking quality, and BM25 did not change the final top 5 on this evaluation. See [Evaluation](#evaluation-and-findings).
 
---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| PDF parsing | `pypdf` (page-aware extraction) |
| Chunking | Custom fixed-size, `langchain-text-splitters` (recursive), custom semantic |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector store | Qdrant (local on-disk mode) |
| Keyword search | `rank-bm25` (BM25Okapi, in-memory) |
| Fusion | Reciprocal Rank Fusion (k = 60) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| Config | `python-dotenv` |
 
---
 
## How it works
 
**Ingestion (run once per strategy):**
 
```
PDF files in data/
        ↓
pypdf extracts text page by page (page numbers kept, 1-indexed)
        ↓
Whitespace collapsed (some PDFs extract one token per line)
        ↓
Chunking: fixed | recursive | semantic  (default: fixed, 1000 chars / 100 overlap)
        ↓
Each chunk keeps its source file and page number
        ↓
all-MiniLM-L6-v2 embeds chunks in batches
        ↓
Points written to a Qdrant collection, one collection per strategy
        ↓
Stored point count checked against chunks written (warns on stale data)
```
 
**Answering a question (`POST /ask`):**
 
```
User question
        ↓
Vector search → top 20 chunks from Qdrant
Keyword search → top 20 chunks from the in-memory BM25 index
        ↓
Reciprocal Rank Fusion merges both rankings → 20 candidates
        ↓
Cross-encoder scores each (question, chunk) pair → top 5 kept
        ↓
Prompt built with numbered excerpts [1]..[5], each tagged with file and page
        ↓
LLM answers using only those excerpts, citing [n] after each claim
        ↓
Output normalized (fullwidth brackets 【1】 → [1], odd unicode spaces cleaned)
        ↓
Citation validator checks markers exist and refer to real excerpts
        ↓
Response: answer + cited sources (id, file, page, label, score) + warnings
```
 
---
 
## Project Structure
 
```
RAG-Knowledge-Assistant/
├── app/
│   ├── __init__.py
│   ├── config.py             # Paths, model names, chunk settings
│   ├── ingest.py             # PDF → chunks → embeddings → Qdrant (fixed | recursive | semantic)
│   ├── retrieval.py          # Vector search (Qdrant client singleton)
│   ├── hybrid_search.py      # BM25 index + Reciprocal Rank Fusion
│   ├── reranker.py           # Cross-encoder reranking
│   ├── llm.py                # Prompt building, Groq call, citation normalization + validation
│   ├── main.py               # FastAPI app (/ask)
│   ├── evaluate.py           # Retrieval + answer evaluation
│   └── debug_query.py        # Print the exact chunks the LLM receives
├── data/                     # PDFs go here (git-ignored)
├── qdrant_data/              # Qdrant on-disk storage (git-ignored)
├── eval_set.json             # Evaluation questions with expected phrases
├── requirements.txt
├── .env                      # GROQ_API_KEY (git-ignored)
├── .gitignore
└── README.md
```
 
---
 
## Data Model
 
There is no relational database. Chunks live in Qdrant, and a BM25 index is rebuilt in memory from them at startup.
 
```
Qdrant collection:  rag_knowledge_assistant_{strategy}      (fixed | recursive | semantic)
vectors:            384 dimensions, cosine distance (all-MiniLM-L6-v2)
point id:           integer, assigned sequentially during ingestion
payload:            text    — the chunk text
                    source  — PDF file name
                    page    — 1-indexed page number
 
BM25 index:         built by scrolling every point out of the collection once,
                    tokenizing on word characters, held in memory
```
 
---
 
## Setup and Installation
 
### Pre-requisites
- Python 3.11+
- Groq API key ([console.groq.com](https://console.groq.com))
- One or more PDFs to query
### 1. Clone the repository
 
```bash
git clone https://github.com/krishgoel20/RAG-Knowledge-Assistant.git
cd RAG-Knowledge-Assistant
```
 
### 2. Set up the environment
 
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
 
`requirements.txt` contains: fastapi, uvicorn, python-dotenv, pypdf, qdrant-client, sentence-transformers, groq, langchain-text-splitters, rank-bm25.
 
The first install is large because `sentence-transformers` pulls in PyTorch (CPU is enough).
 
### 3. Configure environment variables
 
Create a `.env` file in the project root:
 
```
GROQ_API_KEY=your_groq_api_key
```
 
Groq retires model IDs from time to time. If you get a `model_not_found` error, update `GROQ_MODEL` in `app/config.py` to a model your key can access.
 
### 4. Add documents and ingest
 
Put your PDFs in `data/`, then build the index:
 
```bash
python -m app.ingest fixed
```
 
Also available: `recursive` and `semantic` (each builds its own collection). `fixed` is the default used by the API.
 
### 5. Run the API
 
```bash
uvicorn app.main:app --reload
```
 
API runs at `http://127.0.0.1:8000`. Interactive docs are at `/docs`.
 
Qdrant's local mode allows only one client per storage folder, so stop the server before running ingestion or evaluation.
 
### 6. Sample questions to try
 
These were checked against the Tesla filings used for testing:
 
| Question | Answer found |
|---|---|
| What was Tesla's total revenue in 2023? | $96.77 billion ($96,773 million) |
| What was Tesla's automotive sales revenue in 2023? | $78,509 million |
| How did automotive leasing revenue change in 2023? | Fell $356 million (14%) |
| How much unrecognized stock-based compensation expense did Tesla have under performance-based grants as of December 31, 2023? | $655 million |
| What was total stock-based compensation expense in 2022? | $1,560 million |
| Who is Tesla's independent registered public accounting firm? | PricewaterhouseCoopers LLP |
 
---
 
## Features
 
### 🔎 Retrieval Features
- ✂️ **Three chunking strategies** — Fixed-size with overlap, recursive (paragraph → sentence → word), and semantic (split where sentence similarity drops), each in its own Qdrant collection
- 📄 **Page-aware ingestion** — Text is extracted page by page, so every chunk carries its file name and page number from the start
- 🔀 **Hybrid search** — Vector search and BM25 keyword search combined with Reciprocal Rank Fusion
- 🎯 **Cross-encoder reranking** — Retrieves 20 candidates, rescores each against the question, keeps the best 5
### 📑 Answer Features
- 🧾 **Inline citations** — The LLM sees numbered excerpts and cites `[n]` after each claim
- 📍 **Source labels** — Each cited source returns its file name, page number, and rerank score
- 🧹 **Output normalization** — Fullwidth brackets and stray unicode spaces in model output are cleaned before validation
- 🛡️ **Citation validation** — Flags answers with no citations or citations to excerpts that don't exist, returned as `warnings`
- 🚫 **Grounded-only prompting** — The model is told to answer only from the excerpts and to say so when they don't contain the answer
### 🧪 Developer Features
- 📊 **Evaluation harness** — hit@5, MRR, and pool recall for vector, hybrid, vector + rerank, and hybrid + rerank, plus optional answer and citation checks
- 🐞 **Debug tool** — Prints the exact top-5 chunks the LLM received for any query
- 🔒 **Stale-data guard** — Ingestion wipes old points explicitly and warns if the stored count doesn't match what it wrote
- ⚡ **Lazy model loading** — Embedding model, reranker, and Qdrant client load once on first use and are reused
---
 
## API Endpoints
 
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/ask` | Ask a question; body is `{"query": "..."}` |
| GET | `/docs` | Interactive API documentation (auto-generated by FastAPI) |
 
**`POST /ask` response**
 
```json
{
  "answer": "Tesla's total revenue for 2023 was $96.77 billion [1][2].",
  "sources": [
    {"id": 1, "source": "tm2412112d4_ars.pdf", "page": 137,
     "label": "Source: tm2412112d4_ars.pdf, Page: 137", "score": 9.31},
    {"id": 2, "source": "tm2412112d4_ars.pdf", "page": 497,
     "label": "Source: tm2412112d4_ars.pdf, Page: 497", "score": 8.41}
  ],
  "warnings": []
}
```
 
The example is illustrative; page numbers and scores depend on your chunk settings. Only sources the answer actually cites are returned, and their ids match the `[n]` markers. If the answer contains no citations, all five retrieved sources are returned and `warnings` explains why.
 
**Command-line tools**
 
| Command | Description |
|---|---|
| `python -m app.ingest <fixed\|recursive\|semantic>` | Build a collection from the PDFs in `data/` |
| `python -m app.evaluate` | Retrieval evaluation over `eval_set.json` |
| `python -m app.evaluate --llm` | Also generate answers and check citations |
| `python -m app.debug_query "your question"` | Show the exact chunks sent to the LLM |
| `python -m app.hybrid_search` | Compare vector-only and hybrid results for a query |
| `python -m app.reranker` | Compare hybrid results before and after reranking |
 
---
 
## Evaluation and Findings

`python -m app.evaluate` runs the questions in `eval_set.json` through each retriever. A retriever scores a hit when a top-5 chunk contains an expected phrase (matching ignores case and whitespace).
 
- **hit@5** — share of questions with an answer-bearing chunk in the top 5
- **MRR** — mean of 1/rank of the first answer-bearing chunk
- **Pool recall** — whether the answer is anywhere in the 20 candidates given to the reranker
With `--llm`, it also reports citation validity and whether the answer contains the expected phrase (a rough string match).

### Results

**Retrieval stages (15 questions, 1000-char chunks)**
 
| Retriever | hit@5 | MRR |
|---|---|---|
| Vector only | 0.80 | 0.55 |
| Hybrid (BM25 + vector, RRF) | 0.93 | 0.55 |
| Vector + rerank | 1.00 | 0.89 |
| Hybrid + rerank | 1.00 | 0.89 |
 
**Chunk size (11 questions)**
 
| | 500 chars | 1000 chars |
|---|---|---|
| Vector hit@5 / MRR | 0.82 / 0.64 | 0.82 / 0.56 |
| Hybrid hit@5 / MRR | 1.00 / 0.65 | 0.91 / 0.50 |
| Hybrid + rerank hit@5 / MRR | 0.91 / 0.85 | 1.00 / 0.89 |
| Answer contains expected phrase | 10/11 | 11/11 |
| Citations valid | 11/11 | 11/11 |

### Findings

1. **Semantic chunking underperformed on dense financial text.** On one test query (measured on the original extraction at 500 chars), semantic chunking produced 10,866 chunks against about 6,700 for fixed and recursive, and its top results were isolated one-sentence fragments unrelated to the question. 10-K prose is full of short, self-contained sentences, so splitting on topic shifts over-fragments it. This comparison was qualitative and was not re-run with the evaluation harness. Fixed-size chunking is the default.
2. **Larger chunks hurt first-stage ranking but helped final answers.** Moving from 500 to 1000 characters lowered vector and hybrid MRR, but with reranking the final answers improved. A question about total 2022 stock-based compensation only started working once a table and its row labels landed in the same chunk.
3. **Reranking drives most of the ranking quality.** MRR rises from 0.55 to 0.89 once the cross-encoder re-sorts the candidates. One observed weakness: on a general "stock-based compensation" query it preferred explanatory passages over a chunk containing the specific dollar figure.
4. **Hybrid search raises recall but did not change the reranked top 5.** Before reranking, hybrid improved hit@5 from 0.80 to 0.93, and it surfaced exact-phrase chunks that vector search missed on queries like "Form 10-K". After reranking, vector + rerank and hybrid + rerank were identical on this evaluation. The corpus is small, so the vector top 20 already contained every answer. Hybrid may matter more on a larger corpus, but this evaluation does not show it. On a numeric lookup ("total revenue in 2023"), BM25 also pushed the right chunk down before reranking by favoring passages that repeat "revenue".
5. **Citation validity went from 4/11 to 11/11 by fixing output handling, not retrieval.** The model sometimes wrote fullwidth brackets (【1】) that the validator did not parse. Normalizing the output and tightening the prompt fixed it.
6. **Whitespace cleaning was roughly neutral.** One PDF extracts one token per line, so text is collapsed before chunking. Chunk count dropped only about 1% and retrieval metrics did not change meaningfully.
---

## Key Concepts Demonstrated
 
- **RAG (Retrieval-Augmented Generation)** — Answers are generated only from retrieved excerpts, and the prompt tells the model to say so when the excerpts don't contain the answer, rather than falling back on training knowledge.
- **Chunking strategies** — Fixed-size windows with overlap, recursive splitting on natural boundaries, and semantic splitting by sentence-embedding similarity, with measured trade-offs between them.
- **Hybrid search with Reciprocal Rank Fusion** — BM25 and vector scores live on different scales, so results are merged by rank: each chunk scores `1 / (k + rank)` per list, summed across lists. Only position matters, not raw score.
- **Bi-encoder vs cross-encoder** — The embedding model encodes query and chunk separately, which is fast and indexable. The cross-encoder reads them together, which is more accurate but can't be precomputed, so it only runs on a small candidate set.
- **Citation grounding** — Excerpts are numbered in the prompt so the model cites by number. The backend normalizes the output, validates the markers, and maps each one back to a file and page.
- **Evaluation and ablation** — hit@5, MRR, and pool recall separate retrieval failures from ranking failures, and comparing vector + rerank with hybrid + rerank isolates what BM25 adds.
- **Data hygiene** — Whitespace cleaning before chunking, and a stored-versus-written count check that caught stale points left behind by `delete_collection` in Qdrant's local mode.
---

## Limitations
 
- **No frontend** — The project is an API with auto-generated docs at `/docs`; there is no chat UI.
- **No cloud deployment** — Everything runs locally. Qdrant's local mode allows one client per storage folder, so the API server and the ingestion or evaluation scripts can't run at the same time.
- **PDF text only** — Extraction uses `pypdf`. Scanned PDFs without a text layer are not supported (no OCR), and tables lose their structure, which is why chunk size mattered.
- **Ingestion is not incremental** — Each ingest run rebuilds the collection, and the BM25 index is rebuilt in memory at startup.
- **Small evaluation set** — 15 questions, with expected phrases taken from chunks seen during development, which biases the set toward answerable questions. Differences of one question are noise.
- **Citation validation checks format only** — It confirms markers exist and refer to real excerpts, not that each claim is supported by the excerpt it cites.
- **No document or year awareness** — A generic question can pull figures from the older filing even when the newer one is loaded (observed on a stock-compensation summary).
- **Near-duplicate chunks across filings** use up top-5 slots.
- **Embedding truncation** — `all-MiniLM-L6-v2` embeds at most 256 tokens per chunk, so number-dense 1000-character chunks may be partly cut off in vector search. BM25 and the reranker see the full text. This was not measured here.
- **LLM responses are non-deterministic** — The same question can produce slightly different wording between runs, and Groq model IDs get retired over time.
- **Single-turn only** — No conversation memory or follow-up questions.
- **Older collections** — The `recursive` and `semantic` collections were built on an earlier extraction and were not re-evaluated.
---

## Possible Next Steps
 
Metadata filtering by document or year, near-duplicate removal, table-aware extraction, a numeric grounding check that verifies figures in the answer appear in the cited chunks, an LLM-judged faithfulness metric, a larger evaluation set, and Qdrant in Docker for concurrent access.