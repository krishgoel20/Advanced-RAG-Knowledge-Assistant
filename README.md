# PageLight 🧠
 
A question-answering API over PDF documents, built in stages from a basic RAG pipeline into something closer to a real application: better chunking, hybrid search, cross-encoder reranking, page-level citations, and an evaluation harness. Each stage was measured before moving on, including the ones that did not help.
 
Test corpus: Tesla's FY2022 Form 10-K and FY2023 annual report (the PDFs are not included in this repo).
 
---
 
## What it does

**For users asking questions:**
- Ask a question in plain English about the indexed PDFs, for example *"What was Tesla's total revenue in 2023?"*
- Get an answer grounded only in the retrieved excerpts, with inline citations like `[1][2]`
- See exactly where each claim came from: **source file and page number**
- Get a clear "the excerpts do not contain this" response instead of an invented answer when the documents lack the information
- Get `warnings` in the response when citations look malformed (missing, or pointing at a source that does not exist)

**For developers and evaluators:**
- Switch between three chunking strategies (fixed, recursive, semantic) and compare them
- Run an evaluation script that scores retrieval (hit@5, MRR, pool recall) across four retrieval setups and, optionally, answer quality and citation validity
- Inspect the exact chunks the LLM received for any query with a debug script

---
 
## What makes it different
 
Most "chat with your PDF" demos embed fixed-size chunks, run a single vector search, and return plain text. This project adds a measured layer on top of that baseline:
 
| Feature | Basic PDF chatbot | PageLight |
|---|---|---|
| **Chunking** | One fixed-size splitter, never compared | Fixed, recursive and semantic chunking built and compared; fixed-size chosen from results |
| **Retrieval** | Vector search only | BM25 keyword search + vector search fused with Reciprocal Rank Fusion |
| **Ranking** | Raw embedding similarity | Cross-encoder reranks 20 candidates down to 5 |
| **Answers** | Plain text, no sources | Inline `[n]` citations tied to file and page; only cited sources returned |
| **Trust** | Take the LLM's word | Citation validation, with warnings returned in the response |
| **Quality claims** | "It seems to work" | Evaluation harness with hit@5, MRR, pool recall, and a retrieval ablation |
| **Negative results** | Rarely reported | Documented, for ex. BM25 adding nothing once a reranker is in place |
 
None of the individual techniques here is new. What distinguishes the project is that each one was measured, and that the results (including the ties and failures) are reported below.
 
---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| PDF extraction | `pypdf` (page-aware) |
| Chunking | Custom fixed-size, LangChain `RecursiveCharacterTextSplitter`, embedding-based semantic splitting |
| Embeddings | `sentence-transformers`: `all-MiniLM-L6-v2` (384-dim) |
| Vector store | Qdrant (local on-disk mode) |
| Keyword search | `rank-bm25` (BM25Okapi, in-memory) |
| Fusion | Reciprocal Rank Fusion (k = 60) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| Config | `python-dotenv` |
 
---
 
## How it works

```
Ingestion (python -m app.ingest <strategy>)
        ↓
PDFs read page by page (page number kept with every chunk)
        ↓
Whitespace collapsed (some PDFs extract one token per line)
        ↓
Chunking: fixed (1000 chars, 100 overlap) | recursive | semantic
        ↓
Chunks embedded with all-MiniLM-L6-v2
        ↓
Vectors + payload {text, source, page} stored in Qdrant
        ↓
Stored count checked against written count (warns on stale data)
 
Query time (POST /ask)
        ↓
Query embedded → vector search in Qdrant (top 20)
        ↓
Same query → BM25 keyword search over all chunks (top 20)
        ↓
Reciprocal Rank Fusion merges both rankings → top 20 candidates
        ↓
Cross-encoder scores each (query, chunk) pair → top 5
        ↓
Top 5 numbered as excerpts [1]-[5] and sent to the LLM
        ↓
LLM answers using only those excerpts, citing [n] after each claim
        ↓
Output normalised (bracket styles, odd unicode spaces)
        ↓
Citations validated (markers present, numbers in range)
        ↓
Response: answer + only the cited sources (file, page, label, score) + warnings
```

---
 
## Project Structure
 
```
Advanced-RAG-Knowledge-Assistant/
├── app/
│   ├── __init__.py
│   ├── config.py            # Paths, model names, chunk settings, collection naming
│   ├── ingest.py            # PDF → chunks → embeddings → Qdrant (fixed | recursive | semantic)
│   ├── retrieval.py         # Vector search (shared Qdrant client)
│   ├── hybrid_search.py     # BM25 index + Reciprocal Rank Fusion
│   ├── reranker.py          # Cross-encoder reranking
│   ├── llm.py               # Prompt building, Groq call, citation normalisation + validation
│   ├── main.py              # FastAPI app (/ and /ask)
│   ├── evaluate.py          # Retrieval + answer evaluation
│   └── debug_query.py       # Print the exact chunks the LLM receives for a query
├── data/                    # Source PDFs (included): Tesla FY2022 10-K, FY2023 annual report
├── qdrant_data/             # Qdrant on-disk storage (not committed)
├── eval_set.json            # Evaluation questions with expected phrases
├── requirements.txt
├── .env                     # GROQ_API_KEY (not committed)
└── README.md
```
 
---

## Data Model

**Qdrant point (one per chunk)**

```
id        — integer, unique per chunk
vector    — 384 floats, cosine distance
payload   — text (chunk content), source (PDF file name), page (1-indexed)
```

Collections, one per chunking strategy: `rag_knowledge_assistant_fixed` (default), `rag_knowledge_assistant_recursive`, `rag_knowledge_assistant_semantic`.

**Retrieved chunk (in memory)**

```
id, text, source, page
score          — cosine similarity (vector search)
fused_score    — Reciprocal Rank Fusion score (hybrid search)
rerank_score   — cross-encoder relevance score (after reranking)
```

**API models**

```
AskRequest    — query: str
AskResponse   — answer: str
                sources: list of {id, source, page, label, score}
                warnings: list of str
```

**Evaluation case (`eval_set.json`)**

```
question          — the question to ask
expected          — phrases; a hit means a top-5 chunk contains one of them
answer_expected   — optional phrases checked in the generated answer instead
```

---
 
## Setup and Installation
 
### Pre-requisites
- Python 3.11+
- Groq API key ([console.groq.com](https://console.groq.com))
- One or more PDFs to index

### 1. Clone the repository
 
```bash
git clone https://github.com/krishgoel20/Advanced-RAG-Knowledge-Assistant.git
cd Advanced-RAG-Knowledge-Assistant
```
 
### 2. Set up the environment
 
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
 
`requirements.txt` contains: fastapi, uvicorn, python-dotenv, pypdf, qdrant-client, sentence-transformers, groq, langchain-text-splitters, rank-bm25.
 
The first run downloads the embedding and reranker models from Hugging Face. A Hugging Face token is optional and only raises download rate limits.
 
### 3. Configure environment variables
 
Create a `.env` file in the project root:
 
```
GROQ_API_KEY=your_groq_api_key
```

### 4. Ingest the documents

The two Tesla PDFs are already in `data/` (see [Data](#data)), so you can build the index straight away:

```bash
python -m app.ingest fixed
```

`fixed` is the default used by the API. You can also run `python -m app.ingest recursive` and `python -m app.ingest semantic` to build the other collections. Ingestion prints the chunk count per PDF and warns if the stored count does not match what was written.

To index your own documents instead, add PDFs to `data/` and run the same command. The included evaluation set only applies to the Tesla PDFs.
 
### 5. Run the API
 
```bash
uvicorn app.main:app --reload
```

API runs at `http://127.0.0.1:8000`. Interactive docs are at `/docs`: open `POST /ask` and send `{"query": "What was Tesla's total revenue in 2023?"}`.

Qdrant's local mode allows only one client per storage folder, so stop the server before running ingestion or evaluation.

### 6. Run the evaluation
 
Stop the server first (Qdrant's local mode allows one client per storage folder), then:
 
```bash
python -m app.evaluate          # retrieval only, no LLM calls
python -m app.evaluate --llm    # also checks answers and citations
```

### 7. Sample questions to try
 
These were checked against the Tesla filings used for testing:

| Question | What it tests | Answer found |
|---|---|---|
| What was Tesla's total revenue in 2023? | Single numeric fact | $96.77 billion ($96,773 million) |
| What was Tesla's automotive sales revenue in 2023? | Line item from a financial statement | $78,509 million |
| How did automotive leasing revenue change in 2023? | Year-over-year change stated in prose | Fell $356 million (14%) |
| How much unrecognized stock-based compensation expense did Tesla have under performance-based grants as of December 31, 2023? | Specific figure inside a long passage | $655 million |
| What was total stock-based compensation expense in 2022? | A table whose labels and numbers must stay together | $1,560 million |
| Who is Tesla's independent registered public accounting firm? | Named entity lookup | PricewaterhouseCoopers LLP |

---

## Data

The repo ships with two public Tesla filings in `data/`, so ingestion and the evaluation set work out of the box.

| File | Document | Fiscal year |
|---|---|---|
| `tsla-20221231-gen.pdf` | Tesla Form 10-K (annual report filed with the SEC) | FY2022 |
| `tm2412112d4_ars.pdf` | Tesla Annual Report to Shareholders (proxy statement plus Form 10-K) | FY2023 |

Both are public SEC filings, available from [Tesla Investor Relations](https://ir.tesla.com) and [SEC EDGAR](https://www.sec.gov/edgar). Having two fiscal years lets the assistant answer year-over-year questions across documents.

`eval_set.json` and its expected answers refer to these two files.

---

## Features
 
### 🔎 Retrieval Features
- ✂️🧩 **Three chunking strategies** — Fixed-size with overlap, recursive (paragraph → sentence → word), and semantic (new chunk where consecutive-sentence similarity drops)
- 🔀 **Hybrid search** — BM25 keyword matching and Vector similarity combined with Reciprocal Rank Fusion
- 🎯 **Cross-encoder reranking** — Retrieves 20 candidates cheaply, then re-scores each query/chunk pair together, and keeps the best 5
- 📄 **Page-aware ingestion** — Every chunk carries its source file name and page number from the moment it is created

### 📑 Answer Features
- 🧾 **Inline citations** — The LLM cites `[n]` for numbered excerpts, and the response maps each number to a file and page
- 📍 **Source labels** — Each cited source returns its file name, page number, and rerank score
- 🧹 **Output normalization** — Fullwidth brackets and stray unicode spaces in model output are cleaned before validation
- 🛡️ **Citation validation** — Flags answers with no citations or citations to excerpts that don't exist, returned as `warnings`
- 🚫 **Grounded-only prompting** — The model is told to answer only from the excerpts and to say so when they don't contain the answer

### 🧪 Engineering Features
- 📊 **Evaluation harness** — hit@5, MRR, and pool recall across vector, hybrid, vector + rerank, and hybrid + rerank
- 🔬 **Debug script** — Shows the exact chunks the LLM received, which separates retrieval failures from generation failures
- 🔒 **Stale-index guard** — Ingestion compares the stored point count with what it wrote and warns on a mismatch

---
 
## API Endpoints
 
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/ask` | Answer a question with citations |
| GET | `/docs` | Interactive Swagger UI / API documentation (auto-generated by FastAPI) |
| GET | `/redoc` | Alternative API documentation, auto-generated by FastAPI |
| GET | `/openapi.json` | OpenAPI schema |
 
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
 
## Evaluation

`python -m app.evaluate` runs the questions in `eval_set.json` through each retriever. A retriever scores a hit when a top-5 chunk contains an expected phrase (matching ignores case and whitespace).
 
- **hit@5** — share of questions with an answer-bearing chunk in the top 5
- **MRR** — mean of 1/rank of the first answer-bearing chunk, so it rewards ranking the answer first
- **Pool recall** — whether the answer is anywhere in the 20 candidates given to the reranker

With `--llm`, it also reports whether citations are valid and whether the answer contains the expected phrase (a rough string match).

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
