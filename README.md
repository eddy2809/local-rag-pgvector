# Local RAG over Italian documents

A retrieval-augmented generation (RAG) pipeline that runs entirely on a local machine and answers
questions about Italian PDF documents, citing the page each statement comes from.
The corpus used for development and evaluation is my Master's thesis (about 100 pages, in Italian).

The project focuses less on the happy path and more on **measuring** each design choice: chunk size,
lexical vs dense vs hybrid retrieval, BM25, and cross-encoder re-ranking, all evaluated on a
hand-checked question set.

## Key results

- With the relevant passage among the retrieved sources, **85% of answers are correct**; without it,
  only 33%. Most errors originate in retrieval, not in generation.
- Hybrid retrieval plus cross-encoder re-ranking raises **hit rate@5 from 0.77 to 0.85** and
  **MRR@10 from 0.680 to 0.715** over plain dense retrieval, for about 170 ms of extra latency.
- **4/4 correct abstentions** on questions whose answer is not in the documents; no fabricated answers.
- End-to-end latency of the baseline pipeline: **0.57 s median**, 1.81 s p95, on an RTX 4090.

## Architecture

```
                         INDEXING
PDF ──► extract (PyMuPDF) ──► chunk (300 tokens, bge-m3 tokenizer) ──► embed (bge-m3)
                                                                          │
                                                                          ▼
                                       PostgreSQL + pgvector: text, metadata, vector(1024),
                                       HNSW index (cosine) and GIN full-text index (Italian)

                         QUERYING
question ──► dense search (pgvector) ─┐
         └─► full-text search ────────┴─► Reciprocal Rank Fusion ──► top 30
                                                                       │
                                     cross-encoder re-ranking ◄────────┘
                                                │ top 5
                                                ▼
                        qwen3:8b via Ollama: answer with [n] citations, or a fixed
                        abstention sentence when the sources do not contain the answer
```

## Stack

| Component | Choice | Why |
|---|---|---|
| LLM | `qwen3:8b` via Ollama | Good Italian, runs on a single consumer GPU |
| Embeddings | `bge-m3` via Ollama | Multilingual, 1024 dimensions, 8192-token context |
| Vector store | PostgreSQL 17 + pgvector, in Docker | Vectors, metadata and full-text search in one database and one query |
| Re-ranker | `BAAI/bge-reranker-v2-m3` (optional) | Multilingual cross-encoder from the same family as bge-m3 |
| PDF parsing | PyMuPDF | Text blocks with coordinates, page labels |
| DB access | psycopg 3 + pgvector-python | Server-side parameter binding, numpy adaptation |

No LangChain or LlamaIndex: every step is plain Python and SQL, so each choice is visible and testable.

## Repository layout

```
rag/                 pipeline package (run modules with python -m rag.<module>)
  config.py, db.py   configuration from .env, connection with pgvector types registered
  extract.py         PDF text extraction and cleaning
  chunk.py           token-based chunking with overlap
  embeddings.py      bge-m3 embeddings through Ollama
  ingest.py          schema, embeddings, HNSW and GIN indexes (idempotent)
  retrieval.py       dense, full-text and hybrid (RRF in SQL) search
  bm25.py            Okapi BM25 over PostgreSQL lexemes, RRF in Python
  rerank.py          cross-encoder re-ranking
  generation.py      grounded prompt, citations, abstention
  query.py           command-line question answering
  evaluate_answers.py  end-to-end run with automatic checks and grading export
eval/                question set, evaluation and error-analysis scripts
results/             evaluation outputs (JSON, graded CSV)
docker-compose.yml   PostgreSQL + pgvector
```

## Quickstart

Prerequisites: Python 3.12 (conda), Docker, [Ollama](https://ollama.com), an NVIDIA GPU is recommended.

```bash
conda create -n rag-locale python=3.12 -y
conda activate rag-locale
pip install -r requirements.txt
cp .env.example .env              # then set the password

docker compose up -d
ollama pull qwen3:8b
ollama pull bge-m3
python -m rag.healthcheck         # checks database, pgvector and models
```

Index a PDF (place it in `data/pdf/`):

```bash
python -m rag.extract data/pdf/thesis.pdf
python -m rag.chunk data/processed/thesis.pages.jsonl --size 300 --overlap 50
python -m rag.ingest data/processed/thesis.chunks300.jsonl --table chunks_300
```

Ask a question:

```bash
python -m rag.query "Quale algoritmo viene usato per calcolare il cammino minimo?"
```

For the best configuration (hybrid retrieval + re-ranking), install PyTorch with CUDA following
[pytorch.org](https://pytorch.org), then `pip install -r requirements-rerank.txt`, and run:

```bash
python -m rag.query "..." --method hybrid_rerank
```

Evaluate:

```bash
python -m eval.evaluate_retrieval --table chunks_300 --method hybrid_rerank
python -m rag.evaluate_answers --table chunks_300
```

## Design choices

**Extraction.** The thesis was typeset with LaTeX without T1 font encoding, so accented letters were
extracted as a separate accent glyph (`realt`a`, `perch ́e`); they are rebuilt with Unicode
normalization. Running headers and page numbers are removed by detecting short blocks repeated at the
top or bottom of many pages. The table of contents and the bibliography are excluded: the former
matches every section title without containing answers. Words hyphenated across page breaks are re-joined.

**Chunking.** Fixed-size chunks of 300 tokens with 50 tokens of overlap, counted with the bge-m3
tokenizer itself. The token count was verified against Ollama: 55,745 tokens on both sides. Chunks may
cross page boundaries; each chunk stores its page range for citations.

**Storage.** One table per chunking configuration rather than one table with a filter column: with an
HNSW index, filters are applied after the approximate search and can return fewer results than
requested. The HNSW index uses `vector_cosine_ops`, matching the `<=>` operator used in queries.
Re-ingesting a document deletes and re-inserts its rows in a single transaction.

**Hybrid search.** Dense and full-text results are fused with Reciprocal Rank Fusion (k = 60), which
uses ranks only and avoids normalizing incomparable scores. The fusion runs in a single SQL query.
The full-text query ORs the question terms, since natural-language questions almost never contain all
terms of a passage. PostgreSQL's `ts_rank` has no IDF, so BM25 was also implemented, in Python, on the
same Italian lexemes produced by PostgreSQL.

**Generation.** Sources are passed as numbered, delimited blocks; the model cites ids such as `[2]` and
the code maps them to pages, so page numbers are never generated by the model. A fixed abstention
sentence makes abstentions measurable. Thinking mode is disabled, temperature is 0.1 and the seed is
fixed, so that runs are reproducible across experiments.

## Evaluation

**Question set**: 30 questions in Italian (13 lexical, 11 paraphrased, 2 answerable only from a table,
4 unanswerable). Each answerable question has an **evidence snippet**, a short verbatim fragment of the
text: a chunk is relevant if it contains it. Page-based relevance was avoided because larger chunks
span more pages and would be favoured regardless of content. The questions were drafted with AI
assistance and checked by the author of the document.

**Metrics**: hit rate@k (at least one relevant chunk in the top k; with a single evidence per question
this is a hit rate rather than a recall) and MRR@10. Answers were graded manually on correctness
(correct / partial / wrong) and faithfulness (every statement supported by the document).

### Retrieval (26 answerable questions, 300-token chunks unless noted)

| Configuration | hit@1 | hit@2 | hit@3 | hit@5 | MRR@10 |
|---|---|---|---|---|---|
| Dense | 0.58 | 0.69 | 0.77 | 0.77 | 0.680 |
| Dense, 800-token chunks | 0.46 | 0.65 | 0.81 | 0.88 | 0.637 |
| Full-text, `ts_rank` | 0.50 | 0.65 | 0.73 | 0.77 | 0.614 |
| Full-text, BM25 | 0.54 | 0.69 | 0.73 | 0.77 | 0.638 |
| Hybrid RRF (dense + `ts_rank`, SQL) | 0.58 | 0.77 | 0.77 | 0.81 | 0.692 |
| Hybrid RRF (dense + BM25) | 0.58 | 0.73 | 0.81 | 0.81 | 0.694 |
| Dense + re-ranking | **0.62** | **0.77** | **0.81** | **0.85** | **0.715** |
| Hybrid + re-ranking | **0.62** | **0.77** | **0.81** | **0.85** | 0.713 |

Findings:

- **Chunk size is a trade-off.** 800-token chunks reach a higher hit@5 but send about 4,000 tokens to
  the LLM instead of 1,500, and rank the exact passage lower (hit@1 0.46). At an equal context budget,
  2 chunks of 800 tokens (hit@2 0.65) do worse than 5 chunks of 300 (hit@5 0.77).
- **Dense and full-text search are complementary**: full-text wins on lexical questions (hit@5 0.92
  vs 0.77), dense wins on paraphrases (0.73 vs 0.55). RRF rewards agreement between the two: a passage
  ranked 10th by both outranks one ranked 1st by only one of them.
- **BM25 ranks better than `ts_rank` on its own**, but the two hybrids are equivalent; the simpler
  all-SQL hybrid was kept.
- **Re-ranking gives the best results** and makes the first stage almost irrelevant. Hybrid is still
  preferred as first stage because exact identifiers (codes, product names) are common in business
  documents and poorly captured by embeddings.
- With 26 questions, one question is worth about 4 points: differences of one or two questions are
  not significant.

### Answers (baseline: dense retrieval, k = 5)

| | Questions | Correct | Partial | Wrong | Faithful |
|---|---|---|---|---|---|
| Evidence among the sources | 20 | 17 | 1 | 2 | 20 |
| Evidence not retrieved | 6 | 2 | 2 | 2 | 6 |
| Unanswerable (abstention expected) | 4 | 4 | 0 | 0 | 4 |
| **Total** | **30** | **23** | **3** | **4** | **30** |

24 of 25 non-abstained answers cite at least one source; when the evidence chunk was retrieved, the
answer cited it in 17 of 19 cases. One answer abstained although the evidence was among the sources.

### Error analysis

- The 6 retrieval misses of the dense baseline were checked by hand: 5 are genuine retrieval errors and
  1 is a false negative of the annotation (another passage answers the question). The metric is
  therefore conservative.
- Two questions (q14, q16) are missed by every configuration: the evidence is not even among the top 30
  candidates, so re-ranking cannot recover it. The answer shares no words with the question
  ("ogni quanto vengono aggiornati" vs "una sola volta, all'avvio"): this is a first-stage recall
  problem, not a ranking problem.
- After re-ranking, q22 drops the annotated passage in favour of another one that also answers the
  question, which however ends up 6th, just outside the top 5.

### Latency (RTX 4090)

| Step | Median |
|---|---|
| Question embedding | ~12 ms |
| Dense / hybrid search | ~1.1 / ~1.3 ms |
| Re-ranking of 30 candidates (fp32) | ~170 ms |
| Generation (~140 tokens/s) | 0.3 to 1.5 s, depending on answer length |

With 185 vectors an exact scan takes about 1 ms, so the HNSW index brings no measurable benefit here;
it becomes relevant from hundreds of thousands of vectors, at the cost of approximate results.

## Limitations

- Small evaluation set (26 answerable questions), single document, graded by the author of both the
  system and the document.
- Tables lose their structure during PDF extraction.
- The BM25 index lives in memory and must be rebuilt when documents change.
- The answer evaluation covers the baseline configuration only.

## Possible next steps

- **First-stage recall**: query rewriting or HyDE (embedding a hypothetical answer) for questions
  whose wording differs from the text.
- **Small-to-big retrieval**: search on small chunks, pass the surrounding larger section to the LLM.
- **Section-aware chunking**: prepend document and section titles to each chunk.
- **Larger evaluation set**: LLM-generated questions with manual review of a sample; LLM-as-judge
  (e.g. RAGAS) for faithfulness at scale.
- **Faster re-ranking**: fp16 inference, fewer candidates.
- **Layout-aware parsing** for tables (e.g. `page.find_tables()`, docling).

## Language

The codebase is in English; the corpus, the evaluation questions and the abstention sentence are in
Italian, the language of the documents. The source PDF is not included in the repository.