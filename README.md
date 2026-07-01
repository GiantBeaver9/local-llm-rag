# local-llm-rag

A tiny, from-scratch **RAG** (Retrieval-Augmented Generation) tool written in
Go, backed by a **local** LLM served by [LM Studio](https://lmstudio.ai/).

It's built to be *read*, not just run — every file is heavily commented so you
can follow exactly how a RAG pipeline works end to end. No frameworks, no
third-party dependencies, just Go's standard library.

> **Polyglot:** similarity search runs in a **Rust HNSW index** that Go calls
> in-process via **cgo/FFI** (`rust/vindex` + `internal/rustindex`). Go owns
> ingestion, text metadata, and orchestration; Rust owns the vectors, the graph,
> and the hot similarity math — and persists them so queries never rebuild.

## How it works

```
                ┌──────────────── ingest ────────────────┐
  your docs ──▶ chunk ──▶ embed (LM Studio) ──▶ append ──▶ store.json  (text)
                                                    └─────▶ store.index (Rust HNSW)
                ┌───────────────── ask ──────────────────┐
  question ──▶ embed ──▶ load store.index ──▶ HNSW search ──▶ ids
                                                              │
                    top-K chunks ──▶ prompt ──▶ chat model ──▶ grounded answer
```

1. **Ingest** – read `.txt`/`.md` files, split them into overlapping chunks,
   embed each chunk, and *append* it to two aligned stores: the text metadata
   (`store.json`) and the Rust HNSW index (`store.index`). HNSW inserts
   incrementally, so re-running ingest grows the index instead of rebuilding it.
2. **Ask** – embed your question, **load** the prebuilt HNSW graph, search it,
   and feed the top chunks to the chat model as grounding context.

The two files stay position-aligned (record *i* ↔ node *i*), so a search result
id indexes straight into the metadata. A built-in check refuses to run if they
ever drift.

## Prerequisites

- [Go](https://go.dev/dl/) 1.24+
- [Rust](https://rustup.rs/) (stable) + a C compiler (`gcc`/`clang`) for cgo
- [LM Studio](https://lmstudio.ai/) with:
  - an **embedding** model loaded (e.g. `nomic-embed-text-v1.5`)
  - a **chat** model loaded (any instruct model)
  - the local server started (LM Studio → **Developer** tab → **Start Server**)

## Usage

The Rust library must be compiled before the Go binary (cgo links it in). The
Makefile handles the ordering:

```bash
# 1. Build (compiles the Rust index, then the Go binary)
make build          # or: make rust && make go

# 2. Ingest the sample docs (or point it at your own file/folder)
./rag ingest docs

# 3. Ask a question (retrieval runs through the Rust index)
./rag ask "What is cosine similarity and why is it used in vector search?"

# Run every test (Rust unit tests + Go tests incl. the cgo bridge)
make test
```

### Configuration (all optional, via environment variables)

| Variable       | Default                        | Meaning                          |
| -------------- | ------------------------------ | -------------------------------- |
| `LMSTUDIO_URL` | `http://localhost:1234/v1`     | LM Studio server address         |
| `EMBED_MODEL`  | `nomic-embed-text-v1.5`        | embedding model name in LM Studio|
| `CHAT_MODEL`   | `local-model`                  | chat model name in LM Studio     |
| `RAG_STORE`    | `store.json`                   | text metadata file               |
| `RAG_INDEX`    | `store.index`                  | Rust HNSW index file             |

> Tip: in LM Studio, the exact model name to use is shown next to each loaded
> model. Set `EMBED_MODEL`/`CHAT_MODEL` to match if the defaults don't.

## Project layout

```
cmd/rag/            CLI entry point (ingest + ask commands)
internal/lmstudio/  HTTP client for LM Studio (embeddings + chat)
internal/chunk/     splits documents into overlapping chunks
internal/store/     text metadata store (source + chunk text), JSON-persisted
internal/rustindex/ cgo bridge: the Go side of the FFI (wraps vindex.h)
rust/vindex/        the Rust HNSW index (compiles to libvindex.a)
docs/               sample documents to try it on
Makefile            builds Rust then Go in the right order
```

## How the Go ↔ Rust bridge works

```
Go (internal/rustindex)  --cgo-->  C ABI (vindex.h)  -->  Rust (rust/vindex)
```

The memory contract that keeps it safe:

- **Rust owns the index.** Go holds an opaque `*C.VectorIndex` and returns it
  via `vindex_free`. It never dereferences or frees it.
- **Go owns the input vectors.** Rust only borrows them (`slice::from_raw_parts`).
- **Go owns the output buffers.** Rust writes results into caller-provided
  arrays — nothing crosses the boundary needing a foreign `free`.

Rust L2-normalizes each vector on insert, so every search is a single dot
product instead of recomputing magnitudes per comparison.

## The HNSW index (`rust/vindex`)

Instead of scanning every vector, `vindex` builds a **Hierarchical Navigable
Small World** graph:

- Each node gets a random top layer (exponentially rarer the higher you go).
- The bottom layer holds every node, richly connected to near neighbours; upper
  layers are sparse "express lanes".
- Search descends the express lanes greedily to get *close* fast, then does a
  careful best-first search on the bottom layer — roughly O(log n) hops instead
  of O(n).

It's incrementally insertable and self-persisting (a hand-rolled binary format,
no serde), with a deterministic PRNG so builds are reproducible. Tuning knobs
(`M`, `efConstruction`, `efSearch`) live at the top of `src/lib.rs`.

> **Approximate, by design.** HNSW trades a little recall for a lot of speed —
> the tests assert it matches brute-force top-1 on ≥18/20 queries, not 20/20.
> Raise `efSearch` for better recall at the cost of speed.

## What to explore next

- Increase `topK` in `cmd/rag/main.go` and watch retrieval change.
- Tune chunk size/overlap in `runIngest`, or `M`/`efSearch` in `src/lib.rs`.
- Add a `rag stats` command that reports index size and layer distribution.
- Support deletion/updates (HNSW makes this genuinely tricky — a great rabbit
  hole).
