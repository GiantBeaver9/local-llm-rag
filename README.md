# local-llm-rag

A tiny, from-scratch **RAG** (Retrieval-Augmented Generation) tool written in
Go, backed by a **local** LLM served by [LM Studio](https://lmstudio.ai/).

It's built to be *read*, not just run — every file is heavily commented so you
can follow exactly how a RAG pipeline works end to end. No frameworks, no
third-party dependencies, just Go's standard library.

> **Polyglot:** similarity search runs in a **Rust** vector index that Go calls
> in-process via **cgo/FFI** (`rust/vindex` + `internal/rustindex`). Go owns
> ingestion, persistence, and orchestration; Rust owns the hot similarity math.
> The Go reference implementation of the same cosine math still lives in
> `internal/store` for comparison.

## How it works

```
                ┌─────────── ingest ───────────┐
  your docs ──▶ chunk ──▶ embed (LM Studio) ──▶ vector store (store.json)
                                                        │
                ┌──────────── ask ───────────────┐      │
  question ──▶ embed ──▶ cosine search ──────────┼──────┘
                                                 ▼
                    top-K chunks ──▶ prompt ──▶ chat model ──▶ grounded answer
```

1. **Ingest** – read `.txt`/`.md` files, split them into overlapping chunks,
   turn each chunk into a vector with the embedding model, and save them.
2. **Ask** – embed your question, find the most similar chunks via cosine
   similarity, and feed them to the chat model as grounding context.

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
| `RAG_STORE`    | `store.json`                   | where the vector DB is saved     |

> Tip: in LM Studio, the exact model name to use is shown next to each loaded
> model. Set `EMBED_MODEL`/`CHAT_MODEL` to match if the defaults don't.

## Project layout

```
cmd/rag/            CLI entry point (ingest + ask commands)
internal/lmstudio/  HTTP client for LM Studio (embeddings + chat)
internal/chunk/     splits documents into overlapping chunks
internal/store/     Go vector store: reference cosine search + JSON persistence
internal/rustindex/ cgo bridge: the Go side of the FFI (wraps vindex.h)
rust/vindex/        the Rust vector index (compiles to libvindex.a)
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

## What to explore next

- Increase `topK` in `cmd/rag/main.go` and watch retrieval change.
- Tune chunk size/overlap in `runIngest`.
- Swap the brute-force scan in `rust/vindex` for an approximate index (HNSW) —
  the interface stays the same, only the Rust internals change.
