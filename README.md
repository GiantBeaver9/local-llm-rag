# local-llm-rag

A tiny, from-scratch **RAG** (Retrieval-Augmented Generation) tool written in
Go, backed by a **local** LLM served by [LM Studio](https://lmstudio.ai/).

It's built to be *read*, not just run — every file is heavily commented so you
can follow exactly how a RAG pipeline works end to end. No frameworks, no
third-party dependencies, just Go's standard library.

> **Roadmap:** the similarity-search core (`internal/store`) is deliberately
> simple Go today. Phase 2 reimplements it as a high-performance **Rust**
> vector index that Go calls into — so this repo grows into a Go + Rust
> polyglot project.

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
- [LM Studio](https://lmstudio.ai/) with:
  - an **embedding** model loaded (e.g. `nomic-embed-text-v1.5`)
  - a **chat** model loaded (any instruct model)
  - the local server started (LM Studio → **Developer** tab → **Start Server**)

## Usage

```bash
# 1. Build
go build -o rag ./cmd/rag

# 2. Ingest the sample docs (or point it at your own file/folder)
./rag ingest docs

# 3. Ask a question
./rag ask "What is cosine similarity and why is it used in vector search?"
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
internal/store/     the vector store: cosine search + JSON persistence
docs/               sample documents to try it on
```

## What to explore next

- Increase `topK` in `cmd/rag/main.go` and watch retrieval change.
- Tune chunk size/overlap in `runIngest`.
- **Phase 2:** swap `internal/store`'s cosine search for a Rust index.
