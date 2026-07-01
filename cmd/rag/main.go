// Command rag is a tiny local RAG (Retrieval-Augmented Generation) tool.
//
// RAG in one sentence: instead of hoping the model already knows the answer,
// we *retrieve* the most relevant passages from your own documents and hand
// them to the model as context, so its answer is grounded in your data.
//
// Two subcommands:
//
//	rag ingest <file-or-dir>   read text, chunk it, embed each chunk, save
//	rag ask "your question"    retrieve relevant chunks, ask the model
//
// Configuration comes from environment variables (all optional):
//
//	LMSTUDIO_URL    default http://localhost:1234/v1
//	EMBED_MODEL     default nomic-embed-text-v1.5
//	CHAT_MODEL      default local-model
//	RAG_STORE       default store.json   (where the vector DB is saved)
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/giantbeaver9/local-llm-rag/internal/chunk"
	"github.com/giantbeaver9/local-llm-rag/internal/lmstudio"
	"github.com/giantbeaver9/local-llm-rag/internal/store"
)

func main() {
	// os.Args[0] is the program name; the subcommand is os.Args[1].
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	client := lmstudio.New(
		env("LMSTUDIO_URL", "http://localhost:1234/v1"),
		env("EMBED_MODEL", "nomic-embed-text-v1.5"),
		env("CHAT_MODEL", "local-model"),
	)
	storePath := env("RAG_STORE", "store.json")

	var err error
	switch os.Args[1] {
	case "ingest":
		err = runIngest(client, storePath, os.Args[2:])
	case "ask":
		err = runAsk(client, storePath, os.Args[2:])
	default:
		usage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "\nerror: %v\n", err)
		os.Exit(1)
	}
}

// runIngest reads the given file or directory, splits everything into chunks,
// embeds each chunk via LM Studio, and saves the growing vector store.
func runIngest(client *lmstudio.Client, storePath string, args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: rag ingest <file-or-dir>")
	}
	target := args[0]

	files, err := gatherTextFiles(target)
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return fmt.Errorf("no .txt or .md files found under %q", target)
	}

	db, err := store.Load(storePath)
	if err != nil {
		return err
	}

	ctx := context.Background()
	totalChunks := 0

	for _, path := range files {
		raw, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("reading %s: %w", path, err)
		}

		// ~200 words per chunk with ~40 words of overlap. Tune later.
		chunks := chunk.Split(path, string(raw), 200, 40)
		fmt.Printf("%-40s %d chunks\n", path, len(chunks))

		for _, docChunk := range chunks {
			embedding, err := client.Embed(ctx, docChunk.Text)
			if err != nil {
				return fmt.Errorf("embedding chunk %d of %s: %w", docChunk.Index, path, err)
			}
			db.Add(store.Record{
				Source:    docChunk.Source,
				Index:     docChunk.Index,
				Text:      docChunk.Text,
				Embedding: embedding,
			})
			totalChunks++
		}
	}

	if err := db.Save(storePath); err != nil {
		return err
	}
	fmt.Printf("\nembedded %d chunks; store now holds %d records -> %s\n",
		totalChunks, db.Len(), storePath)
	return nil
}

// runAsk embeds the user's question, finds the most similar chunks, and asks
// the chat model to answer using only those chunks as context.
func runAsk(client *lmstudio.Client, storePath string, args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("usage: rag ask \"your question\"")
	}
	question := strings.Join(args, " ")

	db, err := store.Load(storePath)
	if err != nil {
		return err
	}
	if db.Len() == 0 {
		return fmt.Errorf("the store is empty — run `rag ingest <path>` first")
	}

	ctx := context.Background()

	// 1. Turn the question into a vector.
	queryVec, err := client.Embed(ctx, question)
	if err != nil {
		return err
	}

	// 2. Retrieve the most relevant chunks.
	const topK = 4
	hits := db.Search(queryVec, topK)

	fmt.Println("retrieved context:")
	var contextBuilder strings.Builder
	for rank, hit := range hits {
		fmt.Printf("  [%d] %.3f  %s#%d\n", rank+1, hit.Score, hit.Record.Source, hit.Record.Index)
		fmt.Fprintf(&contextBuilder, "[%d] %s\n\n", rank+1, hit.Record.Text)
	}

	// 3. Build the prompt: a system instruction + the context + the question.
	messages := []lmstudio.Message{
		{
			Role: "system",
			Content: "You answer questions using ONLY the provided context. " +
				"If the context does not contain the answer, say you don't know. " +
				"Cite the passage numbers you used, like [1] or [2].",
		},
		{
			Role: "user",
			Content: fmt.Sprintf("Context:\n%s\nQuestion: %s",
				contextBuilder.String(), question),
		},
	}

	// 4. Ask the model.
	answer, err := client.Chat(ctx, messages)
	if err != nil {
		return err
	}

	fmt.Printf("\nanswer:\n%s\n", answer)
	return nil
}

// gatherTextFiles returns every .txt/.md file at target. target may be a single
// file or a directory (searched recursively).
func gatherTextFiles(target string) ([]string, error) {
	info, err := os.Stat(target)
	if err != nil {
		return nil, fmt.Errorf("cannot access %q: %w", target, err)
	}

	if !info.IsDir() {
		return []string{target}, nil
	}

	var files []string
	err = filepath.WalkDir(target, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		switch strings.ToLower(filepath.Ext(path)) {
		case ".txt", ".md":
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

// env returns the value of an environment variable, or a fallback if unset.
func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func usage() {
	fmt.Fprint(os.Stderr, `rag — a tiny local RAG tool backed by LM Studio

usage:
  rag ingest <file-or-dir>    chunk + embed documents into the vector store
  rag ask "your question"     retrieve relevant chunks and answer

environment (all optional):
  LMSTUDIO_URL   default http://localhost:1234/v1
  EMBED_MODEL    default nomic-embed-text-v1.5
  CHAT_MODEL     default local-model
  RAG_STORE      default store.json
`)
}
