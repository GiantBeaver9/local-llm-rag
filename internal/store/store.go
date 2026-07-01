// Package store is a minimal vector database.
//
// A "vector database" sounds fancy, but at this scale it's just:
//   - a list of records, each holding some text and its embedding vector
//   - a way to find the records whose vectors are most similar to a query
//     vector (that's the "search")
//   - a way to save/load the whole thing to a JSON file so it survives restarts
//
// The similarity math is cosine similarity: the cosine of the angle between two
// vectors. 1.0 means "pointing the same way" (very similar), 0.0 means
// "unrelated". This naive version compares the query against EVERY record — a
// linear scan. That's perfectly fast for thousands of chunks, and it's the
// exact function we'll later swap out for a faster Rust implementation.
package store

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// Record is one stored chunk plus its embedding.
type Record struct {
	Source    string    `json:"source"`
	Index     int       `json:"index"`
	Text      string    `json:"text"`
	Embedding []float32 `json:"embedding"`
}

// Store holds all records in memory. Simple slice, nothing clever.
type Store struct {
	Records []Record `json:"records"`
}

// New returns an empty store.
func New() *Store {
	return &Store{}
}

// Add appends a record.
func (store *Store) Add(record Record) {
	store.Records = append(store.Records, record)
}

// Len reports how many records are stored.
func (store *Store) Len() int {
	return len(store.Records)
}

// Result is one search hit: the record and how similar it was to the query.
type Result struct {
	Record Record
	Score  float32
}

// Search returns the topK records most similar to the query vector, best first.
func (store *Store) Search(query []float32, topK int) []Result {
	results := make([]Result, 0, len(store.Records))
	for _, record := range store.Records {
		results = append(results, Result{
			Record: record,
			Score:  cosineSimilarity(query, record.Embedding),
		})
	}

	// Sort by score, highest first.
	sort.Slice(results, func(left, right int) bool {
		return results[left].Score > results[right].Score
	})

	if topK < len(results) {
		results = results[:topK]
	}
	return results
}

// cosineSimilarity computes the cosine of the angle between vectors vecA and
// vecB.
//
//	cos = (vecA · vecB) / (|vecA| * |vecB|)
//
// where vecA·vecB is the dot product and |vecA| is the vector's length
// (magnitude). If either vector has zero length (shouldn't happen with real
// embeddings) we return 0 to avoid dividing by zero.
//
// Note the accumulators are float64 even though the inputs are float32: summing
// thousands of tiny float32 products lets rounding error pile up, so we keep the
// running totals in a wider type and narrow back only at the very end.
func cosineSimilarity(vecA, vecB []float32) float32 {
	if len(vecA) != len(vecB) {
		return 0
	}
	var dotProduct, magnitudeA, magnitudeB float64
	for dim := range vecA {
		dotProduct += float64(vecA[dim]) * float64(vecB[dim])
		magnitudeA += float64(vecA[dim]) * float64(vecA[dim])
		magnitudeB += float64(vecB[dim]) * float64(vecB[dim])
	}
	if magnitudeA == 0 || magnitudeB == 0 {
		return 0
	}
	return float32(dotProduct / (math.Sqrt(magnitudeA) * math.Sqrt(magnitudeB)))
}

// ---- persistence ----------------------------------------------------------

// Save writes the whole store to a JSON file.
func (store *Store) Save(path string) error {
	data, err := json.MarshalIndent(store, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("writing store to %s: %w", path, err)
	}
	return nil
}

// Load reads a store back from a JSON file. If the file doesn't exist yet it
// returns an empty store (not an error) so the first "ingest" just works.
func Load(path string) (*Store, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return New(), nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading store from %s: %w", path, err)
	}
	var loaded Store
	if err := json.Unmarshal(data, &loaded); err != nil {
		return nil, fmt.Errorf("parsing store file %s: %w", path, err)
	}
	return &loaded, nil
}
