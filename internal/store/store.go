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
func (s *Store) Add(r Record) {
	s.Records = append(s.Records, r)
}

// Len reports how many records are stored.
func (s *Store) Len() int {
	return len(s.Records)
}

// Result is one search hit: the record and how similar it was to the query.
type Result struct {
	Record Record
	Score  float32
}

// Search returns the topK records most similar to the query vector, best first.
func (s *Store) Search(query []float32, topK int) []Result {
	results := make([]Result, 0, len(s.Records))
	for _, r := range s.Records {
		results = append(results, Result{
			Record: r,
			Score:  cosineSimilarity(query, r.Embedding),
		})
	}

	// Sort by score, highest first.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if topK < len(results) {
		results = results[:topK]
	}
	return results
}

// cosineSimilarity computes the cosine of the angle between vectors a and b.
//
//	cos = (a · b) / (|a| * |b|)
//
// where a·b is the dot product and |a| is the vector's length (magnitude).
// If either vector has zero length (shouldn't happen with real embeddings) we
// return 0 to avoid dividing by zero.
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, magA, magB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		magA += float64(a[i]) * float64(a[i])
		magB += float64(b[i]) * float64(b[i])
	}
	if magA == 0 || magB == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(magA) * math.Sqrt(magB)))
}

// ---- persistence ----------------------------------------------------------

// Save writes the whole store to a JSON file.
func (s *Store) Save(path string) error {
	data, err := json.MarshalIndent(s, "", "  ")
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
	var s Store
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("parsing store file %s: %w", path, err)
	}
	return &s, nil
}
