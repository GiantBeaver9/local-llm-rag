// Package store persists the text metadata for each indexed chunk.
//
// Since Phase 3 the *vectors* and the search graph live in the Rust HNSW index
// (see internal/rustindex + rust/vindex), which owns its own binary file. This
// package holds only the human-readable side: which file each chunk came from
// and its text. The two files are kept aligned by position — record i here
// corresponds to node i in the Rust index — so a search result's id indexes
// straight into Records.
//
// (Earlier phases kept the embeddings and a Go cosine search here; that brute
// -force reference implementation now lives in git history, superseded by the
// Rust index.)
package store

import (
	"encoding/json"
	"fmt"
	"os"
)

// Record is the metadata for one indexed chunk.
type Record struct {
	Source string `json:"source"` // which file it came from
	Index  int    `json:"index"`  // its position within that file
	Text   string `json:"text"`   // the chunk text, used to build the prompt
}

// Store is an ordered list of records. Order matters: it must match the order
// vectors were added to the Rust index.
type Store struct {
	Records []Record `json:"records"`
}

// New returns an empty store.
func New() *Store {
	return &Store{}
}

// Add appends a record and returns its position (which equals the id the Rust
// index will assign the matching vector, since both grow together).
func (store *Store) Add(record Record) int {
	store.Records = append(store.Records, record)
	return len(store.Records) - 1
}

// Len reports how many records are stored.
func (store *Store) Len() int {
	return len(store.Records)
}

// Save writes the metadata to a JSON file.
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

// Load reads metadata back from a JSON file. A missing file yields an empty
// store (not an error) so the first ingest just works.
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
