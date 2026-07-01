package rustindex

import (
	"path/filepath"
	"testing"
)

// TestSearchOrder exercises the entire Go -> cgo -> Rust -> cgo -> Go round
// trip with synthetic vectors, so it verifies the bridge itself without needing
// LM Studio or any real embeddings.
func TestSearchOrder(t *testing.T) {
	index, err := New(3)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer index.Close()

	// Three reference vectors along/near the x axis.
	vectors := [][]float32{
		{1.0, 0.0, 0.0}, // id 0 — points straight along x
		{0.0, 1.0, 0.0}, // id 1 — orthogonal, should score ~0
		{0.9, 0.1, 0.0}, // id 2 — close to x
	}
	for _, vector := range vectors {
		if _, err := index.Add(vector); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	if got := index.Len(); got != 3 {
		t.Fatalf("Len = %d, want 3", got)
	}

	// Query straight along x: id 0 should win, id 2 second.
	hits, err := index.Search([]float32{1.0, 0.0, 0.0}, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("got %d hits, want 2", len(hits))
	}
	if hits[0].ID != 0 {
		t.Errorf("best hit id = %d, want 0", hits[0].ID)
	}
	if hits[0].Score < 0.999 {
		t.Errorf("best score = %f, want ~1.0", hits[0].Score)
	}
	if hits[1].ID != 2 {
		t.Errorf("second hit id = %d, want 2", hits[1].ID)
	}
}

// TestSaveLoadRoundTrip builds an index, saves it, loads it back through the
// FFI, and checks the loaded index returns the same results — proving the
// persistence path works across the language boundary.
func TestSaveLoadRoundTrip(t *testing.T) {
	index, err := New(4)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	vectors := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0.8, 0.2, 0, 0},
		{0, 0, 1, 0},
	}
	for _, vector := range vectors {
		if _, err := index.Add(vector); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	path := filepath.Join(t.TempDir(), "roundtrip.index")
	if err := index.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	index.Close()

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer loaded.Close()

	if loaded.Len() != len(vectors) {
		t.Fatalf("loaded Len = %d, want %d", loaded.Len(), len(vectors))
	}
	if loaded.Dimensions() != 4 {
		t.Fatalf("loaded Dimensions = %d, want 4", loaded.Dimensions())
	}
	hits, err := loaded.Search([]float32{1, 0, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || hits[0].ID != 0 {
		t.Errorf("best hit after load = %+v, want id 0", hits)
	}
}

// TestDimensionMismatch checks the Go-side guard rejects wrong-sized vectors
// before they ever reach Rust.
func TestDimensionMismatch(t *testing.T) {
	index, err := New(4)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer index.Close()

	if _, err := index.Add([]float32{1, 2, 3}); err == nil {
		t.Error("expected an error adding a 3-d vector to a 4-d index, got nil")
	}
}
