package rustindex

import "testing"

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
