// Package rustindex is the Go side of the bridge to the Rust vector index.
//
// It uses cgo — Go's mechanism for calling C — to call the C ABI that our Rust
// crate (rust/vindex) exports. From the rest of the Go program's point of view
// this looks like an ordinary Go package; all the cross-language machinery is
// hidden behind the Index type below.
//
// Build requirement: the Rust static library must exist before `go build` runs,
// because cgo links it in. Build it with:
//
//	cd rust/vindex && cargo build --release
//
// (The Makefile at the repo root does this for you.)
package rustindex

/*
// The comment block immediately above `import "C"` is real C, compiled by cgo.

// #cgo directives tell the C toolchain how to build and link:
//   CFLAGS  — where to find vindex.h (this directory).
//   LDFLAGS — link our Rust archive, then the system libraries the Rust
//             standard library depends on (math, dynamic loader, pthreads).
//             ${SRCDIR} expands to this package's directory, so the path works
//             no matter where `go build` is invoked from.
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: ${SRCDIR}/../../rust/vindex/target/release/libvindex.a -lm -ldl -lpthread

#include "vindex.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// Index is a handle to a Rust-owned vector index. The unexported `handle` field
// is the opaque pointer Rust gave us; Go never looks inside it.
type Index struct {
	handle     *C.VectorIndex
	dimensions int
}

// New creates an index for vectors of the given dimensionality. Call Close when
// you're done so the Rust-side memory is released.
func New(dimensions int) (*Index, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("rustindex: dimensions must be positive, got %d", dimensions)
	}
	handle := C.vindex_new(C.size_t(dimensions))
	if handle == nil {
		return nil, fmt.Errorf("rustindex: Rust failed to allocate an index")
	}
	return &Index{handle: handle, dimensions: dimensions}, nil
}

// Add stores one vector and returns the id (its position) Rust assigned it.
func (index *Index) Add(vector []float32) (int, error) {
	if len(vector) != index.dimensions {
		return 0, fmt.Errorf("rustindex: expected %d dimensions, got %d", index.dimensions, len(vector))
	}

	// &vector[0] is the address of the slice's backing array. Rust reads
	// `len` floats starting there and does not hold the pointer after the call
	// returns, which is exactly what cgo's pointer-passing rules require.
	id := C.vindex_add(
		index.handle,
		(*C.float)(unsafe.Pointer(&vector[0])),
		C.size_t(len(vector)),
	)
	if id < 0 {
		return 0, fmt.Errorf("rustindex: Rust rejected the vector (id %d)", int64(id))
	}
	return int(id), nil
}

// Len reports how many vectors are stored.
func (index *Index) Len() int {
	return int(C.vindex_len(index.handle))
}

// SearchHit is one result: the id of a stored vector and its similarity score.
type SearchHit struct {
	ID    int
	Score float32
}

// Search returns the topK stored vectors most similar to query, best first.
func (index *Index) Search(query []float32, topK int) ([]SearchHit, error) {
	if len(query) != index.dimensions {
		return nil, fmt.Errorf("rustindex: query has %d dimensions, index expects %d", len(query), index.dimensions)
	}
	if topK <= 0 {
		return nil, nil
	}

	// We allocate the output buffers on the Go side and lend them to Rust to
	// fill. That way each side frees only what it allocated — no cross-allocator
	// frees, the classic FFI footgun.
	outIDs := make([]int32, topK)
	outScores := make([]float32, topK)

	written := C.vindex_search(
		index.handle,
		(*C.float)(unsafe.Pointer(&query[0])),
		C.size_t(len(query)),
		C.size_t(topK),
		(*C.int32_t)(unsafe.Pointer(&outIDs[0])),
		(*C.float)(unsafe.Pointer(&outScores[0])),
	)

	hits := make([]SearchHit, int(written))
	for position := range hits {
		hits[position] = SearchHit{ID: int(outIDs[position]), Score: outScores[position]}
	}
	return hits, nil
}

// Close releases the Rust-side index. After Close the Index must not be used.
// Calling Close twice is safe; the second call is a no-op.
func (index *Index) Close() {
	if index.handle != nil {
		C.vindex_free(index.handle)
		index.handle = nil
	}
	// Keep the Go object alive until exactly here, so a finalizer or the GC
	// can't free anything the C call above still needed.
	runtime.KeepAlive(index)
}
