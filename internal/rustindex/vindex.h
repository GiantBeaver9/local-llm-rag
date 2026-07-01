/*
 * vindex.h — the C interface to the Rust vector index (rust/vindex).
 *
 * This header is the shared contract: Rust promises to export exactly these
 * symbols (see rust/vindex/src/lib.rs), and Go's cgo layer includes this file
 * to know their signatures. Keep the two in sync by hand — if you change a
 * signature in Rust, change it here too.
 */
#ifndef VINDEX_H
#define VINDEX_H

#include <stddef.h> /* size_t */
#include <stdint.h> /* int32_t, int64_t */

/*
 * Opaque handle. C (and therefore Go) never sees inside VectorIndex; it only
 * ever holds a pointer to one. The real definition lives in Rust.
 */
typedef struct VectorIndex VectorIndex;

/* Create an index for vectors of `dimensions` length. NULL if dimensions == 0. */
VectorIndex *vindex_new(size_t dimensions);

/* Free an index created by vindex_new. Safe to call with NULL. */
void vindex_free(VectorIndex *index);

/* Add a vector. Returns its id (>= 0), or -1 on error (null / wrong length). */
int64_t vindex_add(VectorIndex *index, const float *vector, size_t length);

/* Number of vectors stored. */
size_t vindex_len(const VectorIndex *index);

/* Dimensionality the index was created with (used after loading from disk). */
size_t vindex_dimensions(const VectorIndex *index);

/*
 * Find the top_k most similar vectors to `query`. Writes ids and scores into
 * the caller-owned out_ids / out_scores arrays (each must hold >= top_k
 * elements) and returns how many results were written.
 */
size_t vindex_search(const VectorIndex *index, const float *query,
                     size_t query_length, size_t top_k, int32_t *out_ids,
                     float *out_scores);

/* Save the index to a file. Returns 0 on success, negative on error. */
int32_t vindex_save(const VectorIndex *index, const char *path);

/* Load an index from a file. Returns NULL on error; free it with vindex_free. */
VectorIndex *vindex_load(const char *path);

#endif /* VINDEX_H */
