//! vindex — a small vector similarity index, exposed over a C ABI.
//!
//! Go calls the functions at the bottom of this file (the ones marked
//! `#[no_mangle] pub extern "C"`) through cgo. Everything above them is
//! ordinary, safe Rust that knows nothing about C.
//!
//! ## The similarity trick
//!
//! Cosine similarity is `dot(a, b) / (|a| * |b|)`. If we L2-*normalize* every
//! vector when it's added (divide it by its own length so `|v| == 1`), then at
//! query time the denominator is just `1 * 1`, and cosine similarity collapses
//! into a single dot product. So we pay the normalization cost once, on insert,
//! and every search afterwards is cheaper than the Go version that recomputed
//! magnitudes on every comparison.
//!
//! ## The memory contract with Go
//!
//! * Rust owns the `VectorIndex`. Go holds an opaque pointer and must return it
//!   via `vindex_free`. Go never dereferences or frees it directly.
//! * Input vectors (`*const f32`) are owned by Go. We only read them.
//! * Output buffers (`*mut i32`, `*mut f32`) are allocated by Go. We only write
//!   into them, never allocate memory that Go would have to free.

use std::slice;

/// An in-memory index of L2-normalized vectors. This is a normal Rust struct;
/// the FFI layer just hands out a pointer to one.
pub struct VectorIndex {
    dimensions: usize,
    /// Every stored vector, already normalized to unit length.
    normalized_vectors: Vec<Vec<f32>>,
}

impl VectorIndex {
    fn new(dimensions: usize) -> Self {
        VectorIndex {
            dimensions,
            normalized_vectors: Vec::new(),
        }
    }

    /// Store a copy of `vector`, normalized to unit length. Returns the id
    /// (its position) it was stored at, or `None` if the dimensions don't match.
    fn add(&mut self, vector: &[f32]) -> Option<usize> {
        if vector.len() != self.dimensions {
            return None;
        }
        let mut owned = vector.to_vec();
        normalize_in_place(&mut owned);
        self.normalized_vectors.push(owned);
        Some(self.normalized_vectors.len() - 1)
    }

    /// Score every stored vector against `query` and return the `top_k` best as
    /// (id, score) pairs, highest score first.
    fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        if query.len() != self.dimensions || self.normalized_vectors.is_empty() {
            return Vec::new();
        }

        let mut normalized_query = query.to_vec();
        normalize_in_place(&mut normalized_query);

        // Because every vector is unit length, cosine similarity == dot product.
        let mut scored: Vec<(usize, f32)> = self
            .normalized_vectors
            .iter()
            .enumerate()
            .map(|(id, stored)| (id, dot_product(&normalized_query, stored)))
            .collect();

        // Sort by score descending. (A binary heap would be asymptotically
        // better when top_k << n, but a full sort is clearer and plenty fast
        // for a brute-force index — an honest note for the reader.)
        scored.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }
}

/// Divide a vector by its own L2 length so it becomes unit length. A zero
/// vector is left untouched (dividing by zero would produce NaNs).
fn normalize_in_place(vector: &mut [f32]) {
    let magnitude: f32 = vector
        .iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt();
    if magnitude > 0.0 {
        for component in vector.iter_mut() {
            *component /= magnitude;
        }
    }
}

/// Plain dot product of two equal-length slices.
fn dot_product(first: &[f32], second: &[f32]) -> f32 {
    first
        .iter()
        .zip(second.iter())
        .map(|(left, right)| left * right)
        .sum()
}

// ---------------------------------------------------------------------------
// C ABI. This is the only part Go sees. Everything here is `unsafe` at heart —
// we are trusting the pointers Go hands us — so each function validates before
// it dereferences.
// ---------------------------------------------------------------------------

/// Create a new index for vectors of `dimensions` length. Returns a pointer Go
/// must later pass to `vindex_free`. Returns null if `dimensions` is 0.
#[no_mangle]
pub extern "C" fn vindex_new(dimensions: usize) -> *mut VectorIndex {
    if dimensions == 0 {
        return std::ptr::null_mut();
    }
    // Box::into_raw moves the struct onto the heap and hands ownership to the
    // caller (Go). Rust will NOT free it until we rebuild the Box in _free.
    Box::into_raw(Box::new(VectorIndex::new(dimensions)))
}

/// Destroy an index created by `vindex_new`. Safe to call with null.
///
/// # Safety
/// `index` must be a pointer returned by `vindex_new` and not used afterward.
#[no_mangle]
pub unsafe extern "C" fn vindex_free(index: *mut VectorIndex) {
    if !index.is_null() {
        // Reconstituting the Box lets Rust's normal drop logic free everything.
        drop(Box::from_raw(index));
    }
}

/// Add one vector to the index. Returns its id (>= 0) on success, or -1 on a
/// null pointer or dimension mismatch.
///
/// # Safety
/// `vector` must point to `length` readable `f32` values.
#[no_mangle]
pub unsafe extern "C" fn vindex_add(
    index: *mut VectorIndex,
    vector: *const f32,
    length: usize,
) -> i64 {
    if index.is_null() || vector.is_null() {
        return -1;
    }
    let index = &mut *index;
    let borrowed = slice::from_raw_parts(vector, length);
    match index.add(borrowed) {
        Some(id) => id as i64,
        None => -1,
    }
}

/// How many vectors are currently stored. Returns 0 for a null pointer.
///
/// # Safety
/// `index` must be null or a valid pointer from `vindex_new`.
#[no_mangle]
pub unsafe extern "C" fn vindex_len(index: *const VectorIndex) -> usize {
    if index.is_null() {
        return 0;
    }
    (*index).normalized_vectors.len()
}

/// Search for the `top_k` vectors most similar to `query`.
///
/// Results are written into the caller-provided `out_ids` and `out_scores`
/// arrays, which must each have room for at least `top_k` elements. The return
/// value is how many results were actually written (<= top_k).
///
/// # Safety
/// `query` must point to `query_length` readable floats; `out_ids` and
/// `out_scores` must each be writable for `top_k` elements.
#[no_mangle]
pub unsafe extern "C" fn vindex_search(
    index: *const VectorIndex,
    query: *const f32,
    query_length: usize,
    top_k: usize,
    out_ids: *mut i32,
    out_scores: *mut f32,
) -> usize {
    if index.is_null() || query.is_null() || out_ids.is_null() || out_scores.is_null() {
        return 0;
    }
    let index = &*index;
    let borrowed_query = slice::from_raw_parts(query, query_length);

    let results = index.search(borrowed_query, top_k);

    let id_output = slice::from_raw_parts_mut(out_ids, top_k);
    let score_output = slice::from_raw_parts_mut(out_scores, top_k);
    for (position, (id, score)) in results.iter().enumerate() {
        id_output[position] = *id as i32;
        score_output[position] = *score;
    }
    results.len()
}

// ---------------------------------------------------------------------------
// Pure-Rust unit tests. These run with `cargo test` and never touch the FFI —
// they check the actual index logic in isolation.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vector_scores_highest() {
        let mut index = VectorIndex::new(3);
        index.add(&[1.0, 0.0, 0.0]).unwrap();
        index.add(&[0.0, 1.0, 0.0]).unwrap();
        index.add(&[0.9, 0.1, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results[0].0, 0); // the identical vector wins
        assert!((results[0].1 - 1.0).abs() < 1e-6); // cosine ~ 1.0
        assert_eq!(results[1].0, 2); // the near one comes second
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let mut index = VectorIndex::new(3);
        assert_eq!(index.add(&[1.0, 2.0]), None);
    }
}
