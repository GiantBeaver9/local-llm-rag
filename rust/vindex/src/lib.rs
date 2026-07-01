//! vindex — a small HNSW vector index, exposed over a C ABI.
//!
//! Go calls the `extern "C"` functions at the bottom of this file through cgo.
//! Everything above them is ordinary, safe Rust.
//!
//! ## What HNSW is (in one paragraph)
//!
//! A brute-force index compares your query against *every* stored vector — fine
//! for thousands, painful for millions. HNSW (Hierarchical Navigable Small
//! World) instead builds a layered graph. The bottom layer contains every node
//! and is richly connected to near neighbours; each higher layer is a sparse
//! "express lane" holding a random subset. A search starts at the top, greedily
//! hops toward the query through the sparse layers to get close fast, then does
//! a careful best-first search on the bottom layer. That turns an O(n) scan into
//! roughly O(log n) hops — the algorithm real vector databases use.
//!
//! ## The similarity metric
//!
//! Vectors are L2-normalized on insert, so cosine similarity is just their dot
//! product. The graph orders nodes by *distance* = `1 - similarity` (smaller =
//! closer), and search returns the similarity back to Go.
//!
//! ## The memory contract with Go
//!
//! * Rust owns the `VectorIndex`. Go holds an opaque pointer and returns it via
//!   `vindex_free`; it never dereferences or frees it.
//! * Input vectors (`*const f32`) are owned by Go; we only read them.
//! * Search output buffers (`*mut i32`, `*mut f32`) are allocated by Go; we only
//!   write into them.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::ffi::CStr;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::os::raw::c_char;

// ---- tuning constants -----------------------------------------------------

const DEFAULT_M: usize = 16; // neighbours per node on upper layers
const DEFAULT_EF_CONSTRUCTION: usize = 200; // candidate list size while inserting
const DEFAULT_EF_SEARCH: usize = 64; // candidate list size while querying
const MAX_LEVEL: usize = 32; // hard cap so a bad random draw can't explode
const FILE_MAGIC: &[u8; 4] = b"HNSW";
const FILE_VERSION: u32 = 1;

// ---- the graph ------------------------------------------------------------

/// One node: its normalized vector, plus one neighbour list per layer it lives
/// on. `neighbors[0]` is the bottom (densest) layer.
struct Node {
    vector: Vec<f32>,
    neighbors: Vec<Vec<usize>>,
}

/// An HNSW index. Plain Rust; the FFI layer just hands out a pointer to one.
pub struct VectorIndex {
    dimensions: usize,
    m: usize,               // upper-layer neighbour cap
    m_max0: usize,          // bottom-layer neighbour cap (denser: 2 * m)
    ef_construction: usize, // breadth while building
    ef_search: usize,       // breadth while querying
    level_factor: f64,      // 1 / ln(m); controls how tall the graph gets
    entry_point: Option<usize>,
    max_layer: usize,
    nodes: Vec<Node>,
    rng: SplitMix64,
}

/// A (distance, id) pair used inside the search heaps. Ordered by distance
/// ascending, with id as a tiebreaker so results are deterministic.
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    distance: f32,
    id: usize,
}

impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // total_cmp gives a total order over floats (handles NaN safely).
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl VectorIndex {
    fn new(dimensions: usize) -> Self {
        let m = DEFAULT_M;
        VectorIndex {
            dimensions,
            m,
            m_max0: m * 2,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
            ef_search: DEFAULT_EF_SEARCH,
            level_factor: 1.0 / (m as f64).ln(),
            entry_point: None,
            max_layer: 0,
            nodes: Vec::new(),
            // Fixed seed => the same inserts always build the same graph, which
            // makes builds reproducible and tests deterministic.
            rng: SplitMix64::new(0x243F_6A88_85A3_08D3),
        }
    }

    /// Cosine distance (1 - similarity) between stored node `id` and `query`.
    /// Both are unit length, so similarity is just their dot product.
    fn distance(&self, id: usize, query: &[f32]) -> f32 {
        1.0 - dot_product(&self.nodes[id].vector, query)
    }

    /// Draw a random top layer for a new node. Most nodes get layer 0; each
    /// higher layer is exponentially rarer.
    fn random_level(&mut self) -> usize {
        let uniform = self.rng.next_f64().max(1e-12); // avoid ln(0)
        let level = (-uniform.ln() * self.level_factor).floor() as i64;
        level.clamp(0, MAX_LEVEL as i64) as usize
    }

    /// Insert a vector and return the id it was stored at (its position).
    /// Dimension mismatches return `None`.
    fn insert(&mut self, vector: &[f32]) -> Option<usize> {
        if vector.len() != self.dimensions {
            return None;
        }
        let mut normalized = vector.to_vec();
        normalize_in_place(&mut normalized);

        let id = self.nodes.len();
        let level = self.random_level();
        self.nodes.push(Node {
            vector: normalized.clone(),
            neighbors: vec![Vec::new(); level + 1],
        });

        // The very first node just becomes the entry point.
        let mut entry = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                self.max_layer = level;
                return Some(id);
            }
            Some(existing) => existing,
        };

        let query = normalized;
        let top = self.max_layer;

        // Phase 1: from the top layer down to just above our own top layer,
        // greedily hop closer. We're only navigating here, not linking.
        if top > level {
            for layer in ((level + 1)..=top).rev() {
                entry = self.greedy_descend(&query, entry, layer);
            }
        }

        // Phase 2: from our top layer down to 0, find neighbours and link.
        let start_layer = top.min(level);
        for layer in (0..=start_layer).rev() {
            let found = self.search_layer(&query, entry, self.ef_construction, layer);
            let neighbour_cap = if layer == 0 { self.m_max0 } else { self.m };

            let selected: Vec<usize> =
                found.iter().take(neighbour_cap).map(|c| c.id).collect();

            // Add links in both directions (an undirected graph).
            for &neighbour in &selected {
                self.nodes[id].neighbors[layer].push(neighbour);
                self.nodes[neighbour].neighbors[layer].push(id);
            }

            // Adding a back-link may push a neighbour over its cap; prune it
            // down to its closest connections so the graph stays sparse.
            for &neighbour in &selected {
                self.prune_neighbors(neighbour, layer, neighbour_cap);
            }

            // Descend to the next layer from the closest node we found.
            if let Some(closest) = found.first() {
                entry = closest.id;
            }
        }

        // A taller-than-everything node becomes the new entry point.
        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(id);
        }
        Some(id)
    }

    /// Trim node `id`'s neighbour list on `layer` to its `cap` closest links.
    fn prune_neighbors(&mut self, id: usize, layer: usize, cap: usize) {
        if self.nodes[id].neighbors[layer].len() <= cap {
            return;
        }
        let own_vector = self.nodes[id].vector.clone();
        let mut connections: Vec<Candidate> = self.nodes[id].neighbors[layer]
            .iter()
            .map(|&other| Candidate {
                distance: self.distance(other, &own_vector),
                id: other,
            })
            .collect();
        connections.sort();
        connections.truncate(cap);
        self.nodes[id].neighbors[layer] = connections.into_iter().map(|c| c.id).collect();
    }

    /// Greedy single-node descent on one layer: keep moving to the neighbour
    /// closest to `query` until no neighbour is closer. Returns where we stop.
    fn greedy_descend(&self, query: &[f32], start: usize, layer: usize) -> usize {
        let mut current = start;
        loop {
            let mut best_distance = self.distance(current, query);
            let mut best_id = current;
            if let Some(neighbours) = self.nodes[current].neighbors.get(layer) {
                for &neighbour in neighbours {
                    let candidate_distance = self.distance(neighbour, query);
                    if candidate_distance < best_distance {
                        best_distance = candidate_distance;
                        best_id = neighbour;
                    }
                }
            }
            if best_id == current {
                return current;
            }
            current = best_id;
        }
    }

    /// Best-first search on a single layer. Returns up to `ef` nodes closest to
    /// `query`, sorted nearest-first. This is the workhorse of both insert and
    /// query. It keeps two heaps: `frontier` (nearest unexplored first) and
    /// `results` (farthest kept first, so we can evict the worst).
    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        visited.insert(entry);

        let entry_candidate = Candidate {
            distance: self.distance(entry, query),
            id: entry,
        };

        // frontier: min-heap (via Reverse) — pop the nearest node to explore.
        let mut frontier = BinaryHeap::new();
        frontier.push(Reverse(entry_candidate));
        // results: max-heap — peek/pop the farthest node we're keeping.
        let mut results = BinaryHeap::new();
        results.push(entry_candidate);

        while let Some(Reverse(current)) = frontier.pop() {
            let farthest_kept = results.peek().expect("results is never empty").distance;
            // If the nearest thing left to explore is farther than our worst
            // kept result (and we already have enough), we're done.
            if current.distance > farthest_kept && results.len() >= ef {
                break;
            }

            if let Some(neighbours) = self.nodes[current.id].neighbors.get(layer) {
                for &neighbour in neighbours {
                    if !visited.insert(neighbour) {
                        continue; // already seen
                    }
                    let neighbour_distance = self.distance(neighbour, query);
                    let farthest_kept = results.peek().expect("results is never empty").distance;
                    if neighbour_distance < farthest_kept || results.len() < ef {
                        let candidate = Candidate {
                            distance: neighbour_distance,
                            id: neighbour,
                        };
                        frontier.push(Reverse(candidate));
                        results.push(candidate);
                        if results.len() > ef {
                            results.pop(); // drop the current farthest
                        }
                    }
                }
            }
        }

        results.into_sorted_vec() // ascending by distance => nearest first
    }

    /// Find the `k` nearest stored vectors to `query`, nearest first, as
    /// (id, similarity) pairs.
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if query.len() != self.dimensions || k == 0 {
            return Vec::new();
        }
        let mut entry = match self.entry_point {
            Some(point) => point,
            None => return Vec::new(),
        };

        let mut normalized = query.to_vec();
        normalize_in_place(&mut normalized);

        // Descend the express lanes to get near the query cheaply.
        for layer in (1..=self.max_layer).rev() {
            entry = self.greedy_descend(&normalized, entry, layer);
        }

        // Careful search on the bottom layer. ef must be >= k to return k hits.
        let ef = self.ef_search.max(k);
        let found = self.search_layer(&normalized, entry, ef, 0);

        found
            .into_iter()
            .take(k)
            .map(|candidate| (candidate.id, 1.0 - candidate.distance))
            .collect()
    }

    // ---- persistence ------------------------------------------------------

    /// Write the whole index (parameters + vectors + graph) to a binary file.
    fn save(&self, path: &str) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);

        writer.write_all(FILE_MAGIC)?;
        write_u32(&mut writer, FILE_VERSION)?;
        write_u64(&mut writer, self.dimensions as u64)?;
        write_u64(&mut writer, self.m as u64)?;
        write_u64(&mut writer, self.m_max0 as u64)?;
        write_u64(&mut writer, self.ef_construction as u64)?;
        write_u64(&mut writer, self.ef_search as u64)?;
        write_u64(&mut writer, self.level_factor.to_bits())?;
        write_u64(&mut writer, self.rng.state)?;
        // entry point encoded as i64 (-1 means "none").
        write_i64(
            &mut writer,
            self.entry_point.map(|p| p as i64).unwrap_or(-1),
        )?;
        write_u64(&mut writer, self.max_layer as u64)?;
        write_u64(&mut writer, self.nodes.len() as u64)?;

        for node in &self.nodes {
            write_u64(&mut writer, node.neighbors.len() as u64)?;
            for &component in &node.vector {
                write_u32(&mut writer, component.to_bits())?;
            }
            for layer in &node.neighbors {
                write_u64(&mut writer, layer.len() as u64)?;
                for &neighbour in layer {
                    write_u64(&mut writer, neighbour as u64)?;
                }
            }
        }
        writer.flush()
    }

    /// Read an index back from a file written by `save`.
    fn load(path: &str) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != FILE_MAGIC {
            return Err(bad_data("not a vindex file (bad magic bytes)"));
        }
        if read_u32(&mut reader)? != FILE_VERSION {
            return Err(bad_data("unsupported vindex file version"));
        }

        let dimensions = read_u64(&mut reader)? as usize;
        let m = read_u64(&mut reader)? as usize;
        let m_max0 = read_u64(&mut reader)? as usize;
        let ef_construction = read_u64(&mut reader)? as usize;
        let ef_search = read_u64(&mut reader)? as usize;
        let level_factor = f64::from_bits(read_u64(&mut reader)?);
        let rng_state = read_u64(&mut reader)?;
        let entry_raw = read_i64(&mut reader)?;
        let entry_point = if entry_raw < 0 {
            None
        } else {
            Some(entry_raw as usize)
        };
        let max_layer = read_u64(&mut reader)? as usize;
        let node_count = read_u64(&mut reader)? as usize;

        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let layer_count = read_u64(&mut reader)? as usize;
            let mut vector = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                vector.push(f32::from_bits(read_u32(&mut reader)?));
            }
            let mut neighbors = Vec::with_capacity(layer_count);
            for _ in 0..layer_count {
                let neighbour_count = read_u64(&mut reader)? as usize;
                let mut layer = Vec::with_capacity(neighbour_count);
                for _ in 0..neighbour_count {
                    layer.push(read_u64(&mut reader)? as usize);
                }
                neighbors.push(layer);
            }
            nodes.push(Node { vector, neighbors });
        }

        Ok(VectorIndex {
            dimensions,
            m,
            m_max0,
            ef_construction,
            ef_search,
            level_factor,
            entry_point,
            max_layer,
            nodes,
            rng: SplitMix64 { state: rng_state },
        })
    }
}

// ---- small helpers --------------------------------------------------------

/// Divide a vector by its L2 length so it becomes unit length. A zero vector is
/// left untouched (dividing by zero would make NaNs).
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

/// Dot product of two equal-length slices, accumulated in f64 for precision.
fn dot_product(first: &[f32], second: &[f32]) -> f32 {
    let mut total = 0.0f64;
    for (left, right) in first.iter().zip(second.iter()) {
        total += f64::from(*left) * f64::from(*right);
    }
    total as f32
}

/// A tiny, dependency-free PRNG (SplitMix64). Deterministic given its seed —
/// which is exactly what we want for reproducible index builds.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A float in [0, 1), using the top 53 bits (f64's mantissa width).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn bad_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
fn write_i64<W: Write>(writer: &mut W, value: i64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buffer = [0u8; 8];
    reader.read_exact(&mut buffer)?;
    Ok(u64::from_le_bytes(buffer))
}
fn read_i64<R: Read>(reader: &mut R) -> io::Result<i64> {
    let mut buffer = [0u8; 8];
    reader.read_exact(&mut buffer)?;
    Ok(i64::from_le_bytes(buffer))
}
fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_le_bytes(buffer))
}

// ---------------------------------------------------------------------------
// C ABI. The only part Go sees. Each function validates the raw pointers Go
// hands it before dereferencing.
// ---------------------------------------------------------------------------

/// Create a new index for vectors of `dimensions` length. Null if `dimensions`
/// is 0. The returned pointer must be released with `vindex_free`.
#[no_mangle]
pub extern "C" fn vindex_new(dimensions: usize) -> *mut VectorIndex {
    if dimensions == 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(VectorIndex::new(dimensions)))
}

/// Destroy an index. Safe to call with null.
///
/// # Safety
/// `index` must come from `vindex_new`/`vindex_load` and be unused afterward.
#[no_mangle]
pub unsafe extern "C" fn vindex_free(index: *mut VectorIndex) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}

/// Add a vector. Returns its id (>= 0), or -1 on null / dimension mismatch.
///
/// # Safety
/// `vector` must point to `length` readable f32 values.
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
    let borrowed = std::slice::from_raw_parts(vector, length);
    match index.insert(borrowed) {
        Some(id) => id as i64,
        None => -1,
    }
}

/// Number of vectors stored. 0 for a null pointer.
///
/// # Safety
/// `index` must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn vindex_len(index: *const VectorIndex) -> usize {
    if index.is_null() {
        return 0;
    }
    (*index).nodes.len()
}

/// The dimensionality the index was created with. 0 for a null pointer. Go uses
/// this to learn the shape of an index it just loaded from disk.
///
/// # Safety
/// `index` must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn vindex_dimensions(index: *const VectorIndex) -> usize {
    if index.is_null() {
        return 0;
    }
    (*index).dimensions
}

/// Search for the `top_k` vectors most similar to `query`. Writes ids and
/// scores into the caller-owned `out_ids` / `out_scores` arrays (each must hold
/// at least `top_k` elements) and returns how many were written.
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
    let borrowed_query = std::slice::from_raw_parts(query, query_length);
    let results = index.search(borrowed_query, top_k);

    let id_output = std::slice::from_raw_parts_mut(out_ids, top_k);
    let score_output = std::slice::from_raw_parts_mut(out_scores, top_k);
    for (position, (id, score)) in results.iter().enumerate() {
        id_output[position] = *id as i32;
        score_output[position] = *score;
    }
    results.len()
}

/// Save the index to `path`. Returns 0 on success, negative on error.
///
/// # Safety
/// `index` must be valid; `path` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn vindex_save(index: *const VectorIndex, path: *const c_char) -> i32 {
    if index.is_null() || path.is_null() {
        return -1;
    }
    let path = match CStr::from_ptr(path).to_str() {
        Ok(text) => text,
        Err(_) => return -2,
    };
    match (*index).save(path) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

/// Load an index from `path`. Returns null on error. The returned pointer must
/// be released with `vindex_free`.
///
/// # Safety
/// `path` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn vindex_load(path: *const c_char) -> *mut VectorIndex {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let path = match CStr::from_ptr(path).to_str() {
        Ok(text) => text,
        Err(_) => return std::ptr::null_mut(),
    };
    match VectorIndex::load(path) {
        Ok(index) => Box::into_raw(Box::new(index)),
        Err(_) => std::ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust unit tests (run with `cargo test`; they never touch the FFI).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // Build a deterministic set of distinct vectors on the unit circle-ish.
    fn sample_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
        let mut rng = SplitMix64::new(42);
        (0..count)
            .map(|_| {
                (0..dimensions)
                    .map(|_| rng.next_f64() as f32 - 0.5)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn finds_the_exact_match() {
        let mut index = VectorIndex::new(8);
        let vectors = sample_vectors(200, 8);
        for vector in &vectors {
            index.insert(vector).unwrap();
        }
        // Querying with an exact copy of vector #137 should return it first.
        let hits = index.search(&vectors[137], 1);
        assert_eq!(hits[0].0, 137);
        assert!(hits[0].1 > 0.999, "similarity to itself should be ~1.0");
    }

    #[test]
    fn recall_matches_brute_force() {
        let dimensions = 16;
        let vectors = sample_vectors(500, dimensions);
        let mut index = VectorIndex::new(dimensions);
        for vector in &vectors {
            index.insert(vector).unwrap();
        }

        // For 20 queries, check HNSW's top-1 equals the true brute-force top-1
        // most of the time (approximate indexes aren't required to be perfect).
        let queries = sample_vectors(20, dimensions);
        let mut correct = 0;
        for query in &queries {
            let hnsw_top = index.search(query, 1)[0].0;

            let mut normalized_query = query.clone();
            normalize_in_place(&mut normalized_query);
            let mut best_id = 0;
            let mut best_similarity = f32::MIN;
            for (id, _) in vectors.iter().enumerate() {
                let similarity = 1.0 - index.distance(id, &normalized_query);
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_id = id;
                }
            }
            if hnsw_top == best_id {
                correct += 1;
            }
        }
        assert!(correct >= 18, "recall too low: {}/20", correct);
    }

    #[test]
    fn save_then_load_is_identical() {
        let vectors = sample_vectors(100, 8);
        let mut index = VectorIndex::new(8);
        for vector in &vectors {
            index.insert(vector).unwrap();
        }

        let path = std::env::temp_dir().join("vindex_test_roundtrip.index");
        let path = path.to_str().unwrap();
        index.save(path).unwrap();
        let loaded = VectorIndex::load(path).unwrap();

        assert_eq!(loaded.nodes.len(), index.nodes.len());
        assert_eq!(loaded.entry_point, index.entry_point);
        // Same query gives the same top-3 ids before and after a round trip.
        let before = index.search(&vectors[10], 3);
        let after = loaded.search(&vectors[10], 3);
        let before_ids: Vec<usize> = before.iter().map(|hit| hit.0).collect();
        let after_ids: Vec<usize> = after.iter().map(|hit| hit.0).collect();
        assert_eq!(before_ids, after_ids);
    }
}
