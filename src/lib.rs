//! # plato-math
//!
//! Shared vector math primitives for the PLATO ecosystem.
//!
//! Zero dependencies. Pure Rust. No panics — all functions handle edge cases
//! (empty vectors, mismatched lengths) gracefully, returning default values.

// ── Type Aliases ──────────────────────────────────────────────────────────────

/// A vector of f64 values, the core type for all operations.
pub type VecF64 = Vec<f64>;

// ── Distance Functions ────────────────────────────────────────────────────────

/// Cosine similarity between two vectors. Returns 0.0 for empty or mismatched.
///
/// Range: [-1.0, 1.0]. Identical vectors → 1.0, orthogonal → 0.0, opposite → -1.0.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot = dot_product(a, b);
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Cosine distance: `1.0 - cosine_similarity`. Returns 1.0 for empty/mismatched.
pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    1.0 - cosine_similarity(a, b)
}

/// Euclidean (L2) distance between two vectors.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    euclidean_squared(a, b).sqrt()
}

/// Squared Euclidean distance — avoids sqrt for comparison-only use cases.
pub fn euclidean_squared(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Manhattan (L1) distance between two vectors.
pub fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Dot product of two vectors.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Vector Operations ─────────────────────────────────────────────────────────

/// Normalize a vector to unit length. Returns zero vector if input is zero or empty.
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let mag = magnitude(v);
    if mag == 0.0 {
        return vec![0.0; v.len()]
    }
    v.iter().map(|x| x / mag).collect()
}

/// Magnitude (L2 norm) of a vector.
pub fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Element-wise addition of two vectors. Returns empty if lengths differ.
pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != b.len() {
        return Vec::new();
    }
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Element-wise subtraction of two vectors. Returns empty if lengths differ.
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != b.len() {
        return Vec::new();
    }
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Scale a vector by a scalar.
pub fn scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

/// Linear interpolation between two vectors at parameter `t`.
///
/// At t=0 returns a, at t=1 returns b, at t=0.5 returns the midpoint.
/// Returns empty if lengths differ.
pub fn lerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    if a.len() != b.len() {
        return Vec::new();
    }
    a.iter().zip(b.iter()).map(|(x, y)| x + t * (y - x)).collect()
}

/// Weighted average of multiple vectors. Weights are normalized internally.
///
/// Returns empty if any vector has a different length, or if inputs are empty.
pub fn weighted_average(vectors: &[&[f64]], weights: &[f64]) -> Vec<f64> {
    if vectors.is_empty() || weights.is_empty() || vectors.len() != weights.len() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
        return Vec::new();
    }
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return vec![0.0; dim];
    }
    let mut result = vec![0.0; dim];
    for (v, &w) in vectors.iter().zip(weights.iter()) {
        for (r, &val) in result.iter_mut().zip(v.iter()) {
            *r += val * w / total_weight;
        }
    }
    result
}

// ── Statistics ─────────────────────────────────────────────────────────────────

/// Arithmetic mean. Returns 0.0 for empty.
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Variance (population). Returns 0.0 for empty.
pub fn variance(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let m = mean(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64
}

/// Standard deviation (population). Returns 0.0 for empty.
pub fn stddev(v: &[f64]) -> f64 {
    variance(v).sqrt()
}

/// Softmax: exponentiate and normalize so results sum to 1.0.
///
/// Uses the max-subtraction trick for numerical stability.
/// Returns empty vec for empty input.
pub fn softmax(v: &[f64]) -> Vec<f64> {
    if v.is_empty() {
        return Vec::new();
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; v.len()];
    }
    exps.iter().map(|x| x / sum).collect()
}

// ── Mono Operations ───────────────────────────────────────────────────────────

/// Blend two scalar values: at ratio=0 returns a, at ratio=1 returns b.
pub fn mono_blend(a: f64, b: f64, ratio: f64) -> f64 {
    a + ratio * (b - a)
}

/// Diffuse a mono value toward the weighted average of its neighbors.
///
/// Returns the original `vibe` if inputs are empty or lengths mismatch.
pub fn mono_diffuse(vibe: f64, neighbors: &[f64], weights: &[f64], rate: f64) -> f64 {
    if neighbors.is_empty() || neighbors.len() != weights.len() {
        return vibe;
    }
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return vibe;
    }
    let weighted_avg: f64 = neighbors.iter().zip(weights.iter()).map(|(n, w)| n * w).sum::<f64>() / total_weight;
    vibe + rate * (weighted_avg - vibe)
}

/// Surprise: absolute difference between predicted and actual.
pub fn mono_surprise(predicted: f64, actual: f64) -> f64 {
    (predicted - actual).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const TOLERANCE: f64 = 1e-10;

    // ── Cosine Similarity ─────────────────────────────────────────────────────

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < TOLERANCE);
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < TOLERANCE);
    }

    // ── Euclidean Distance ────────────────────────────────────────────────────

    #[test]
    fn euclidean_distance_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < TOLERANCE);
    }

    #[test]
    fn euclidean_squared_avoids_sqrt() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let sq = euclidean_squared(&a, &b);
        assert!((sq - 25.0).abs() < TOLERANCE);
    }

    // ── Manhattan Distance ────────────────────────────────────────────────────

    #[test]
    fn manhattan_distance_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        let dist = manhattan_distance(&a, &b);
        assert!((dist - 7.0).abs() < TOLERANCE);
    }

    // ── Dot Product ───────────────────────────────────────────────────────────

    #[test]
    fn dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b);
        assert!((dot - 32.0).abs() < TOLERANCE);
    }

    // ── Normalize ─────────────────────────────────────────────────────────────

    #[test]
    fn normalize_gives_unit_vector() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        let mag = magnitude(&n);
        assert!((mag - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn magnitude_of_normalized_is_one() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = normalize(&v);
        assert!((magnitude(&n) - 1.0).abs() < TOLERANCE);
    }

    // ── Vector Arithmetic ─────────────────────────────────────────────────────

    #[test]
    fn add_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = add(&a, &b);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn sub_vectors() {
        let a = vec![4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = sub(&a, &b);
        assert_eq!(result, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn scale_vector() {
        let v = vec![1.0, 2.0, 3.0];
        let result = scale(&v, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    // ── Lerp ───────────────────────────────────────────────────────────────────

    #[test]
    fn lerp_at_zero_is_a() {
        let a = vec![1.0, 2.0];
        let b = vec![5.0, 6.0];
        let result = lerp(&a, &b, 0.0);
        assert!((result[0] - 1.0).abs() < TOLERANCE);
        assert!((result[1] - 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn lerp_at_one_is_b() {
        let a = vec![1.0, 2.0];
        let b = vec![5.0, 6.0];
        let result = lerp(&a, &b, 1.0);
        assert!((result[0] - 5.0).abs() < TOLERANCE);
        assert!((result[1] - 6.0).abs() < TOLERANCE);
    }

    #[test]
    fn lerp_at_half_is_midpoint() {
        let a = vec![0.0, 0.0];
        let b = vec![10.0, 10.0];
        let result = lerp(&a, &b, 0.5);
        assert!((result[0] - 5.0).abs() < TOLERANCE);
        assert!((result[1] - 5.0).abs() < TOLERANCE);
    }

    // ── Weighted Average ──────────────────────────────────────────────────────

    #[test]
    fn weighted_average_equal_weights() {
        let a = vec![2.0, 4.0];
        let b = vec![4.0, 8.0];
        let result = weighted_average(&[&a, &b], &[1.0, 1.0]);
        assert!((result[0] - 3.0).abs() < TOLERANCE);
        assert!((result[1] - 6.0).abs() < TOLERANCE);
    }

    #[test]
    fn weighted_average_unequal_weights() {
        let a = vec![0.0];
        let b = vec![10.0];
        let result = weighted_average(&[&a, &b], &[0.25, 0.75]);
        assert!((result[0] - 7.5).abs() < TOLERANCE);
    }

    // ── Statistics ─────────────────────────────────────────────────────────────

    #[test]
    fn mean_basic() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((mean(&v) - 2.0).abs() < TOLERANCE);
    }

    #[test]
    fn variance_basic() {
        let v = vec![1.0, 2.0, 3.0];
        // population variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        assert!((variance(&v) - 2.0 / 3.0).abs() < TOLERANCE);
    }

    #[test]
    fn stddev_basic() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((stddev(&v) - (2.0_f64 / 3.0).sqrt()).abs() < TOLERANCE);
    }

    #[test]
    fn softmax_sums_to_one() {
        let v = vec![1.0, 2.0, 3.0];
        let s = softmax(&v);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < TOLERANCE);
        // Should be monotonically increasing
        assert!(s[0] < s[1]);
        assert!(s[1] < s[2]);
    }

    // ── Mono Operations ───────────────────────────────────────────────────────

    #[test]
    fn mono_blend_at_zero() {
        assert!((mono_blend(10.0, 20.0, 0.0) - 10.0).abs() < TOLERANCE);
    }

    #[test]
    fn mono_blend_at_one() {
        assert!((mono_blend(10.0, 20.0, 1.0) - 20.0).abs() < TOLERANCE);
    }

    #[test]
    fn mono_diffuse_moves_toward_weighted_avg() {
        let vibe = 1.0;
        let neighbors = vec![5.0, 7.0];
        let weights = vec![1.0, 1.0];
        let result = mono_diffuse(vibe, &neighbors, &weights, 0.5);
        let expected = 1.0 + 0.5 * (6.0 - 1.0); // 3.5
        assert!((result - expected).abs() < TOLERANCE);
    }

    #[test]
    fn mono_surprise_basic() {
        assert!((mono_surprise(3.0, 5.0) - 2.0).abs() < TOLERANCE);
        assert!((mono_surprise(5.0, 3.0) - 2.0).abs() < TOLERANCE);
    }

    // ── Edge Cases ─────────────────────────────────────────────────────────────

    #[test]
    fn empty_vector_handling() {
        let empty: Vec<f64> = vec![];
        assert!((cosine_similarity(&empty, &empty)).abs() < TOLERANCE);
        assert!((euclidean_distance(&empty, &empty)).abs() < TOLERANCE);
        assert!((manhattan_distance(&empty, &empty)).abs() < TOLERANCE);
        assert!((dot_product(&empty, &empty)).abs() < TOLERANCE);
        assert!(normalize(&empty).is_empty());
        assert!((magnitude(&empty)).abs() < TOLERANCE);
        assert!(add(&empty, &empty).is_empty());
        assert!(sub(&empty, &empty).is_empty());
        assert!(softmax(&empty).is_empty());
        assert!((mean(&empty)).abs() < TOLERANCE);
        assert!((variance(&empty)).abs() < TOLERANCE);
        assert!((stddev(&empty)).abs() < TOLERANCE);
    }

    #[test]
    fn mismatched_length_handling() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &b)).abs() < TOLERANCE);
        assert!((euclidean_distance(&a, &b)).abs() < TOLERANCE);
        assert!((manhattan_distance(&a, &b)).abs() < TOLERANCE);
        assert!((dot_product(&a, &b)).abs() < TOLERANCE);
        assert!(add(&a, &b).is_empty());
        assert!(sub(&a, &b).is_empty());
        assert!(lerp(&a, &b, 0.5).is_empty());
    }

    #[test]
    fn zero_vector_normalize_no_crash() {
        let zero = vec![0.0, 0.0, 0.0];
        let n = normalize(&zero);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn large_vector_performance() {
        let size = 10_000;
        let a: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
        let b: Vec<f64> = (0..size).map(|i| (i as f64).cos()).collect();

        let start = Instant::now();
        for _ in 0..100 {
            let _ = cosine_similarity(&a, &b);
            let _ = euclidean_distance(&a, &b);
            let _ = dot_product(&a, &b);
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 1000, "100 iterations of 3 ops on 10K vectors took {:?}", elapsed);
    }
}
