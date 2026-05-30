# plato-math

Shared vector math primitives for the PLATO ecosystem — distance functions, vector ops, statistics, and mono operations.

## Features

- **Zero dependencies** — pure Rust, no serde, no nothing
- **No panics** — all functions handle edge cases (empty vectors, mismatched lengths) gracefully
- **Comprehensive** — 20+ functions covering everything the PLATO crates need

## API

### Distance Functions
- `cosine_similarity(a, b)` → `[-1.0, 1.0]`
- `cosine_distance(a, b)` → `[0.0, 2.0]`
- `euclidean_distance(a, b)`
- `euclidean_squared(a, b)` — avoids sqrt
- `manhattan_distance(a, b)`
- `dot_product(a, b)`

### Vector Operations
- `normalize(v)` — unit vector
- `magnitude(v)` — L2 norm
- `add(a, b)` / `sub(a, b)` — element-wise
- `scale(v, s)` — scalar multiply
- `lerp(a, b, t)` — linear interpolation
- `weighted_average(vectors, weights)` — normalized weighted mean

### Statistics
- `mean(v)` / `variance(v)` / `stddev(v)` — population
- `softmax(v)` — numerically stable

### Mono Operations
- `mono_blend(a, b, ratio)` — scalar lerp
- `mono_diffuse(vibe, neighbors, weights, rate)` — diffusion step
- `mono_surprise(predicted, actual)` — absolute difference

## Usage

```rust
use plato_math::*;

let a = vec![1.0, 2.0, 3.0];
let b = vec![4.0, 5.0, 6.0];

let sim = cosine_similarity(&a, &b);
let dist = euclidean_distance(&a, &b);
let blended = lerp(&a, &b, 0.5);
```

## License

MIT
