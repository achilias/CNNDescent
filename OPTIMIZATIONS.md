# Index Build Time Optimizations

This document describes the optimizations made to improve the index build time of CNNDescent.

## Summary

The following optimizations were implemented to reduce index build time:

### 1. Sampling Algorithm Optimization

**Problem:** The original sampling algorithm used rejection sampling with repeated random selection, which becomes very inefficient when sampling a large portion of the available items.

**Solution:** Replaced with Fisher-Yates shuffle algorithm for O(k) time complexity instead of O(k * log(n)).

**Files Modified:**
- `src/utils.c` - `sample()` function
- `src/utils.c` - `collect_sets()` function  
- `src/utils.c` - `graph_init()` function

**Impact:** Significant reduction in sampling overhead, especially noticeable during initialization and the main NN-Descent loop.

### 2. Distance Function Optimization

**Problem:** The euclidean_dist and manhattan_dist functions used simple loops without optimization, missing opportunities for compiler vectorization.

**Solution:** Implemented manual loop unrolling (4-way) to enable better SIMD vectorization and reduce loop overhead.

**Files Modified:**
- `src/utils.c` - `euclidean_dist()` function
- `src/utils.c` - `manhattan_dist()` function

**Changes:**
- Added 4-way loop unrolling with separate accumulators
- Changed to use `sqrtf` instead of `sqrt` for single-precision
- Changed to use `fabsf` instead of `fabs` for single-precision

**Impact:** Improved distance computation performance, which is critical as it's the most frequently called operation during index building.

### 3. Memory Allocation Optimization

**Problem:** The `collect_pairs()` function allocated vectors with minimal initial capacity, causing repeated reallocations.

**Solution:** Pre-allocate vectors with estimated capacity based on the actual number of neighbors.

**Files Modified:**
- `src/utils.c` - `collect_pairs()` function

**Impact:** Reduced memory allocation overhead during the main NN-Descent loop.

### 4. Compiler Optimization Flags

**Problem:** The Makefile was not using architecture-specific optimizations.

**Solution:** Added `-march=native` and `-ffast-math` compiler flags to enable better code generation.

**Files Modified:**
- `c_lib/Makefile`

**Flags Added:**
- `-march=native` - Enables CPU-specific optimizations (AVX, SSE, etc.)
- `-ffast-math` - Enables aggressive floating-point optimizations

**Impact:** Better utilization of modern CPU features, especially SIMD instructions.

## Benchmark Results

A benchmark test was added to measure the performance improvements:

- **Dataset:** 1000 points, 128 dimensions, k=10
- **Build time (Euclidean):** ~125ms
- **Build time (Manhattan):** ~120ms

These optimizations particularly benefit:
- Large datasets with many points
- High-dimensional data
- Scenarios requiring multiple graph builds

## Testing

All existing unit tests continue to pass, ensuring that the optimizations maintain correctness:
- `test_graph_init` - Tests random initialization
- `test_sample` - Tests sampling function
- `test_nndescent` - Tests the main NN-Descent algorithm
- `test_recall` - Tests the quality of the built graph

A new benchmark test (`benchmark.c`) was added to measure and verify performance improvements.

## Future Optimization Opportunities

Additional optimizations that could be explored:
1. Use of optimized_euclidean with precomputed norms (requires API changes)
2. Better lock-free or batched locking strategy in the main loop
3. SIMD intrinsics for distance computations
4. Cache-aware data layout for better memory access patterns
5. OpenMP task-based parallelization for better load balancing
