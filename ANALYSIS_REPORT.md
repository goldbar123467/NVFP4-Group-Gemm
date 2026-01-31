# NVFP4 Group GEMM Analysis Report

**Version:** v7-clean-20260124
**Date:** 2026-01-31
**File:** `good_submission.py` (backup created)

---

## Executive Summary

This is a CuTe DSL implementation of NVFP4 Block-Scaled Group GEMM targeting NVIDIA B200 (Blackwell) GPUs. The kernel uses 4-bit floating point (FP4) data with FP8 scale factors for efficient matrix multiplication.

**Current Status:** Functional but significantly slower than speed-of-light targets (~20-100x gap).

---

## Architecture Overview

### Data Types
| Component | Type | Description |
|-----------|------|-------------|
| A, B matrices | `Float4E2M1FN` | 4-bit FP (2-bit exponent, 1-bit mantissa) |
| Scale factors | `Float8E4M3FN` | 8-bit FP for block scaling |
| Output C | `Float16` | Half precision output |
| Accumulator | `Float32` | Full precision for accuracy |

### Kernel Configuration
```python
mma_tiler_mnk = (128, 128, 256)  # MMA tile dimensions
mma_inst_shape_k = 64            # K dimension of MMA instruction
threads_per_cta = 128            # Threads per CTA
num_ab_stage = 1                 # Pipeline stages (1 is optimal!)
num_tmem_alloc_cols = 512        # Tensor memory columns
sf_vec_size = 16                 # Scale factor vector size
```

### Key Finding: Single-Stage is Faster
> **CRITICAL:** 3-stage pipelining was tested and is **30% SLOWER** (488µs vs 373µs baseline). The current 1-stage configuration is correct.

---

## Performance Benchmarks

### Current Results vs Targets

| Test Case | Groups | Dimensions | Current | Target | Gap |
|-----------|--------|------------|---------|--------|-----|
| Large K | 8 | M=64-248, N=4096, K=7168 | 433µs | 18.8µs | **23x** |
| Wide N | 8 | M=40-196, N=7168, K=2048 | 440µs | 10.7µs | **41x** |
| Small groups | 2 | M=192-320, N=3072, K=4096 | 191µs | 2.4µs | **80x** |
| Smallest | 2 | M=128-384, N=4096, K=1536 | 166µs | 1.5µs | **110x** |

### Variance Analysis
- Variance is low (0.5-0.9µs stddev) indicating consistent execution
- Best-case times are only ~5% better than average
- No outlier slowdowns detected

---

## Code Structure

### Entry Points
```
solve() → custom_kernel() → compile_kernel() → my_kernel → kernel
```

### Two Input Formats Supported

1. **GROUP GEMM format** (4-tuple): `(abc_tensors, _, sfasfb_reordered_tensors, problem_sizes)`
   - Used by gpumode evaluation
   - All groups in single kernel launch

2. **TASK format** (10-tuple): `(a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c)`
   - Local testing with dual GEMM + SiLU fusion
   - Computes: `C = silu(A @ B1) * (A @ B2)`

### Key Components

| Component | Purpose |
|-----------|---------|
| `kernel()` | GPU device kernel with TMA loads, MMA execution, epilogue |
| `my_kernel()` | JIT wrapper that sets up layouts and launches kernel |
| `compile_kernel()` | Caches compiled kernels by group count |
| `TensorMapManager` | Handles TMA descriptor updates per-group |

---

## Memory Hierarchy

### Global → Shared Memory (TMA)
- Uses `CopyBulkTensorTileG2SOp` for async TMA copies
- 4 tensor maps: A, B, SFA, SFB (128 bytes each)
- Scale factors use `internal_type=Int16` for TMA

### Shared → Tensor Memory (S2T)
- `Cp4x32x128bOp` for scale factor copies
- Accumulator in tensor memory (Float32)

### Tensor Memory → Registers → Global (Epilogue)
- `Ld32x32bOp` with x128 repetition
- Predicated stores for boundary handling

---

## Block Scheduling

### Group-to-CTA Mapping
```
bidz → (group_idx, coord_x, coord_y)

total_num_clusters = Σ ceil(M_i/128) × ceil(N_i/128)
```

The kernel linearizes all tiles from all groups into bidz, then delinearizes at runtime by iterating through `cta_mn_list`.

### Scale Factor Layout
```python
atom_shape = ((32, 4), (sf_vec_size, 4))
atom_stride = ((16, 4), (0, 1))
```
Tiled to full tensor shape using `tile_to_shape()`.

---

## Optimization Opportunities

### High Priority

1. **Persistent Kernel** - Current implementation launches one CTA per tile. A persistent kernel could reduce launch overhead significantly for small groups.

2. **Cluster Utilization** - Currently `cluster=(1,1,1)`. B200 supports larger clusters for better L2 locality.

3. **Warp Specialization** - Only warp 0 does TMA loads. Consider producer/consumer warp split.

4. **K-dimension Parallelism** - Split-K could help for large K values (7168).

### Medium Priority

5. **Epilogue Fusion** - The SiLU fusion path does two separate kernel launches + PyTorch ops. Could fuse into single kernel.

6. **Tensor Map Caching** - TMA descriptors are rebuilt per-CTA. Could pre-compute for common sizes.

7. **Register Pressure** - Check if reducing `num_tmem_alloc_cols` from 512 helps.

### Low Priority (Investigated, Not Effective)

8. ~~Multi-stage pipelining~~ - Already tested, made things worse
9. ~~Larger MMA tiles~~ - 128x128 is minimum for NVFP4 MMA

---

## Code Quality Assessment

### Strengths
- Clean separation of host/device code
- Proper kernel caching by group count
- Handles boundary conditions with predicates
- Good use of CuTe abstractions (TiledMMA, layouts, partitions)

### Issues
- Dual GEMM path allocates temp buffers per call
- No stream support for concurrent execution
- `ceil_div` function defined but uses `cute.ceil_div` internally

---

## Recommendations

1. **Profile with Nsight Compute** to identify actual bottleneck (memory vs compute bound)

2. **Test persistent kernel** approach for small group counts (2 groups)

3. **Investigate cluster launch** with `cluster=(2,1,1)` or larger

4. **Pre-allocate temp buffers** for dual GEMM path

5. **Consider CUTLASS 3.x grouped GEMM** reference for comparison

---

## Files

| File | Description |
|------|-------------|
| `submission.py` | Current working version |
| `good_submission.py` | Backup of v7-clean-20260124 |
| `RESEARCH.md` | Research notes and optimization strategies |
| `ANALYSIS_REPORT.md` | This report |

---

## RAG Brain Entries (Pending)

The following learnings should be stored when RAG Brain is available:

1. **Kernel Config Pattern** - mma_tiler, stages, threads settings
2. **Performance Benchmarks** - Current vs target times
3. **Group GEMM Architecture** - CTA scheduling, TMA patterns
4. **Scale Factor Layout** - Blocked format details
5. **Compilation Caching** - Cache key strategy

---

*Generated by analysis of v7-clean-20260124*
