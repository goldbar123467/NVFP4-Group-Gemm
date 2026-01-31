# NVFP4 Group GEMM - Claude Context

## MANDATORY READING (Before Any Work)

**You MUST read these files before making any changes:**

1. **`CHALLENGE.md`** - Challenge description, input format, speed-of-light targets
2. **`REFERENCE_IMPL.md`** - Reference implementation that your kernel must match
3. **`GAU_NERNST_TECHNIQUES.md`** - Proven optimization path to 98% cuBLAS (2.7x-6x gains documented)
4. **`EXTERNAL_REFERENCES.md`** - Curated links to reference implementations and blog posts

These define the contract and proven optimization path. Do not deviate from the input/output format or validation tolerances.

### Key External Resource
**tcgen05 for dummies:** https://gau-nernst.github.io/tcgen05/
- Achieved 98% cuBLAS on Blackwell using techniques directly applicable to our kernel
- Performance progression: 254 TFLOPS (17%) → 1475 TFLOPS (98%)
- **Priority techniques:** Swizzling (2.7x) > Pipelining (1.35x) > Warp Spec (1.29x)

---

## Project Overview

NVFP4 Block-Scaled Group GEMM implementation for NVIDIA B200 (Blackwell) using CuTe DSL and CUTLASS. Competition/benchmark optimization challenge.

**Goal:** Achieve speed-of-light performance targets for grouped matrix multiplication with FP4 data and FP8 scale factors.

---

## Challenge Rules (IMPORTANT)

**Must use:**
- Raw CUDA (CUTLASS/CuTe DSL is fine - compiles to CUDA)
- Single-kernel optimizations only

**NOT allowed (considered cheating):**
- Triton (must be raw CUDA)
- Dual-streaming the GEMMs (running B1 and B2 GEMMs on separate CUDA streams)
- Any multi-stream parallelism to game the benchmark

**The performance targets must be achieved through kernel-level optimizations**, not by orchestrating multiple kernels in parallel. All gains must come from within the kernel itself.

---

## Current Status

**Version:** v7-clean-20260124
**Working file:** `submission.py`
**Backup:** `good_submission.py` (identical, safe copy)

### Performance (Current vs Target)

| Test | Groups | Dimensions | Current | Target | Gap |
|------|--------|------------|---------|--------|-----|
| 1 | 8 | M=64-248, N=4096, K=7168 | **433µs** | 18.8µs | 23x |
| 2 | 8 | M=40-196, N=7168, K=2048 | **440µs** | 10.7µs | 41x |
| 3 | 2 | M=192-320, N=3072, K=4096 | **191µs** | 2.4µs | 80x |
| 4 | 2 | M=128-384, N=4096, K=1536 | **166µs** | 1.5µs | 110x |

---

## Key Configuration (DO NOT CHANGE WITHOUT TESTING)

```python
mma_tiler_mnk = (128, 128, 256)  # MMA tile - minimum for NVFP4
mma_inst_shape_k = 64
threads_per_cta = 128
num_ab_stage = 1                 # DO NOT increase - 3-stage was 30% SLOWER
num_tmem_alloc_cols = 512
sf_vec_size = 16

# Data types
ab_dtype = cutlass.Float4E2M1FN  # 4-bit FP
sf_dtype = cutlass.Float8E4M3FN  # 8-bit FP scale factors
c_dtype = cutlass.Float16        # Output
```

---

## Architecture

### Input Format (GROUP GEMM - 4-tuple)
```python
data = (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)
# abc_tensors: list of (a, b, c) - FP4 matrices
# sfasfb_reordered_tensors: list of (sfa, sfb) - pre-permuted scale factors
# problem_sizes: list of (M, N, K, L) tuples
```

### Kernel Flow
```
solve() → custom_kernel() → compile_kernel() → my_kernel (JIT) → kernel (device)
```

### Block Scheduling
- All groups linearized into `bidz` dimension
- `total_num_clusters = Σ ceil(M_i/128) × ceil(N_i/128)`
- Delinearize at runtime to `(group_idx, coord_x, coord_y)`

### Memory Path
```
Global → TMA → Shared Memory → S2T → Tensor Memory → MMA → Registers → Global
```

---

## Tested Optimizations

| Change | Result | Keep? |
|--------|--------|-------|
| 3-stage pipelining | 30% SLOWER (488µs vs 373µs) | NO |
| 128x128 MMA tile | Required minimum for NVFP4 | YES |
| Kernel caching by group count | Works | YES |

---

## Optimization Opportunities (Priority Order from gau-nernst)

Based on proven results from tcgen05 blog (see `GAU_NERNST_TECHNIQUES.md`):

| Priority | Technique | Expected Gain | Status |
|----------|-----------|---------------|--------|
| 1 | **128-byte Swizzling** | 2.7x | NOT IMPLEMENTED |
| 2 | **2-stage Pipelining** | 1.35x | Test carefully (3-stage was slower) |
| 3 | **Warp Specialization** | 1.29x | NOT IMPLEMENTED |
| 4 | **Cluster Launch (2-SM)** | 1.08x | Currently cluster=(1,1,1) |
| 5 | **Persistent Kernel** | 1.13x | NOT IMPLEMENTED |

### Other Opportunities
- **Split-K** - For large K (7168), could parallelize K dimension
- **Epilogue Fusion** - Dual GEMM path does 2 launches + PyTorch ops

### Expected Total Improvement
Swizzling + Pipelining + Warp Spec = 2.7 × 1.35 × 1.29 = **4.7x**

All techniques combined: **5-6x improvement possible**

---

## Files

| File | Description |
|------|-------------|
| `CHALLENGE.md` | **READ FIRST** - Challenge description, targets, constraints |
| `REFERENCE_IMPL.md` | **READ FIRST** - Reference implementation to match |
| `GAU_NERNST_TECHNIQUES.md` | **READ FIRST** - Proven 98% cuBLAS optimization path |
| `EXTERNAL_REFERENCES.md` | Curated external links (repos, blogs, docs) |
| `submission.py` | Current working code - EDIT THIS |
| `good_submission.py` | Backup of v7-clean |
| `RESEARCH.md` | API docs, optimization strategies |
| `ANALYSIS_REPORT.md` | Detailed technical analysis |
| `RAG_BRAIN_IMPORT.json` | Learnings for RAG brain (import later) |

---

## Quick Commands

```bash
# Run benchmark (on GPU machine)
python -m task  # or however gpumode evaluates

# Check current times
grep -E "⏱|⚡|🐌" submission_results.txt
```

---

## References

### Primary (Proven 98% cuBLAS)
- [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) - **READ THIS FIRST** for optimization techniques
- [learn-cuda/02e_matmul_sm100](https://github.com/gau-nernst/learn-cuda/tree/main/02e_matmul_sm100) - Progressive implementation v1→v6
- [gn-kernels](https://github.com/gau-nernst/gn-kernels) - Production NVFP4 kernels

### Official Documentation
- [GPU-Mode Reference Kernels](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia/nvfp4_group_gemm)
- [cuBLAS Block Scaling Layout](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout)
- [CUTLASS Blackwell Examples](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/blackwell)

### Additional Blog Posts
- [NVRTC matmul](https://gau-nernst.github.io/nvrtc-matmul/) - FP4/FP8 exploration, JIT compilation
- [Flash Attention 5090](https://gau-nernst.github.io/fa-5090/) - Swizzling, ldmatrix patterns

---

## When Resuming

1. **Read `CHALLENGE.md`** - Understand the problem and targets
2. **Read `REFERENCE_IMPL.md`** - Understand the reference implementation
3. **Read `GAU_NERNST_TECHNIQUES.md`** - Understand proven optimization path
4. Read `submission.py` to understand current state
5. Check `RESEARCH.md` for API details
6. Try optimizations **in priority order** (swizzling first, then pipelining, etc.)
7. Benchmark and compare to current times above
8. If worse, revert to `good_submission.py`

**Key insight:** Don't guess at optimizations. Follow the proven path that achieved 98% cuBLAS.
