# External References for NVFP4 Group GEMM

Curated external resources for Blackwell GPU kernel optimization. These are proven, high-quality references from experienced CUDA developers.

---

## Primary Author: Thien Tran (gau-nernst)

Thien achieved **98% cuBLAS performance** on Blackwell using the techniques documented here. His work is the gold standard for tcgen05 optimization.

---

## Repositories

### 1. learn-cuda (Tutorial Series)
**URL:** https://github.com/gau-nernst/learn-cuda

Progressive CUDA learning from basics to Blackwell tensor cores.

| Directory | Description | Relevance |
|-----------|-------------|-----------|
| `02e_matmul_sm100/` | **Blackwell tcgen05 matmul** | Direct reference for our kernel |
| `02d_matmul_cute/` | CuTe DSL patterns | Layout, copy, MMA patterns |
| `02c_matmul_wmma/` | WMMA tensor cores | Historical context |

**Key Files to Study:**
```
learn-cuda/
├── 02e_matmul_sm100/
│   ├── v1_basic.cu          # Starting point
│   ├── v2_swizzle.cu        # +128-byte swizzling (2.7x)
│   ├── v3_pipeline.cu       # +pipelining (1.35x)
│   ├── v4_warp_spec.cu      # +warp specialization (1.29x)
│   ├── v5_2sm.cu            # +cluster launch (1.08x)
│   └── v6_persistent.cu     # +persistent kernel (1.13x)
└── 02d_matmul_cute/
    └── *.cu                 # CuTe layout examples
```

### 2. gn-kernels (Production Kernels)
**URL:** https://github.com/gau-nernst/gn-kernels

Production-quality CUTLASS kernels including NVFP4 support.

| Component | Description |
|-----------|-------------|
| NVFP4 GEMM | Block-scaled FP4 matrix multiplication |
| FP8 GEMM | Reference for scale factor handling |
| Fused kernels | Epilogue fusion patterns |

### 3. gpu-mode-kernels (Competition Solutions)
**URL:** https://github.com/gau-nernst/gpu-mode-kernels

Solutions to GPU-Mode competition problems.

**Notable:** Achieved **71x speedup** on optimization challenges.

---

## Blog Posts (Critical Reading)

### 1. tcgen05 for dummies
**URL:** https://gau-nernst.github.io/tcgen05/

**This is the most important reference for our kernel.**

| Section | Key Content |
|---------|-------------|
| Swizzling | 128-byte pattern, bank conflict elimination |
| Pipelining | mbarrier usage, N-stage implementation |
| Warp Specialization | Producer/consumer split |
| 2-SM MMA | Cluster launch, distributed smem |
| Persistent Kernels | Work-stealing, tile ordering |

**Performance progression chart shows exactly how to reach 98% cuBLAS.**

### 2. NVRTC matmul exploration
**URL:** https://gau-nernst.github.io/nvrtc-matmul/

Runtime compilation techniques and FP4/FP8 exploration.

| Topic | Takeaway |
|-------|----------|
| NVRTC | 1000x faster than nvcc for JIT |
| FP4 layouts | Block scaling patterns |
| Profiling | nsight-compute methodology |

### 3. Flash Attention 5090
**URL:** https://gau-nernst.github.io/fa-5090/

Flash Attention implementation on Blackwell.

| Topic | Applicability |
|-------|---------------|
| Swizzling patterns | Same as GEMM |
| ldmatrix | TMEM load patterns |
| Softmax fusion | Epilogue patterns |

---

## NVIDIA Official References

### CUTLASS

**Repo:** https://github.com/NVIDIA/cutlass

| Path | Description |
|------|-------------|
| `examples/cute/blackwell/` | Official Blackwell examples |
| `include/cute/arch/copy_sm100.hpp` | TMA copy traits |
| `include/cute/arch/mma_sm100.hpp` | tcgen05 MMA |
| `include/cutlass/gemm/collective/` | Collective patterns |

### cuBLAS Block Scaling

**URL:** https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

Documents the block-scaled layout our kernel must match:
- 1:32 scaling (32 elements per scale factor)
- Interleaved vs separate layouts
- FP4/FP8 combinations

### PTX ISA (tcgen05)

**URL:** https://docs.nvidia.com/cuda/parallel-thread-execution/

Search for "tcgen05" for:
- Instruction encoding
- Operand constraints
- Memory semantics

---

## GPU-Mode Resources

### Reference Kernels Repo
**URL:** https://github.com/gpu-mode/reference-kernels

| Path | Description |
|------|-------------|
| `problems/nvidia/nvfp4_group_gemm/` | Our challenge definition |
| `problems/nvidia/nvfp4_group_gemm/reference.py` | Reference implementation |

### Discord
GPU-Mode Discord has channels for:
- Blackwell optimization
- CUTLASS/CuTe help
- Competition discussion

---

## Quick Access URLs

```
# Highest priority (read these first)
https://gau-nernst.github.io/tcgen05/
https://github.com/gau-nernst/learn-cuda/tree/main/02e_matmul_sm100

# Production reference
https://github.com/gau-nernst/gn-kernels

# Official docs
https://github.com/NVIDIA/cutlass/tree/main/examples/cute/blackwell
https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

# Challenge definition
https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia/nvfp4_group_gemm
```

---

## Reading Order for New Agents

1. **`GAU_NERNST_TECHNIQUES.md`** (this repo) - Extracted optimization techniques
2. **tcgen05 blog post** - Full context and code
3. **learn-cuda v1-v6 files** - Progressive implementation
4. **CUTLASS Blackwell examples** - Official patterns
5. **gn-kernels** - Production code reference

---

## Contact / Community

- **Thien Tran (gau-nernst):** GitHub issues on his repos
- **GPU-Mode Discord:** Active community for GPU optimization
- **NVIDIA Developer Forums:** Official support

---

## Version History

| Date | Change |
|------|--------|
| 2026-01-31 | Initial creation with curated links |
