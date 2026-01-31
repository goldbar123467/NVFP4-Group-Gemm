# NVFP4 Group GEMM Challenge Description

## Overview

Implement a **block scaled group matrix-matrix multiplication kernel** optimized for **NVIDIA B200**.

## Input Format

You will be given a tuple of tensors:

```python
(abc_tensors, sfasfb_tensors, problem_sizes)
```

Where:

### `abc_tensors` - List of tuples `(a, b, c)`

| Tensor | Data Type | Shape |
|--------|-----------|-------|
| `a` | `torch.float4e2m1fn_x2` | `[M, K // 2, L]` |
| `b` | `torch.float4e2m1fn_x2` | `[N, K // 2, L]` |
| `c` | `torch.float16` | `[M, N, L]` |

### `sfasfb_tensors` - List of tuples `(sfa, sfb)`

| Tensor | Data Type | Shape |
|--------|-----------|-------|
| `sfa` | `torch.float8_e4m3fnuz` | `[M, K // 16, L]` |
| `sfb` | `torch.float8_e4m3fnuz` | `[N, K // 16, L]` |

### `problem_sizes` - List of tuples `(M, N, K, L)`

Each group's matrix sizes:
- `M` is divisible by `mma_tiler_mn[0]`
- `N` is divisible by `mma_tiler_mn[1]`
- `K` is divisible by 256

## Ranking Criteria

**Geometric mean** of the benchmark results.

## Grand Prize

Your kernel will be evaluated against the **speed of light analysis**. The solution closest to speed of light wins the grand prize.

## Speed of Light Targets

Based on `max(FP4 Tensor Core math throughput, DRAM memory throughput)` of B200, tested under **1.5GHz clock** with average M, N, K values per group:

| Groups | M_values | N_values | K_values | L | Target Time |
|--------|----------|----------|----------|---|-------------|
| 8 | [80, 176, 128, 72, 64, 248, 96, 160] | [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096] | [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168] | 1 | **18.833 µs** |
| 8 | [40, 76, 168, 72, 164, 148, 196, 160] | [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168] | [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048] | 1 | **10.667 µs** |
| 2 | [192, 320] | [3072, 3072] | [4096, 4096] | 1 | **2.406 µs** |
| 2 | [128, 384] | [4096, 4096] | [1536, 1536] | 1 | **1.525 µs** |

## Key Constraints

1. **Raw CUDA only** - No Triton
2. **Single kernel optimization** - No dual-streaming GEMMs (considered cheating)
3. **All gains must come from kernel-level optimizations**

## Scale Factor Layout

The customized data layout for scale factors follows:
https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout

### Reordered Scale Factor Shape

```python
# Shape: (32, 4, rest_m, 4, rest_k, l)
# Where:
#   rest_m = ceil(M / 128)
#   rest_k = ceil(K / 16 / 4)
```

### Block Layout Parameters

```python
sf_vec_size = 16
atom_m = (32, 4)
atom_k = 4
```
