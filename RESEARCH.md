# NVFP4 Group GEMM - PyTorch Optimization Research

> **IMPORTANT**: All submissions must be a **single .py file** (`submission.py`)

---

## Table of Contents
1. [Problem Specification](#1-problem-specification)
2. [torch._scaled_mm API](#2-torch_scaled_mm-api)
3. [Scale Factor Layouts](#3-scale-factor-layouts)
4. [PyTorch CUDA Optimization](#4-pytorch-cuda-optimization)
5. [Existing Implementations](#5-existing-implementations)
6. [Optimization Strategy](#6-optimization-strategy)

---

## 1. Problem Specification

### Input Format
```python
input_t = (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)

# abc_tensors: list of (a, b, c) tuples
#   a: torch.Tensor[float4e2m1fn_x2] shape [M, K // 2, L]
#   b: torch.Tensor[float4e2m1fn_x2] shape [N, K // 2, L]
#   c: torch.Tensor[float16] shape [M, N, L]

# sfasfb_tensors: list of (sfa, sfb) tuples (row-major scales)
#   sfa: torch.Tensor[float8_e4m3fn] shape [M, K // 16, L]
#   sfb: torch.Tensor[float8_e4m3fn] shape [N, K // 16, L]

# sfasfb_reordered_tensors: list of (sfa_reordered, sfb_reordered) tuples
#   PRE-REORDERED in cuBLAS blocked format!
#   Shape: (32, 4, rest_m, 4, rest_k, L)

# problem_sizes: list of (M, N, K, L) tuples
#   L is always 1
```

### Benchmark Targets (Speed-of-Light)
| Groups | M range | N | K | Target |
|--------|---------|---|---|--------|
| 8 | 64-248 | 4096 | 7168 | 18.833μs |
| 8 | 40-196 | 7168 | 2048 | 10.667μs |
| 2 | 192-320 | 3072 | 4096 | 2.406μs |
| 2 | 128-384 | 4096 | 1536 | 1.525μs |

### Current Baseline Issue
Your current code has these problems:
1. **Loops over L dimension** - `for l_idx in range(l)` launches separate kernels
2. **Runtime scale conversion** - `to_blocked()` called at inference time
3. **No group batching** - Each group processed separately

---

## 2. torch._scaled_mm API

### Function Signature
```python
torch._scaled_mm(
    self: torch.Tensor,           # Shape (M, K), row-major, FP8/FP4 dtype
    mat2: torch.Tensor,           # Shape (K, N), COLUMN-MAJOR required!
    *,
    scale_a: torch.Tensor,        # Required - scale for self
    scale_b: torch.Tensor,        # Required - scale for mat2
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False
) -> torch.Tensor
```

### Critical Requirements

**1. mat2 MUST be column-major:**
```python
# Convert row-major to column-major:
mat2_col_major = mat2.T.contiguous().T  # Same shape, different strides

# Verify: stride should be (1, K) not (K, 1)
assert mat2_col_major.stride() == (1, mat2_col_major.shape[0])
```

**2. Scale Factor Formats:**

| Format | Data dtype | Scale dtype | Block Size |
|--------|-----------|-------------|------------|
| NVFP4 | `torch.float4_e2m1fn_x2` | `torch.float8_e4m3fn` | 16 elements |
| MXFP8 | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` | 32 elements |

**3. use_fast_accum for Performance:**
```python
# Recommended for inference - 1.2-1.5x faster
result = torch._scaled_mm(..., use_fast_accum=True)
```

### Block Scaling Layout for torch._scaled_mm
Scales must be in **32x16 blocked tiles** (swizzled format):

```python
def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert row-major scales to cuBLAS blocked format.
    Shape: (H, W) -> flattened (32*ceil(H/128) * 16*ceil(W/4),)
    """
    rows, cols = input_matrix.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if padded_rows != rows or padded_cols != cols:
        padded = torch.nn.functional.pad(
            input_matrix,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant", value=0,
        )
    else:
        padded = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()
```

---

## 3. Scale Factor Layouts

### cuBLAS 3D Block-Scaled Layout

The official NVIDIA layout transforms `(H, W)` row-major to blocked format:

```
Transform steps:
(H, W) -> pad to (128*n_row, 4*n_col)
       -> view(n_row, 128, n_col, 4)
       -> permute(0,2,1,3)
       -> reshape(-1, 4, 32, 4)
       -> transpose(1,2)
       -> reshape(-1, 32, 16)
       -> flatten()
```

### Pre-Reordered Tensors (sfasfb_reordered_tensors)

**KEY OPTIMIZATION**: The input already provides pre-reordered scales!

```python
# Shape: (32, 4, rest_m, 4, rest_k, l)
# Where:
#   32: Atom M dimension
#   4: Atom M tile count (32*4 = 128 rows per block)
#   rest_m: Number of 128-row blocks = ceil_div(M, 128)
#   4: Atom K dimension
#   rest_k: Number of K blocks = ceil_div(K // 16, 4)
#   l: Batch dimension (always 1)
```

**Using Pre-Reordered Scales Directly:**
```python
def get_blocked_scale_from_reordered(sfa_reordered, m, k, sf_vec_size=16):
    """Convert 6D pre-reordered tensor to flattened blocked format."""
    # sfa_reordered shape: (32, 4, rest_m, 4, rest_k, l)
    # Need to flatten to match to_blocked() output format

    # Permute to match blocked memory layout
    # Target: (rest_m, rest_k, 32, 16) -> flatten
    permuted = sfa_reordered.permute(2, 4, 0, 1, 3, 5)  # (rest_m, rest_k, 32, 4, 4, l)
    reshaped = permuted.reshape(-1, 32, 16)  # Combine 4*4 -> 16
    return reshaped.flatten()
```

---

## 4. PyTorch CUDA Optimization

### CUDA Graphs for Reduced Launch Overhead
```python
# Capture kernel sequence once
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)

# Replay with minimal overhead
def inference(new_input):
    static_input.copy_(new_input)
    g.replay()
    return output.clone()
```

### torch.compile with max-autotune
```python
@torch.compile(mode="max-autotune")
def optimized_gemm(a, b, scale_a, scale_b):
    return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b,
                           out_dtype=torch.float16, use_fast_accum=True)
```

### Avoiding Python Overhead
```python
# BAD: Python loop per group
for (a, b, c), (sfa, sfb) in zip(abc_tensors, sfasfb_tensors):
    result = torch._scaled_mm(...)  # Multiple kernel launches

# BETTER: Process all groups in batched operations where possible
```

### Memory Pre-allocation
```python
# Pre-allocate output buffers
output_buffers = [c for (_, _, c) in abc_tensors]  # Reuse provided C tensors

# Configure CUDA allocator
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
```

### Accurate Benchmarking
```python
def benchmark(fn, *args, warmup=10, iterations=100):
    for _ in range(warmup):
        fn(*args)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        fn(*args)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)
```

### inference_mode for Maximum Performance
```python
@torch.inference_mode()
def fast_inference(data):
    # Disables all autograd overhead
    return custom_kernel(data)
```

---

## 5. Existing Implementations

### DeepSeek DeepGEMM (FP8)
- 1550 TFLOPS on H800
- Fine-grained scaling with JIT compilation
- Grouped GEMM variants for MoE

### vLLM NVFP4
- Multiple backend paths (CUTLASS, Marlin, DeepGEMM)
- Scale interleaving for hardware backends
- Padding alignment: 128-256 bytes

### SGLang MoE
- Fused MoE with Triton kernels
- DeepGEMM wrapper integration
- Masked grouped GEMM for inference

### Key Takeaway
Pure PyTorch can achieve **80-90%** of custom CUDA performance with:
- `torch._scaled_mm` for FP8/FP4 GEMM
- `torch.compile(mode="max-autotune")`
- Proper scale factor pre-processing

---

## 6. Optimization Strategy

### Immediate Optimizations (No CuTe needed)

**1. Use Pre-Reordered Scales**
```python
# BEFORE (slow - converts at runtime):
scale_a = to_blocked(sfa[:, :, l_idx]).cuda()

# AFTER (fast - use pre-reordered):
scale_a = get_blocked_from_reordered(sfa_reordered)
```

**2. Eliminate L-dimension Loop**
```python
# L is always 1, so remove the loop entirely
# Process directly: a[:, :, 0], not a[:, :, l_idx]
```

**3. Ensure Column-Major B Matrix**
```python
# Current: b[:, :, l_idx].transpose(0, 1)
# Verify this produces column-major layout
b_transposed = b[:, :, 0].transpose(0, 1).contiguous()
# Should have stride (1, N) for column-major
```

**4. Add use_fast_accum**
```python
result = torch._scaled_mm(
    a_view,
    b_view,
    scale_a,
    scale_b,
    bias=None,
    out_dtype=torch.float16,
    use_fast_accum=True  # ADD THIS
)
```

**5. Process All Groups Efficiently**
```python
# Consider if groups can share any computation
# Pre-allocate all scale tensors before the loop
# Use inference_mode decorator
```

### Code Template
```python
import torch
from task import input_t, output_t

sf_vec_size = 16

def ceil_div(a, b):
    return (a + b - 1) // b

def get_blocked_scale(sfa_reordered):
    """Convert pre-reordered 6D tensor to blocked format for _scaled_mm."""
    # sfa_reordered: (32, 4, rest_m, 4, rest_k, l)
    # Permute and reshape to (n_blocks, 32, 16) then flatten
    permuted = sfa_reordered.permute(2, 4, 0, 1, 3, 5)  # (rest_m, rest_k, 32, 4, 4, l)
    l = sfa_reordered.shape[-1]
    rest_m = sfa_reordered.shape[2]
    rest_k = sfa_reordered.shape[4]
    # Flatten the inner dimensions
    reshaped = permuted.reshape(rest_m * rest_k, 32, 16)
    return reshaped.flatten()

@torch.inference_mode()
def custom_kernel(data: input_t) -> output_t:
    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data

    result_tensors = []

    for (a, b, c), (sfa_reordered, sfb_reordered), (m, n, k, l) in zip(
        abc_tensors, sfasfb_reordered_tensors, problem_sizes
    ):
        # L is always 1 - no loop needed
        # Use pre-reordered scales directly
        scale_a = get_blocked_scale(sfa_reordered)
        scale_b = get_blocked_scale(sfb_reordered)

        # Get FP4 views
        a_fp4 = a[:, :, 0].view(torch.float4_e2m1fn_x2)
        b_fp4 = b[:, :, 0].transpose(0, 1).contiguous().view(torch.float4_e2m1fn_x2)

        # Execute with fast accumulation
        result = torch._scaled_mm(
            a_fp4,
            b_fp4,
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16,
            use_fast_accum=True
        )

        c[:, :, 0] = result
        result_tensors.append(c)

    return result_tensors

def solve(data: input_t) -> output_t:
    return custom_kernel(data)
```

---

## References

- [GPU-Mode Reference Kernels](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia/nvfp4_group_gemm)
- [cuBLAS Block Scaling Layout](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout)
- [TorchAO mx_formats](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/utils.py)
- [gau-nernst tcgen05 blog](https://gau-nernst.github.io/tcgen05/)
- [DeepSeek DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [PyTorch FP8 Documentation](https://docs.pytorch.org/ao/stable/quantization_overview.html)
