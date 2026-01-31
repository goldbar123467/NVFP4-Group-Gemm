# Gau-Nernst Optimization Techniques for Blackwell GEMM

> **Source:** [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) - Achieved **98% cuBLAS performance** on RTX 5090
> **Author:** Thien Tran (gau-nernst)

This document extracts proven optimization techniques from gau-nernst's tcgen05 blog post and repositories. These techniques are directly applicable to NVFP4 Group GEMM on Blackwell architecture.

---

## Performance Progression (Proven Path to 98% cuBLAS)

| Version | Technique Added | TFLOPS | % cuBLAS | Speedup |
|---------|----------------|--------|----------|---------|
| v1 | Basic tcgen05 | 254 | 17% | baseline |
| v2 | 128-byte swizzling | 695 | 46% | **2.7x** |
| v3 | Pipelining | 939 | 62% | 1.35x |
| v4 | Warp specialization | 1208 | 80% | 1.29x |
| v5 | 2-SM MMA | 1300 | 86% | 1.08x |
| v6 | Persistent kernel | 1475 | 98% | 1.13x |

**Key insight:** Swizzling alone provides the biggest single gain (2.7x). The progression is additive - each technique builds on previous ones.

---

## Technique 1: 128-Byte Swizzling (2.7x Speedup)

### Problem
Bank conflicts in shared memory kill performance. Naive layouts cause 32-way conflicts.

### Solution
Apply 128-byte swizzle pattern to shared memory layout.

### Code Pattern (CuTe)
```cpp
// Swizzle pattern: B=3, M=4, S=3 for 128-byte swizzle
using SmemSwizzle = cute::Swizzle<3, 4, 3>;

// Apply to shared memory layout
auto smem_layout = cute::make_layout(
    cute::make_shape(cute::Int<128>{}, cute::Int<256>{}),
    cute::make_stride(cute::Int<256>{}, cute::Int<1>{})
);
auto swizzled_layout = cute::composition(SmemSwizzle{}, smem_layout);
```

### Python/CUTLASS Pattern
```python
from cutlass.cute import Swizzle

# 128-byte swizzle for FP4 data
swizzle = Swizzle(B=3, M=4, S=3)
```

### Why It Works
- Eliminates shared memory bank conflicts
- 128-byte aligns with Blackwell's memory subsystem
- Works for both TMA loads and TMEM stores

### Implementation Notes
- Must match swizzle pattern in TMA descriptor
- Apply to both A and B matrix shared memory
- Swizzle B=3, M=4, S=3 is optimal for 128-byte alignment

---

## Technique 2: Pipelining (35% Speedup)

### Problem
Memory latency is hidden but only if next tile's data is already in flight.

### Solution
N-stage pipeline: while MMA computes tile N, TMA loads tile N+1 (or N+2, N+3...).

### Code Pattern
```cpp
// Pipeline structure
constexpr int STAGES = 3;  // 2-4 stages typical

// Producer: issue TMA loads ahead
for (int stage = 0; stage < STAGES; stage++) {
    tma_load(smem[stage], gmem[stage]);
}

// Main loop: overlap compute and memory
for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
    int load_stage = (k_tile + STAGES) % STAGES;
    int compute_stage = k_tile % STAGES;

    // Wait for this stage's data
    wait_mbarrier(mbar[compute_stage]);

    // Issue next load
    if (k_tile + STAGES < num_k_tiles) {
        tma_load(smem[load_stage], gmem[k_tile + STAGES]);
    }

    // Compute on current data
    mma(accum, smem_a[compute_stage], smem_b[compute_stage]);
}
```

### Memory Barriers (mbarrier)
```cpp
// Initialize mbarrier for expected bytes
cute::initialize_barrier(mbar[stage], expected_bytes);

// TMA signals completion to mbarrier
cute::tma_copy_async(tma_desc, smem, mbar[stage]);

// Wait for arrival
cute::wait_barrier(mbar[stage]);
```

### Implementation Notes
- **WARNING:** Our test showed 3-stage was 30% SLOWER. Reason unknown.
- Try 2-stage first, measure carefully
- Each stage needs its own mbarrier
- Memory usage: stages * (smem_A + smem_B)
- Balance: more stages = more hiding, but more memory pressure

---

## Technique 3: Warp Specialization (29% Speedup)

### Problem
Single warp doing TMA + MMA + epilogue has poor instruction mix.

### Solution
Dedicate warps to specific tasks. Producer warps do TMA, consumer warps do MMA.

### Code Pattern
```cpp
// Warp roles
constexpr int TMA_WARPS = 1;      // Producer: memory loads
constexpr int MMA_WARPS = 3;      // Consumer: compute
constexpr int EPILOGUE_WARPS = 1; // Store results

int warp_id = threadIdx.x / 32;

if (warp_id < TMA_WARPS) {
    // Producer: TMA loads only
    for (int k = 0; k < num_k_tiles; k++) {
        issue_tma_load(k);
        arrive_mbarrier(producer_mbar);
    }
} else if (warp_id < TMA_WARPS + MMA_WARPS) {
    // Consumer: MMA compute only
    for (int k = 0; k < num_k_tiles; k++) {
        wait_mbarrier(consumer_mbar);
        mma_compute(k);
    }
} else {
    // Epilogue: store results
    wait_for_mma_complete();
    store_results();
}
```

### Synchronization Pattern
- Producers signal via `arrive_mbarrier()` when data ready
- Consumers wait via `wait_mbarrier()` before computing
- Epilogue warps wait for all MMA completion

### Implementation Notes
- Common split: 1 TMA warp, 3 MMA warps for 128 threads
- MMA warps need good occupancy (multiple tiles per warp)
- Requires careful barrier management
- More complex but significant gains

---

## Technique 4: 2-SM MMA / Cluster Launch (8% Speedup)

### Problem
Single SM has limited resources; adjacent SMs sit idle.

### Solution
Use CUDA cluster launch to have 2+ SMs cooperate on same tile.

### Code Pattern
```cpp
// Cluster launch configuration
cudaLaunchConfig_t config = {};
config.gridDim = {grid_x, grid_y, grid_z};
config.blockDim = {128, 1, 1};

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {2, 1, 1};  // 2 SMs per cluster
config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, kernel, args...);
```

### CuTe Cluster Pattern
```cpp
// Distributed shared memory across cluster
extern __shared__ __align__(128) char smem[];
auto cluster_smem = cute::make_tensor(
    cute::make_smem_ptr(smem),
    cute::make_layout(shape, stride)
);

// Access neighbor SM's shared memory
int cluster_rank = cute::block_rank_in_cluster();
auto neighbor_smem = cluster_smem[1 - cluster_rank];
```

### Implementation Notes
- B200 supports clusters up to (8,4,4)
- Start with cluster=(2,1,1), measure gains
- Requires distributed shared memory layout
- Communication via cluster-level barriers

---

## Technique 5: Persistent Kernels (13% Speedup)

### Problem
Kernel launch overhead adds up; each CTA handles only one tile.

### Solution
Keep CTAs alive, have each process multiple tiles via work-stealing.

### Code Pattern
```cpp
// Grid = num_SMs, not num_tiles
__global__ void persistent_kernel(Params params) {
    __shared__ int tile_idx;

    while (true) {
        // Leader thread gets next tile
        if (threadIdx.x == 0) {
            tile_idx = atomicAdd(&params.next_tile, 1);
        }
        __syncthreads();

        // Check if done
        if (tile_idx >= params.total_tiles) break;

        // Delinearize tile index
        auto [m_idx, n_idx] = delinearize(tile_idx, params);

        // Process this tile
        process_tile(m_idx, n_idx, params);

        __syncthreads();
    }
}
```

### Work Distribution
```cpp
// Simple linear work-stealing
int next_tile = atomicAdd(global_counter, 1);

// Or tile-swizzled for better L2 locality
int swizzled = swizzle_tile_idx(next_tile, grid_dim);
```

### Implementation Notes
- Grid size = number of SMs (not tiles)
- Need global atomic counter for work distribution
- Consider tile ordering for L2 cache locality
- Reduces kernel launch overhead significantly

---

## Blackwell-Specific Hardware Notes

### TMA (Tensor Memory Accelerator)
- Async bulk copy from global to shared memory
- Handles address calculation, predication, conversion
- Used via `cute::Copy_Traits<SM100_TMA_LOAD>`

### TMEM (Tensor Memory)
- Fast register-like storage for MMA operands
- Accessed via S2T (shared-to-tensor) operations
- 512 columns per SM on Blackwell

### mbarrier (Memory Barrier)
- Hardware synchronization primitive
- Tracks expected vs arrived bytes
- Essential for pipelining

### tcgen05 MMA
- Blackwell's tensor core instruction
- Shape: typically 128x128x64 for FP4
- Operates on TMEM, not registers

---

## Applicability to NVFP4 Group GEMM

### Currently Implemented
- Basic tcgen05 MMA (v1 equivalent)
- Single-stage memory path
- Simple block scheduling

### Not Yet Implemented (Priority Order)
1. **128-byte swizzling** - Highest ROI (2.7x)
2. **2-stage pipelining** - Test carefully (our 3-stage was slower)
3. **Warp specialization** - Complex but 29% gain
4. **Persistent kernel** - Good for group GEMM with many small tiles
5. **Cluster launch** - Try cluster=(2,1,1) for free 8%

### Group GEMM Considerations
- Persistent kernel especially valuable: many groups = many tiles
- Tile ordering matters more with multiple groups
- Consider per-group work distribution

---

## Quick Reference: Expected Gains

| Starting Point | Add Technique | Expected Improvement |
|----------------|---------------|---------------------|
| Naive | Swizzling | 2-3x |
| Swizzled | Pipelining | 1.3-1.4x |
| Pipelined | Warp Spec | 1.2-1.3x |
| Warp Spec | 2-SM MMA | 1.05-1.1x |
| 2-SM | Persistent | 1.1-1.15x |
| **Total** | All techniques | **5-6x** |

**Note:** Gains are multiplicative. Swizzled + Pipelined + Warp Spec = 2.7 * 1.35 * 1.29 = **4.7x**

---

## References

- [tcgen05 for dummies (full blog post)](https://gau-nernst.github.io/tcgen05/)
- [learn-cuda repo: 02e_matmul_sm100](https://github.com/gau-nernst/learn-cuda/tree/main/02e_matmul_sm100)
- [gn-kernels NVFP4 implementations](https://github.com/gau-nernst/gn-kernels)
- [CUTLASS Blackwell examples](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/blackwell)
