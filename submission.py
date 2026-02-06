"""
NVFP4 Block-Scaled Group GEMM for NVIDIA B200
True Warp-Specialized Implementation using CuTe DSL

VERSION: v10-gated-fixes
CHANGES: Gate 0 — correctness (cache key collision fix, dead subtraction removal),
         Gate 1 — proper kernel cache keying on full problem dimensions,
         Gate 2 — Triton fused SiLU kernel replacing torch.compile FP32 round-trip
NOTE:    Predicate axis order (residue_n, residue_m) preserved — see epilogue comments.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

import functools
from typing import Tuple, List

import torch
from task import input_t, output_t

# =============================================================================
# KERNEL CONFIGURATION
# =============================================================================

# Tensormap configuration
bytes_per_tensormap = 128
num_tensormaps = 4

# MMA tile dimensions (HARDWARE CONSTRAINT: 128x128 minimum for NVFP4)
mma_tiler_mnk = (128, 128, 256)
mma_inst_shape_k = 64

# Data types
ab_dtype = cutlass.Float4E2M1FN   # 4-bit floating point for A/B matrices
sf_dtype = cutlass.Float8E4M3FN   # 8-bit floating point for scale factors
c_dtype = cutlass.Float16         # 16-bit floating point for output

# Scale factor configuration
sf_vec_size = 16

# Thread block configuration
threads_per_cta = 128  # 4 warps x 32 threads

# =============================================================================
# WARP SPECIALIZATION CONFIGURATION
# =============================================================================

# Warp roles
WARP_PRODUCER = 0    # Warp 0: TMA loads
WARP_CONSUMER = 1    # Warp 1: MMA compute
# Warps 2-3: Reserved for future epilogue pipelining

# Pipeline stages
# DO NOT CHANGE: RAG Brain recorded 3-stage was 30% SLOWER, 4-stage was 46% SLOWER
# Using 2-stage for warp specialization overlap (producer loads N+1 while consumer computes N)
num_ab_stage = 2     # 2-stage for warp specialization overlap
num_acc_stage = 1

# Tensor memory allocation
num_tmem_alloc_cols = 512


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ceil_div(a, b):
    """Integer ceiling division."""
    return (a + b - 1) // b


# =============================================================================
# DEVICE KERNEL
# =============================================================================

@cute.kernel
def kernel(
    # MMA configuration
    tiled_mma: cute.TiledMma,
    # TMA atoms for data loading
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB_nkl: cute.Tensor,
    # Pointer tensors for group GEMM
    tensor_of_abc_ptrs: cute.Tensor,
    tensor_of_sfasfb_ptrs: cute.Tensor,
    tensormaps: cute.Tensor,
    tensor_of_problem_sizes: cute.Tensor,
    # Shared memory layouts
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    # Group GEMM configuration
    cta_mn_list: List[Tuple[int, int]],
    num_tma_load_bytes: cutlass.Constexpr[int],
):
    """
    True Warp-Specialized NVFP4 Group GEMM Kernel

    Warp Roles:
    - Warp 0 (PRODUCER): Issues TMA loads only, runs ahead filling pipeline
    - Warp 1 (CONSUMER): Executes MMA compute only, chases producer
    - Warps 2-3: Idle during mainloop, participate in epilogue store
    """
    # -------------------------------------------------------------------------
    # Thread/Warp Identification
    # -------------------------------------------------------------------------
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    tidx, _, _ = cute.arch.thread_idx()

    # -------------------------------------------------------------------------
    # Block Delinearization (Group GEMM scheduling)
    # -------------------------------------------------------------------------
    bidx, bidy, bidz = cute.arch.block_idx()

    # Delinearize bidz to (group_idx, coord_x, coord_y)
    # Each group has cta_m x cta_n CTAs
    group_idx = 0
    find = False
    coord_x = 0
    coord_y = 0
    cta_rest = bidz

    for _, (cta_m, cta_n) in enumerate(cta_mn_list):
        if cta_rest >= (cta_m * cta_n):
            group_idx += 1
            cta_rest -= cta_m * cta_n
        else:
            if not find:
                coord_y = cta_rest // cta_m
                coord_x = cta_rest % cta_m
                # NOTE: This subtraction looks dead but is a LOOP GUARD.
                # It drives cta_rest negative so subsequent iterations never
                # satisfy `cta_rest >= cta_m * cta_n`, preventing group_idx
                # from being incorrectly incremented for multi-group problems.
                cta_rest -= cta_m * cta_n
                find = True

    # -------------------------------------------------------------------------
    # Output Tensor Construction
    # -------------------------------------------------------------------------
    # Get output pointer for this group
    mC_mnl_iter = cute.make_ptr(
        c_dtype, tensor_of_abc_ptrs[group_idx, 2], cute.AddressSpace.gmem
    ).align(32)

    # Get problem dimensions
    m = tensor_of_problem_sizes[group_idx, 0]
    n = tensor_of_problem_sizes[group_idx, 1]
    k = tensor_of_problem_sizes[group_idx, 2]
    l = tensor_of_problem_sizes[group_idx, 3]

    # Create output tensor layout (row-major)
    mC_mnl_layout = cute.make_layout(
        (m, n, l),
        stride=(cute.assume(n, 32), 1, cute.assume(m * n, 32),)
    )
    mC_mnl = cute.make_tensor(mC_mnl_iter, mC_mnl_layout)

    # Tile output for this CTA
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (coord_x, coord_y, 0)
    )

    # -------------------------------------------------------------------------
    # Shared Memory Allocation
    # -------------------------------------------------------------------------

    # Calculate tensormap storage size
    size_tensormap_in_i64 = num_tensormaps * bytes_per_tensormap // 8

    @cute.struct
    class SharedStorage:
        # TMA descriptor storage (128 bytes per tensormap)
        tensormap_buffer: cute.struct.MemRange[cutlass.Int64, size_tensormap_in_i64]

        # Pipeline barrier storage
        # Each stage needs 16 bytes (2 x Int64) for mbarrier
        ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
        acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]

        # Tensor memory allocation holding buffer
        tmem_holding_buf: cutlass.Int32

    # Allocate shared memory
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # Extract tensormap pointers
    tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
    tensormap_a_smem_ptr = tensormap_smem_ptr
    tensormap_b_smem_ptr = tensormap_a_smem_ptr + bytes_per_tensormap // 8
    tensormap_sfa_smem_ptr = tensormap_b_smem_ptr + bytes_per_tensormap // 8
    tensormap_sfb_smem_ptr = tensormap_sfa_smem_ptr + bytes_per_tensormap // 8

    # -------------------------------------------------------------------------
    # Shared Memory Tensor Allocation (A, B, SFA, SFB)
    # -------------------------------------------------------------------------

    # A matrix: with 128-byte swizzle for bank conflict avoidance
    sA = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=a_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=a_smem_layout_staged.inner,
    )

    # B matrix: with 128-byte swizzle
    sB = smem.allocate_tensor(
        element_type=ab_dtype,
        layout=b_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=b_smem_layout_staged.inner,
    )

    # Scale factor A: NO swizzle (per gau-nernst pattern)
    sSFA = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfa_smem_layout_staged,
        byte_alignment=128,
    )

    # Scale factor B: NO swizzle
    sSFB = smem.allocate_tensor(
        element_type=sf_dtype,
        layout=sfb_smem_layout_staged,
        byte_alignment=128,
    )

    # =========================================================================
    # PIPELINE INITIALIZATION (Warp-Specialized)
    # =========================================================================
    #
    # Warp Specialization Pipeline Architecture:
    #
    #   Warp 0 (Producer)          Warp 1 (Consumer)
    #   -----------------          -----------------
    #   acquire_and_advance()      wait_and_advance()
    #         |                           |
    #         v                           v
    #   TMA Load (async)           MMA Compute
    #         |                           |
    #         v                           v
    #   [barrier signal]  ---->    [barrier wait]
    #         |                           |
    #         v                           v
    #   next iteration             release()
    #                                     |
    #                                     v
    #                              next iteration
    #
    # The 2-stage pipeline allows Warp 0 to load tile N+1 while Warp 1
    # computes tile N, hiding TMA latency.
    # =========================================================================

    # Producer group: Warp 0 issues TMA commands
    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)

    # Consumer group: Single thread from consumer warp issues MMA
    # Warp 1 waits on barriers and drives UMMA compute
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)

    # Create TMA-UMMA pipeline with 2-stage buffering
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,  # 2 stages for overlap
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
    ).make_participants()

    # Accumulator pipeline: Producer (compute) -> Consumer (epilogue)
    # All threads participate in epilogue
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
    ).make_participants()

    # -------------------------------------------------------------------------
    # Global Tensor Partitioning (S7)
    # -------------------------------------------------------------------------

    # Local_tile partition global tensors
    gA_mkl = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gB_nkl = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )
    gSFA_mkl = cute.local_tile(
        mSFA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None)
    )
    gSFB_nkl = cute.local_tile(
        mSFB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None, None)
    )

    # Partition for TiledMMA
    thr_mma = tiled_mma.get_slice(tidx)
    tCgA = thr_mma.partition_A(gA_mkl)
    tCgB = thr_mma.partition_B(gB_nkl)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    tCgSFB = thr_mma.partition_B(gSFB_nkl)
    tCgC = thr_mma.partition_C(gC_mnl)

    # -------------------------------------------------------------------------
    # TMA Descriptor Setup (S8)
    # -------------------------------------------------------------------------

    # Update TMA descriptors
    tensormap_manager = utils.TensorMapManager(utils.TensorMapUpdateMode.SMEM, 128)
    tensormap_a_gmem_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(bidz, 0, None)].iterator)
    tensormap_b_gmem_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(bidz, 1, None)].iterator)
    tensormap_sfa_gmem_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(bidz, 2, None)].iterator)
    tensormap_sfb_gmem_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(bidz, 3, None)].iterator)

    mA_mkl_iter = cute.make_ptr(ab_dtype, tensor_of_abc_ptrs[group_idx, 0], cute.AddressSpace.gmem).align(32)
    mB_nkl_iter = cute.make_ptr(ab_dtype, tensor_of_abc_ptrs[group_idx, 1], cute.AddressSpace.gmem).align(32)
    sfa_mkl_iter = cute.make_ptr(sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 0], cute.AddressSpace.gmem).align(32)
    sfb_nkl_iter = cute.make_ptr(sf_dtype, tensor_of_sfasfb_ptrs[group_idx, 1], cute.AddressSpace.gmem).align(32)

    mA_mkl_layout = cute.make_layout((m, k, l), stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32),))
    mB_nkl_layout = cute.make_layout((n, k, l), stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32),))

    atom_shape = ((32, 4), (sf_vec_size, 4))
    atom_stride = ((16, 4), (0, 1))
    sfa_layout = cute.tile_to_shape(
        cute.make_layout(atom_shape, stride=atom_stride),
        mA_mkl_layout.shape, (2, 1, 3),
    )
    sfb_layout = cute.tile_to_shape(
        cute.make_layout(atom_shape, stride=atom_stride),
        mB_nkl_layout.shape, (2, 1, 3),
    )
    real_tensor_a = cute.make_tensor(mA_mkl_iter, mA_mkl_layout)
    real_tensor_b = cute.make_tensor(mB_nkl_iter, mB_nkl_layout)
    real_tensor_sfa = cute.make_tensor(sfa_mkl_iter, sfa_layout)
    real_tensor_sfb = cute.make_tensor(sfb_nkl_iter, sfb_layout)

    if warp_idx == 0:
        tensormap_manager.init_tensormap_from_atom(tma_atom_a, tensormap_a_smem_ptr, 0)
        tensormap_manager.init_tensormap_from_atom(tma_atom_b, tensormap_b_smem_ptr, 0)
        tensormap_manager.init_tensormap_from_atom(tma_atom_sfa, tensormap_sfa_smem_ptr, 0)
        tensormap_manager.init_tensormap_from_atom(tma_atom_sfb, tensormap_sfb_smem_ptr, 0)
        tensormap_manager.update_tensormap(
            (real_tensor_a, real_tensor_b, real_tensor_sfa, real_tensor_sfb),
            (tma_atom_a, tma_atom_b, tma_atom_sfa, tma_atom_sfb),
            (tensormap_a_gmem_ptr, tensormap_b_gmem_ptr, tensormap_sfa_gmem_ptr, tensormap_sfb_gmem_ptr),
            0,
            (tensormap_a_smem_ptr, tensormap_b_smem_ptr, tensormap_sfa_smem_ptr, tensormap_sfb_smem_ptr),
        )
        tensormap_manager.fence_tensormap_update(tensormap_a_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_b_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfa_gmem_ptr)
        tensormap_manager.fence_tensormap_update(tensormap_sfb_gmem_ptr)

    cute.arch.barrier()

    # -------------------------------------------------------------------------
    # TMA Partitions (S9)
    # -------------------------------------------------------------------------

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
    )
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa, 0, cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb, 0, cute.make_layout(1),
        cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    # -------------------------------------------------------------------------
    # MMA Fragments & Tensor Memory (S10)
    # -------------------------------------------------------------------------

    # Shared/tensor memory partitions for MMA
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    # Allocate tensor memory
    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(num_tmem_alloc_cols)
    tmem.wait_for_alloc()
    acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

    # SFA/SFB tmem tensors
    sfa_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc), dtype=sf_dtype
    )
    tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
        tiled_mma, mma_tiler_mnk, sf_vec_size,
        cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

    sfb_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc) + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
        dtype=sf_dtype,
    )
    tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
        tiled_mma, mma_tiler_mnk, sf_vec_size,
        cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
    )
    tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

    # -------------------------------------------------------------------------
    # S2T Copy Setup (S11)
    # -------------------------------------------------------------------------

    copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype)
    tCsSFA_compact = cute.filter_zeros(sSFA)
    tCtSFA_compact = cute.filter_zeros(tCtSFA)
    tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
    thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
    tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
    tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_)
    tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

    tCsSFB_compact = cute.filter_zeros(sSFB)
    tCtSFB_compact = cute.filter_zeros(tCtSFB)
    tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
    thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
    tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
    tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_)
    tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

    # -------------------------------------------------------------------------
    # K-tile Count & Coordinate Slicing (S12)
    # -------------------------------------------------------------------------

    k_tile_cnt = cute.ceil_div(real_tensor_a.shape[1], mma_tiler_mnk[2])

    # Slice to per mma tile index
    mma_tile_coord_mnl = (coord_x, coord_y, 0)
    tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

    # =========================================================================
    # MAIN LOOP (S13) - True Warp-Specialized 2-Stage Pipeline
    # =========================================================================
    #
    # ARCHITECTURE: Warp 0 = Producer (TMA only), Warp 1 = Consumer (MMA only)
    #
    # Timeline (true overlap):
    #   Warp 0: [load 0][load 1][load 2][load 3]...  (runs ahead)
    #   Warp 1:         [mma 0][mma 1][mma 2][mma 3]... (chases)
    #
    # Producer acquires empty slot → issues 4 TMA loads → loops immediately.
    # Consumer waits for full slot → S2T + MMA → releases slot → loops.
    # 2-stage buffer: producer can be 1 tile ahead of consumer at all times.
    # Warps 2-3: idle during mainloop, participate in epilogue.
    #
    # =========================================================================

    # Pre-compute number of k-blocks per tile (needed by consumer)
    num_kblocks = cute.size(tCrA, mode=[2])

    if warp_idx == 0:
        # =================================================================
        # WARP 0 — PRODUCER: TMA loads only, runs ahead of consumer
        # =================================================================
        for k_tile in range(k_tile_cnt):
            ab_empty = ab_producer.acquire_and_advance()

            # Issue TMA loads for all 4 tensors (async, non-blocking)
            cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_a_gmem_ptr, cute.AddressSpace.generic))
            cute.copy(tma_atom_b, tBgB[(None, k_tile)], tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_b_gmem_ptr, cute.AddressSpace.generic))
            cute.copy(tma_atom_sfa, tAgSFA[(None, k_tile)], tAsSFA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_sfa_gmem_ptr, cute.AddressSpace.generic))
            cute.copy(tma_atom_sfb, tBgSFB[(None, k_tile)], tBsSFB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
                tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_sfb_gmem_ptr, cute.AddressSpace.generic))
            # Producer does NOT wait — loops back to acquire next slot immediately

    elif warp_idx == 1:
        # =================================================================
        # WARP 1 — CONSUMER: MMA compute only, chases producer
        # =================================================================
        # Acquire accumulator slot (signals epilogue when all k-tiles done)
        acc_empty = acc_producer.acquire_and_advance()

        # First tile: overwrite accumulator (no accumulate)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

        for k_tile in range(k_tile_cnt):
            # Wait for producer to fill this stage
            ab_full = ab_consumer.wait_and_advance()

            # Copy scale factors: shared memory → tensor memory (S2T)
            s2t_stage_coord = (None, None, None, None, ab_full.index)
            tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
            tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
            cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_staged, tCtSFA_compact_s2t)
            cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_staged, tCtSFB_compact_s2t)

            # Execute MMA for all K-blocks in this tile
            for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                kblock_coord = (None, None, kblock_idx, ab_full.index)
                sf_kblock_coord = (None, None, kblock_idx)
                tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tCtSFB[sf_kblock_coord].iterator)
                cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Release buffer slot back to producer
            ab_full.release()

        # Signal accumulator ready for epilogue
        acc_empty.commit()

    # Warps 2-3: skip mainloop entirely, proceed to epilogue wait

    # =========================================================================
    # EPILOGUE (S14)
    # =========================================================================

    op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
    copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc[None,0,0])
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    tDtAcc = thr_copy_t2r.partition_S(tCtAcc[None,0,0])
    tDgC = thr_copy_t2r.partition_D(tCgC[None,0,0])

    tDrAcc = cute.make_rmem_tensor(tDgC.shape, cutlass.Float32)
    tDrC = cute.make_rmem_tensor(tDgC.shape, c_dtype)

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()

    cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)
    acc_vec = tDrAcc.load()
    tDrC.store(acc_vec.to(c_dtype))

    simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_dtype, num_bits_per_copy=16)
    thread_layout = cute.make_layout((1, threads_per_cta), stride=(threads_per_cta, 1))
    value_layout = cute.make_layout((1, 1))
    tiled_copy_r2g = cute.make_tiled_copy_tv(simt_atom, thread_layout, value_layout)
    thr_copy_r2g = tiled_copy_r2g.get_slice(tidx)

    # Fast-path: skip predicate construction for full (non-boundary) tiles
    residue_m = mC_mnl.shape[0] - cutlass.Int32(coord_x) * mma_tiler_mnk[0]
    residue_n = mC_mnl.shape[1] - cutlass.Int32(coord_y) * mma_tiler_mnk[1]

    if residue_m >= mma_tiler_mnk[0] and residue_n >= mma_tiler_mnk[1]:
        # Full tile — unpredicated vectorized store
        cute.copy(simt_atom, cute.flatten(tDrC), cute.flatten(tDgC))
    else:
        # Boundary tile — predicated store with residue masking
        cC = cute.make_identity_tensor(gC_mnl.shape)
        tDcC = thr_copy_r2g.partition_D(cC)
        tDpC = cute.make_rmem_tensor(tDrC.shape, cutlass.Boolean)
        for i in range(cute.size(tDrC.shape)):
            # NOTE: Axis order is (residue_n, residue_m) because tDcC is partitioned
            # via thr_copy_r2g (SIMT R2G) which distributes along stride-1 (N-axis) first.
            # After partition_D, coordinate[0] maps to N and coordinate[1] maps to M.
            # Verified empirically on B200 hardware — do NOT swap without re-testing.
            tDpC[i] = cute.elem_less(tDcC[i], (residue_n, residue_m))
        cute.copy(simt_atom, cute.flatten(tDrC), cute.flatten(tDgC), pred=cute.flatten(tDpC))

    acc_full.release()
    cute.arch.barrier()
    tmem.free(acc_tmem_ptr)


# =============================================================================
# JIT KERNEL WRAPPER (S15)
# =============================================================================

@cute.jit
def my_kernel(
    ptr_of_tensor_of_problem_sizes: cute.Pointer,
    ptr_of_tensor_of_abc_ptrs: cute.Pointer,
    ptr_of_tensor_of_sfasfb_ptrs: cute.Pointer,
    ptr_of_tensor_of_tensormap: cute.Pointer,
    total_num_clusters: cutlass.Int32,
    problem_sizes: List[Tuple[int, int, int, int]],
    num_groups: cutlass.Int32,
):
    tensor_of_abc_ptrs = cute.make_tensor(
        ptr_of_tensor_of_abc_ptrs, cute.make_layout((num_groups, 3), stride=(3, 1))
    )
    tensor_of_sfasfb_ptrs = cute.make_tensor(
        ptr_of_tensor_of_sfasfb_ptrs, cute.make_layout((num_groups, 2), stride=(2, 1))
    )
    tensor_of_problem_sizes = cute.make_tensor(
        ptr_of_tensor_of_problem_sizes, cute.make_layout((num_groups, 4), stride=(4, 1))
    )
    tensor_of_tensormap = cute.make_tensor(
        ptr_of_tensor_of_tensormap, cute.make_layout((total_num_clusters, 4, 16), stride=(64, 16, 1))
    )

    min_a_shape = (cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(1))
    min_b_shape = (cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(64), cutlass.Int32(1))
    initial_a = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout(
            (min_a_shape[0], cute.assume(min_a_shape[2], 32), min_a_shape[3]),
            stride=(cute.assume(min_a_shape[2], 32), 1, cute.assume(min_a_shape[0] * min_a_shape[2], 32)),
        ),
    )
    initial_b = cute.make_tensor(
        cute.make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout(
            (min_b_shape[1], cute.assume(min_b_shape[2], 32), min_b_shape[3]),
            stride=(cute.assume(min_b_shape[2], 32), 1, cute.assume(min_b_shape[1] * min_b_shape[2], 32)),
        ),
    )

    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(initial_a.shape, sf_vec_size)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(initial_b.shape, sf_vec_size)
    initial_sfa = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16), sfa_layout)
    initial_sfb = cute.make_tensor(
        cute.make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=16), sfb_layout)

    mma_op = tcgen05.MmaMXF4NVF4Op(
        sf_dtype, (mma_tiler_mnk[0], mma_tiler_mnk[1], mma_inst_shape_k),
        tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
    )
    tiled_mma = cute.make_tiled_mma(mma_op)
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
    )

    a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage)
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, ab_dtype, num_ab_stage)
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage)
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, num_ab_stage)
    atom_thr_size = cute.size(tiled_mma.thr_id.shape)

    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_a, a_smem_layout, mma_tiler_mnk, tiled_mma, cluster_layout_vmnk.shape,
    )
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_b, b_smem_layout, mma_tiler_mnk, tiled_mma, cluster_layout_vmnk.shape,
    )
    sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfa, sfa_smem_layout, mma_tiler_mnk, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
        initial_sfb, sfb_smem_layout, mma_tiler_mnk, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )

    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

    cta_mn_list = []
    for group_idx, (m, n, k, l) in enumerate(problem_sizes):
        x, y = cute.ceil_div(problem_sizes[group_idx][:2], mma_tiler_mnk[0:2])
        cta_mn_list.append((x, y))

    grid = (1, 1, total_num_clusters)

    kernel(
        tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
        tma_atom_sfa, tma_tensor_sfa, tma_atom_sfb, tma_tensor_sfb,
        tensor_of_abc_ptrs, tensor_of_sfasfb_ptrs, tensor_of_tensormap,
        tensor_of_problem_sizes, a_smem_layout_staged, b_smem_layout_staged,
        sfa_smem_layout_staged, sfb_smem_layout_staged, cta_mn_list, num_tma_load_bytes,
    ).launch(grid=grid, block=[threads_per_cta, 1, 1], cluster=(1, 1, 1))


# =============================================================================
# PYTHON RUNTIME (S16)
# =============================================================================

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _silu_gate_kernel(
        t1_ptr, t2_ptr, out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SiLU gating: out = silu(t1) * t2, FP16 I/O with FP32 accumulation for sigmoid."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        t1 = tl.load(t1_ptr + offsets, mask=mask).to(tl.float32)
        t2 = tl.load(t2_ptr + offsets, mask=mask).to(tl.float32)

        silu_val = t1 * tl.sigmoid(t1)
        result = silu_val * t2

        tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)


def _fused_silu_gate(temp1, temp2, out_dtype):
    """Fused SiLU gating: silu(temp1) * temp2."""
    if _HAS_TRITON:
        output = torch.empty_like(temp1, dtype=out_dtype)
        n_elements = temp1.numel()
        BLOCK_SIZE = 1024
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _silu_gate_kernel[grid](
            temp1, temp2, output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
    else:
        # Fallback: plain torch, no torch.compile overhead
        return (torch.nn.functional.silu(temp1.float()) * temp2.float()).to(out_dtype)


_compiled_kernel_cache = {}

def compile_kernel(problem_sizes):
    global _compiled_kernel_cache
    cache_key = str(tuple(tuple(ps) for ps in problem_sizes))
    if cache_key in _compiled_kernel_cache:
        return _compiled_kernel_cache[cache_key]

    cute_ptr_of_tensor_of_problem_sizes = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    cute_ptr_of_tensor_of_abc_ptrs = make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16)
    cute_ptr_of_tensor_of_sfasfb_ptrs = make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16)
    total_num_clusters = cutlass.Int32(1)
    num_groups = cutlass.Int32(len(problem_sizes))
    cute_ptr_of_tensor_of_tensormap = make_ptr(cutlass.Int64, 0, cute.AddressSpace.gmem, assumed_align=16)

    compiled_func = cute.compile(
        my_kernel,
        cute_ptr_of_tensor_of_problem_sizes,
        cute_ptr_of_tensor_of_abc_ptrs,
        cute_ptr_of_tensor_of_sfasfb_ptrs,
        cute_ptr_of_tensor_of_tensormap,
        total_num_clusters,
        problem_sizes,
        num_groups
    )
    _compiled_kernel_cache[cache_key] = compiled_func
    return compiled_func


def run_single_gemm(a, b, sfa_perm, sfb_perm, output, problem_sizes):
    """Execute a single block-scaled GEMM: output = A @ B."""
    compiled_func = compile_kernel(problem_sizes)

    # Create pointer arrays for the kernel
    abc_ptrs = [(a.data_ptr(), b.data_ptr(), output.data_ptr())]
    sfasfb_ptrs = [(sfa_perm.data_ptr(), sfb_perm.data_ptr())]

    tensor_of_problem_sizes = torch.tensor(problem_sizes, dtype=torch.int32, device="cuda")
    tensor_of_abc_ptrs = torch.tensor(abc_ptrs, dtype=torch.int64, device="cuda")
    tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, dtype=torch.int64, device="cuda")

    cta_tile_shape_mn = (128, mma_tiler_mnk[1])
    cluster_tile_shape_mn = cta_tile_shape_mn

    total_num_clusters = 0
    num_groups = len(problem_sizes)
    for m, n, _, _ in problem_sizes:
        num_clusters_mn = tuple((x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn))
        total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

    tensormap_shape = (total_num_clusters, num_tensormaps, bytes_per_tensormap // 8)
    tensor_of_tensormap = torch.empty(tensormap_shape, dtype=torch.int64, device="cuda")

    cute_ptr_of_tensor_of_abc_ptrs = make_ptr(
        cutlass.Int64, tensor_of_abc_ptrs.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    cute_ptr_of_tensor_of_sfasfb_ptrs = make_ptr(
        cutlass.Int64, tensor_of_sfasfb_ptrs.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    cute_ptr_of_tensor_of_problem_sizes = make_ptr(
        cutlass.Int32, tensor_of_problem_sizes.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    cute_ptr_of_tensor_of_tensormap = make_ptr(
        cutlass.Int64, tensor_of_tensormap.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    compiled_func(
        cute_ptr_of_tensor_of_problem_sizes,
        cute_ptr_of_tensor_of_abc_ptrs,
        cute_ptr_of_tensor_of_sfasfb_ptrs,
        cute_ptr_of_tensor_of_tensormap,
        total_num_clusters,
        problem_sizes,
        num_groups,
    )

    return output


def custom_kernel(data: input_t) -> output_t:
    """
    Main entry point for NVFP4 dual GEMM with SiLU fusion.

    Computes: C = silu(A @ B1) * (A @ B2)

    Handles two input formats:
    1. GROUP GEMM format (4 elements): (abc_tensors, _, sfasfb_tensors, problem_sizes)
       - Used by gpumode evaluation
       - abc_tensors[0] = (a, b1, c) for GEMM1
       - abc_tensors[1] = (a, b2, c) for GEMM2 (same c buffer, will be overwritten)

    2. TASK format (10 elements): (a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c)
       - Used by local task.py testing
    """
    # Detect input format by length
    if len(data) == 4:
        # GROUP GEMM format from gpumode evaluation
        # (abc_tensors, _, sfasfb_reordered_tensors, problem_sizes)
        # This is regular GROUP GEMM - independent GEMMs with potentially different sizes
        abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
        compiled_func = compile_kernel(problem_sizes)
        num_groups = len(abc_tensors)

        # Standard GROUP GEMM - run all groups in single kernel launch
        abc_ptrs = []
        sfasfb_ptrs = []
        for i, ((a, b, c), (sfa_reordered, sfb_reordered), (m, n, k, l)) in enumerate(
            zip(abc_tensors, sfasfb_reordered_tensors, problem_sizes)
        ):
            abc_ptrs.append((a.data_ptr(), b.data_ptr(), c.data_ptr()))
            sfasfb_ptrs.append((sfa_reordered.data_ptr(), sfb_reordered.data_ptr()))

        tensor_of_problem_sizes = torch.tensor(problem_sizes, dtype=torch.int32, device="cuda")
        tensor_of_abc_ptrs = torch.tensor(abc_ptrs, dtype=torch.int64, device="cuda")
        tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, dtype=torch.int64, device="cuda")

        cta_tile_shape_mn = (128, mma_tiler_mnk[1])
        cluster_tile_shape_mn = cta_tile_shape_mn

        total_num_clusters = 0
        for m, n, _, _ in problem_sizes:
            num_clusters_mn = tuple((x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn))
            total_num_clusters += functools.reduce(lambda x, y: x * y, num_clusters_mn)

        tensormap_shape = (total_num_clusters, num_tensormaps, bytes_per_tensormap // 8)
        tensor_of_tensormap = torch.empty(tensormap_shape, dtype=torch.int64, device="cuda")

        cute_ptr_of_tensor_of_abc_ptrs = make_ptr(
            cutlass.Int64, tensor_of_abc_ptrs.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        cute_ptr_of_tensor_of_sfasfb_ptrs = make_ptr(
            cutlass.Int64, tensor_of_sfasfb_ptrs.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        cute_ptr_of_tensor_of_problem_sizes = make_ptr(
            cutlass.Int32, tensor_of_problem_sizes.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        cute_ptr_of_tensor_of_tensormap = make_ptr(
            cutlass.Int64, tensor_of_tensormap.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

        compiled_func(
            cute_ptr_of_tensor_of_problem_sizes,
            cute_ptr_of_tensor_of_abc_ptrs,
            cute_ptr_of_tensor_of_sfasfb_ptrs,
            cute_ptr_of_tensor_of_tensormap,
            total_num_clusters,
            problem_sizes,
            num_groups,
        )

        res = []
        for i in range(num_groups):
            res.append(abc_tensors[i][2])
        return res

    else:
        # TASK format (10 elements) from local testing
        # (a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c)
        a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c = data

        # Get dimensions from output tensor [M, N, L]
        m, n, l = c.shape
        # K dimension from A tensor shape [M, K//2, L] -> K = shape[1] * 2
        k = a.shape[1] * 2

        # Problem sizes for the kernel
        problem_sizes = [(m, n, k, l)]

        # Allocate temporary buffers for GEMM results (fp16 like output)
        temp1 = torch.empty_like(c)
        temp2 = torch.empty_like(c)

        # Pass 1: GEMM1 = A @ B1
        run_single_gemm(a, b1, sfa_perm, sfb1_perm, temp1, problem_sizes)

        # Pass 2: GEMM2 = A @ B2
        run_single_gemm(a, b2, sfa_perm, sfb2_perm, temp2, problem_sizes)

        # Fused: C = silu(GEMM1) * GEMM2 (single kernel launch)
        c.copy_(_fused_silu_gate(temp1, temp2, c.dtype))

        return c


def solve(data: input_t) -> output_t:
    """Alias for custom_kernel - main entry point."""
    return custom_kernel(data)
