#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm {

enum class Layout {
    RowMajor,
    ColMajor
};

__device__ __host__ constexpr int get_num_math_warpgroups(int block_m) {
    return block_m == 64 ? 1 : 2;
}

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return get_num_math_warpgroups(block_m) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

template <int kNumFormerIters, int kGap, int kEnd>
__device__ __host__ void outer_launch_k_iterations(const auto& inner_launch_k_iterations, const auto& func, int num_former_iters) {
    if (num_former_iters == kNumFormerIters) {
        inner_launch_k_iterations(func, cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        outer_launch_k_iterations<kNumFormerIters + kGap, kGap, kEnd>(inner_launch_k_iterations, func, num_former_iters);
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t BLOCK_N_PADDING,
          uint32_t kSwizzleDMode,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          GemmType kGemmType,
          bool RowwiseScaling,
          bool FastAccum>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                uint32_t shape_m,
                const __grid_constant__ CUtensorMap tensor_map_a,
                const __grid_constant__ CUtensorMap tensor_map_b,
                const __grid_constant__ CUtensorMap tensor_map_scales_a,
                const __grid_constant__ CUtensorMap tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_N, BLOCK_K) == 1 or (constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");
    DG_STATIC_ASSERT(!RowwiseScaling || kGemmType == GemmType::Normal, "Rowwise scaling only supports normal GEMM");
    DG_STATIC_ASSERT(!FastAccum || kGemmType == GemmType::Normal, "FastAccum only supports normal GEMM");
    if constexpr (FastAccum) {
        DG_STATIC_ASSERT(RowwiseScaling, "FastAccum is only supported when RowwiseScaling is enabled");
    }

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // Shared memory
    static constexpr int kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * (BLOCK_N + BLOCK_N_PADDING) * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE_UNALIGNED = RowwiseScaling ? BLOCK_N * sizeof(float) : SHAPE_K_SCALES * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float);
    // need to align shared mem slice for B scales to barrier size, because next slice is going to be barrier
    static constexpr uint32_t SMEM_SCALES_B_SIZE = ceil_div<uint32_t>(SMEM_SCALES_B_SIZE_UNALIGNED, sizeof(Barrier)) * sizeof(Barrier);
    // Persistent copy of the A‑row scales (BLOCK_M floats, aligned to Barrier)
    static constexpr uint32_t SMEM_ROW_SCALES_SIZE = ceil_div<uint32_t>(BLOCK_M * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));

        // `tensor_map_d` is only used in swizzling mode
        // For the `kSwizzleDMode == 0 and BLOCK_N_PADDING == 0` case, it will be treated as padding mode
        if constexpr (kSwizzleDMode > 0)
            cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_scales_a[kNumStages];
    float* smem_scales_b;
    float* smem_row_scales;

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

    // Fill shared memory pointers
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_a[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_A_SIZE_PER_STAGE);
    }
    smem_scales_b = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));
    smem_row_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_scales_b) + SMEM_SCALES_B_SIZE);

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_row_scales) + SMEM_ROW_SCALES_SIZE);
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    auto launch_k_iterations = [](const auto& func, int num_former_iters) {
        constexpr bool kShouldOptimize = BLOCK_K / constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
        constexpr int kGap = constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
        constexpr int kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

        // NOTES: for too-many branches (> 5), we disable this optimization
        // Otherwise, the compiler must know the dynamic variable `num_former_iters`'s real value
        outer_launch_k_iterations<0, kGap, kEnd>([](const auto& func, auto num_former_iters_type) {
            if constexpr (SHAPE_K % kFullKOfAllStages == 0) {
                for (int k_iter = 0; k_iter < kNumIterations; ++ k_iter)
                    func(k_iter, DivisibleK{}, num_former_iters_type);
            } else {
                for (int k_iter = 0; k_iter < kNumIterations - 1; ++ k_iter)
                    func(k_iter, DivisibleK{}, num_former_iters_type);
                func(kNumIterations - 1, NotDivisibleK{}, num_former_iters_type);
            }
        }, func, kShouldOptimize ? num_former_iters : 0);
    };

    // Register reconfigurations
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA>(shape_m, grouped_layout);

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                launch_k_iterations([&](int k_iter, auto type, auto _) {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                    // Assign TMA multicast number into A and B
                    constexpr int kNumTMAMulticastOnA = kIsTMAMulticastOnA ? kNumTMAMulticast : 1;
                    constexpr int kNumTMAMulticastOnB = kIsTMAMulticastOnA ? 1 : kNumTMAMulticast;

                    // NOTES: unrolling and `kNumInnerStages` are vital for performance, NVCC will try to eliminate all
                    // shared memory pointers, e.g. `full_barriers` registers, if all the access indices are constant
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // Wait consumer release
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                        // Issue TMA A
                        auto& full_barrier = *full_barriers[s];
                        int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                        tma_copy<kNumTMAMulticastOnA>(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                      smem_a[s], k_idx, scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx));
                        int k_addr = RowwiseScaling ? 0 : k_idx/BLOCK_K;
                        tma_copy<kNumTMAMulticastOnA>(&tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                      smem_scales_a[s], m_block_idx * BLOCK_M,
                                                      scheduler.get_global_idx(SHAPE_K_SCALES, 1, k_addr));

                        // Issue TMA B
                        tma_copy<kNumTMAMulticastOnB>(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                                      smem_b[s], k_idx, scheduler.get_global_idx<false>(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx));
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                    }

                    // Wait unaligned cases
                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                        full_barriers[s]->arrive();
                    }
                }, 0);
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s)
                    empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + 1) & 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales B to load
            DG_STATIC_ASSERT(SHAPE_N % 8 == 0, "Invalid shape N");
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
            if constexpr (!RowwiseScaling) {
                if constexpr (not kMustUseUniformedScaleB) {
                    num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                    num_full_iters = min(SHAPE_N - n_block_idx * BLOCK_N, BLOCK_N) / 8;
                }
                uint32_t num_scales_b = SHAPE_K_SCALES * (num_former_iters >= num_full_iters ? 1 : 2);

                if (threadIdx.x >= 32) {
                    auto num_previous_lines = scheduler.get_global_idx<false>(ceil_div(SHAPE_N, BLOCK_K), 0, 0, m_block_idx);
                    auto local_scales_b = scales_b + (num_previous_lines + ((n_block_idx * BLOCK_N) / BLOCK_K)) * SHAPE_K_SCALES;
                    #pragma unroll
                    for (uint32_t i = threadIdx.x - 32; i < num_scales_b; i += kNumMathThreads - 32)
                        st_shared(smem_scales_b + i, __ldg(local_scales_b + i));
                }
            } else {
                uint32_t num_scales_b = BLOCK_N;
                if (threadIdx.x >= 32) {
                    auto local_scales_b = scales_b + n_block_idx * BLOCK_N;
                    #pragma unroll
                    for (uint32_t i = threadIdx.x - 32; i < num_scales_b; i += kNumMathThreads - 32) {
                        st_shared(smem_scales_b + i, __ldg(local_scales_b + i));
                    }
                }
            }
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Accumulation for WGMMA or CUDA promotion
            constexpr int WAVE_BLOCK_M = WGMMA::M * get_num_math_warpgroups(BLOCK_M);
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum] = {0};
            float final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void();
                }
            };

            // Launch MMAs
            launch_k_iterations([&](int k_iter, auto type, auto num_former_iters_type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                #pragma unroll
                for (int s = 0; s < kNumInnerStages; ++ s) {
                    float scale_b_0, scale_b_1;
                    if constexpr (!RowwiseScaling) {
                        // Read B scales
                        scale_b_0 = ld_shared(smem_scales_b + k_iter * kNumStages + s);
                        // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_b_1 = ld_shared(smem_scales_b + k_iter * kNumStages + s + SHAPE_K_SCALES);
                    }

                    // Wait TMA arrivals
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                    //------------------------------------------------------------------
                    // One‑time copy of the BLOCK_M row scales into the persistent slice
                    //------------------------------------------------------------------
                    if constexpr (FastAccum) {
                        if ((k_iter == 0) && (s == 0)) {              // only once per tile
                            if (warp_idx == 0) {                      // first warp does the copy
                                #pragma unroll
                                for (uint32_t r = lane_idx; r < BLOCK_M; r += 32) {
                                    float v = ld_shared(smem_scales_a[0] + r);
                                    st_shared(smem_row_scales + r, v);
                                }
                            }
                            cutlass::arch::NamedBarrier(kNumMathThreads).sync();   // ensure visibility
                        }
                    }

                    // TODO: remove some useless computation for unaligned Ms
                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                      	auto m_offset = local_idx * WAVE_BLOCK_M;

                    	// Read A scales
                    	// NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                    	auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0 + m_offset);
                        auto scale_a_1 = ld_shared(smem_scales_a[s] + r_1 + m_offset);

                    	// Commit WGMMA instructions
                    	#pragma unroll
                    	for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                    	warpgroup_arrive();
                    	#pragma unroll
                    	for (int k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            auto desc_a = make_smem_desc(smem_a[s] + (math_wg_idx * WGMMA::M + m_offset) * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            // NOTE(yf225): `scale_d` is true: D = A*B+D, false: D = A*B
                            bool scale_d;
                            if constexpr (FastAccum) {
                                scale_d = true;
                            } else {
                                scale_d = k;
                            }
                            WGMMA::wgmma(desc_a, desc_b, accum, scale_d);
                    	}
                    	warpgroup_commit_batch();
                    	#pragma unroll
                    	for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                    	warpgroup_wait<0>();

                    	// Notify barrier arrival at the last warpgroup wave
                        if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                    	    empty_barrier_arrive(s);

                        if constexpr (!FastAccum) {
                    	    // Promote with scales (only applies for FastAccum=false case)
                            // NOTES: making it as predicates is very important for performance, comparing to two loops
                            float scale_0_0, scale_1_0;
                            float scale_0_1, scale_1_1;
                            if constexpr (!RowwiseScaling) {
                                scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                                if constexpr (not kMustUseUniformedScaleB)
                                    scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;
                            } else {
                                // NOTE(yf225): In rowwise scaling, we apply the A‑scale when doing the multiply‑accumulate step,
                                // then apply the B‑scale later when we write the results out in the write-back (STSM) stage.
                                // That's why we only see scale_a being used (and not scale_b) here.
                                scale_0_0 = scale_a_0, scale_1_0 = scale_a_1;
                            }
                            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                            #pragma unroll
                            for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                // NOTES: for unrolled `num_former_iters` cases, we expect the compiler to automatically make it a constant
                                bool predicate = RowwiseScaling or kMustUseUniformedScaleB or i < num_former_iters;
                                shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                            }
                        }
                    }
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                    empty_barrier_arrive(s);
                }
            }, num_former_iters);

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");
            DG_STATIC_ASSERT(static_cast<int>(kSwizzleDMode > 0) + static_cast<int>(BLOCK_N_PADDING > 0) <= 1,
                            "Swizzling and padding are not compatible");

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        // Calculate the swizzling atom offset and in-atom offset
                        constexpr int kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                        // Calculate the index of the bank group to be written in the atom
                        auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                        // Reshape the atom in another view and swizzle
                        //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                        //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                        constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);

                        // Add back into the base pointer
                        // NOTES: think twice before modifying this, as changes may affect the number of instructions
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                            m_offset * kSwizzleDMode +                                 // Wave offset
                            atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                    } else {
                        // No swizzling, just padding
                        // NOTES: padding must be zero for BF16 output
                        DG_STATIC_ASSERT(BLOCK_N_PADDING == 0, "Padding must be zero for BF16 output");
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * (BLOCK_N + BLOCK_N_PADDING) + i * 8);
                    }

                    nv_bfloat162 v0, v1;
                    if constexpr (RowwiseScaling) {
                        // lane‑local 2‑column pair inside the 8‑column group
                        int pair_in_group = (lane_idx & 0x3) << 1;       // 0,2,4,6

                        // 8‑column groups are stepped by the loop variable `i`
                        int col0 = (i << 3) + pair_in_group;             // i*8 + {0,2,4,6}
                        int col1 = col0 + 1;                             //   + {1,3,5,7}

                        float s0 = ld_shared(smem_scales_b + col0);
                        float s1 = ld_shared(smem_scales_b + col1);

                        if constexpr (FastAccum) {
                            // Pre‑load the two row scales that the lane owns (once per warp‑row)
                            const int row_even = warp_idx * 16 + (lane_idx / 4);  // 0‑7
                            const int row_odd  = row_even + 8;                                // 8‑15
                            float a_even = ld_shared(smem_row_scales + row_even);   // ← use persistent copy
                            float a_odd  = ld_shared(smem_row_scales + row_odd);

                            float as0 = a_even * s0,  as1 = a_even * s1,
                                as2 = a_odd * s0,  as3 = a_odd * s1;

                            v0 = __float22bfloat162_rn({ accum[i*4 + 0] * as0,
                                                        accum[i*4 + 1] * as1 });
                            v1 = __float22bfloat162_rn({ accum[i*4 + 2] * as2,
                                                        accum[i*4 + 3] * as3 });
                        } else {
                            float t0 = shifted_accum[i*4 + 0] * s0;
                            float t1 = shifted_accum[i*4 + 1] * s1;
                            float t2 = shifted_accum[i*4 + 2] * s0;
                            float t3 = shifted_accum[i*4 + 3] * s1;
                            v0 = __float22bfloat162_rn({ t0, t1 });
                            v1 = __float22bfloat162_rn({ t2, t3 });
                        }
                    } else {
                        // final_accum already holds (sA·sB) products
                        v0 = __float22bfloat162_rn({ shifted_accum[i*4 + 0],
                                                    shifted_accum[i*4 + 1] });
                        v1 = __float22bfloat162_rn({ shifted_accum[i*4 + 2],
                                                    shifted_accum[i*4 + 3] });
                    }

                    // NOTES: only 16 lanes' addresses are used
                    SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        v0,
                        v1,
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            DG_STATIC_ASSERT(kNumMathThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr,
                                              n_block_idx * BLOCK_N + in_block_n_offset,
                                              scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx));
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t BLOCK_N_PADDING,
          uint32_t kSwizzleDMode,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          GemmType kGemmType,
          bool RowwiseScaling,
          bool FastAccum>
class Gemm {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    Gemm() = default;

    static void run(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                    uint32_t shape_m,
                    const CUtensorMap& tma_a_desc,
                    const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc,
                    const CUtensorMap& tma_d_desc,
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K,
                                      BLOCK_M, BLOCK_N, BLOCK_K,
                                      BLOCK_N_PADDING,
                                      kSwizzleDMode,
                                      kNumGroups, kNumStages,
                                      kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, kIsTMAMulticastOnA, kGemmType, RowwiseScaling, FastAccum>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;

        // Launch
        auto status = cudaLaunchKernelEx(&config, kernel,
                                         gmem_d, scales_b, grouped_layout,
                                         shape_m,
                                         tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_a_desc(T* global_address, uint32_t shape_m) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_K, BLOCK_M, BLOCK_K);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_b_desc(T* global_address) {
        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                SHAPE_K, SHAPE_N * (kGemmType != GemmType::Normal ? kNumGroups : 1), BLOCK_K, BLOCK_N);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_d_desc(T* global_address, uint32_t shape_m) {
        auto swizzle_mode = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
        if constexpr (kSwizzleDMode == 32)  swizzle_mode = CU_TENSOR_MAP_SWIZZLE_32B;
        if constexpr (kSwizzleDMode == 64)  swizzle_mode = CU_TENSOR_MAP_SWIZZLE_64B;
        if constexpr (kSwizzleDMode == 128) swizzle_mode = CU_TENSOR_MAP_SWIZZLE_128B;

        // Swizzling requires the inner box dim less or equal than `kSwizzleDMode` bytes
        // So `BLOCK_N * sizeof(T) / kSwizzleDMode` TMA stores are required
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_N,
                                BLOCK_M, kSwizzleDMode == 0 ? BLOCK_N : kSwizzleDMode / sizeof(T),
                                swizzle_mode);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_scales_a_desc(T* global_address, uint32_t shape_m) {
        // Make TMA aligned to 16 bytes
        constexpr uint32_t kAlignment = 16 / sizeof(T);
        shape_m = ceil_div(shape_m, kAlignment) * kAlignment;

        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                shape_m, ceil_div(SHAPE_K, BLOCK_K) * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), BLOCK_M, 1,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_desc(
            T* global_address, Layout layout,
            uint32_t gmem_rows, uint32_t gmem_cols,
            uint32_t smem_rows, uint32_t smem_cols,
            CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
        if (layout == Layout::RowMajor) {
            uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
            uint32_t smem_dim[2] = {smem_cols, smem_rows};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_cols * sizeof(T), smem_dim, swizzle_type);
        } else {
            uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
            uint32_t smem_dim[2] = {smem_rows, smem_cols};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
        }
    }
};

};  // namespace deep_gemm

#pragma clang diagnostic pop
