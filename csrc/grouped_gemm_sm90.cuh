#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

// --------------------------------------------------------------------------
// CUTLASS 3.x SM90 (Hopper) Persistent Grouped GEMM Kernel
//
// Key optimizations over naive grouped GEMM:
//   1. Persistent thread blocks: CTAs stay resident, pull tiles from a global
//      pool spanning ALL groups → automatic load balancing
//   2. TMA (Tensor Memory Accelerator): async, hardware-accelerated global→smem
//      copies, fully overlapped with GMMA compute
//   3. Warp-specialized: separate producer/consumer warp groups for load/compute
//   4. Cooperative schedule: multiple CTAs cooperate on a single output tile
//      when beneficial (larger cluster shapes)
// --------------------------------------------------------------------------

namespace grouped_gemm {

using namespace cute;

// ========================= Kernel Configuration =========================

// Tile shapes tuned for H100 SM90
// For MoE where per-expert M can be small, 128x128x64 provides good balance
// between occupancy and per-tile efficiency
struct GemmConfig128x128x64 {
    using TileShape     = Shape<_128, _128, _64>;
    using ClusterShape  = Shape<_2, _1, _1>;
    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
};

// Larger tile for cases with sufficient M per expert
struct GemmConfig128x256x64 {
    using TileShape     = Shape<_128, _256, _64>;
    using ClusterShape  = Shape<_1, _1, _1>;
    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
};

// Smaller tile for very small M (< 128 tokens per expert)
struct GemmConfig64x128x64 {
    using TileShape     = Shape<_64, _128, _64>;
    using ClusterShape  = Shape<_1, _1, _1>;
    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
};

// ================ Kernel Type Builder (CollectiveBuilder API) ================

template <
    typename ElementA_,
    typename ElementB_,
    typename ElementC_,
    typename ElementAccum_,
    typename GemmConfigT
>
struct Sm90GroupedGemmKernel {
    using ElementA     = ElementA_;
    using ElementB     = ElementB_;
    using ElementC     = ElementC_;
    using ElementAccum = ElementAccum_;

    // MoE convention: A=[M,K] row-major, B=[K,N] column-major (i.e. B^T is row-major)
    // This matches: input [tokens, hidden] × weight^T [hidden, ffn]
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using TileShape    = typename GemmConfigT::TileShape;
    using ClusterShape = typename GemmConfigT::ClusterShape;

    static constexpr int AlignmentA = GemmConfigT::AlignmentA;
    static constexpr int AlignmentB = GemmConfigT::AlignmentB;
    static constexpr int AlignmentC = GemmConfigT::AlignmentC;

    // Grouped problem shape: array of (M_i, N_i, K_i) per expert
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

    // Build epilogue via CollectiveBuilder
    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90,
            cutlass::arch::OpClassTensorOp,
            TileShape,
            ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccum,     // accumulator
            ElementAccum,     // compute type for epilogue
            ElementC, LayoutC, AlignmentC,
            ElementC, LayoutC, AlignmentC,
            cutlass::epilogue::NoSmemWarpSpecialized
        >::CollectiveOp;

    // Build mainloop via CollectiveBuilder
    // Uses TMA + GMMA warp-specialized cooperative schedule
    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90,
            cutlass::arch::OpClassTensorOp,
            ElementA, LayoutA, AlignmentA,
            ElementB, LayoutB, AlignmentB,
            ElementAccum,
            TileShape,
            ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::KernelTmaWarpSpecializedCooperative
        >::CollectiveOp;

    // GemmUniversal with PersistentScheduler: tiles from ALL groups are pooled
    // and dynamically assigned to CTAs → near-perfect load balancing
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
    >;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Stride types needed for argument construction
    using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
    using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
    using StrideD = StrideC;  // D shares layout with C
};

// ========================= Concrete Instantiations =========================

// FP16 with 128x128x64 tile (default for MoE)
using GroupedGemmF16_128x128 = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    GemmConfig128x128x64>;

// BF16 with 128x128x64 tile
using GroupedGemmBF16_128x128 = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    GemmConfig128x128x64>;

// FP16 with 128x256x64 tile (for large M per expert)
using GroupedGemmF16_128x256 = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    GemmConfig128x256x64>;

// BF16 with 128x256x64 tile
using GroupedGemmBF16_128x256 = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    GemmConfig128x256x64>;

// FP16 with 64x128x64 tile (for very small M per expert)
using GroupedGemmF16_64x128 = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    GemmConfig64x128x64>;

// BF16 with 64x128x64 tile
using GroupedGemmBF16_64x128 = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    GemmConfig64x128x64>;

}  // namespace grouped_gemm
