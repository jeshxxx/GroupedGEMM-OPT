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
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

// --------------------------------------------------------------------------
// CUTLASS 3.x SM90 (Hopper) Persistent Grouped GEMM Kernel
//
// REQUIRES CUTLASS >= v3.6.0 for the group-aware tile scheduler
// (PersistentTileSchedulerSm90GroupParams) that was added in PR #1851.
//
// API matches CUTLASS example 57_hopper_grouped_gemm:
//   - Pointer-decorated layouts (LayoutA *) in CollectiveBuilder
//   - PtrArrayTmaWarpSpecializedCooperative epilogue with TMA stores
//   - Per-group stride device arrays
//   - GemmUniversal with default (void) tile scheduler
// --------------------------------------------------------------------------

namespace grouped_gemm {

using namespace cute;

// ========================= Kernel Configuration =========================

struct GemmConfig128x128x64 {
    using TileShape    = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    static constexpr int AlignmentA = 128 / 16;  // 8 elements for 16-bit types
    static constexpr int AlignmentB = 128 / 16;
    static constexpr int AlignmentC = 128 / 16;
};

struct GemmConfig128x256x64 {
    using TileShape    = Shape<_128, _256, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    static constexpr int AlignmentA = 128 / 16;
    static constexpr int AlignmentB = 128 / 16;
    static constexpr int AlignmentC = 128 / 16;
};

// Cooperative schedule requires tile M >= 128.
// For small per-expert M, use a narrower N (64) to produce more tiles
// along M, improving SM occupancy when tokens/expert is small.
struct GemmConfig128x64x64 {
    using TileShape    = Shape<_128, _64, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    static constexpr int AlignmentA = 128 / 16;
    static constexpr int AlignmentB = 128 / 16;
    static constexpr int AlignmentC = 128 / 16;
};

// ================ Kernel Type Builder (CollectiveBuilder API) ================
//
// Follows the pattern of CUTLASS example 57_hopper_grouped_gemm exactly.
// Key difference from single-GEMM kernels: layout tags are pointer-decorated
// (e.g., LayoutA * instead of LayoutA) which tells the CollectiveBuilder to
// generate Ptr-Array-aware mainloop and epilogue collectives.

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

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using TileShape    = typename GemmConfigT::TileShape;
    using ClusterShape = typename GemmConfigT::ClusterShape;

    static constexpr int AlignmentA = GemmConfigT::AlignmentA;
    static constexpr int AlignmentB = GemmConfigT::AlignmentB;
    static constexpr int AlignmentC = GemmConfigT::AlignmentC;

    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

    // Schedule types for Ptr-Array grouped GEMM
    using KernelSchedule  = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

    // Epilogue: TMA cooperative with per-group C/D pointer arrays.
    // Pointer-decorated layout tags (LayoutC *) tell the builder to generate
    // an epilogue that accepts arrays of C/D pointers and strides.
    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90,
            cutlass::arch::OpClassTensorOp,
            TileShape,
            ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccum,
            ElementAccum,
            ElementC, LayoutC *, AlignmentC,
            ElementC, LayoutC *, AlignmentC,
            EpilogueSchedule
        >::CollectiveOp;

    // Mainloop: TMA + GMMA warp-specialized cooperative.
    // Pointer-decorated layout tags (LayoutA *, LayoutB *) tell the builder to
    // generate a mainloop that accepts arrays of A/B pointers and strides, and
    // creates per-group TMA descriptors on the fly.
    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90,
            cutlass::arch::OpClassTensorOp,
            ElementA, LayoutA *, AlignmentA,
            ElementB, LayoutB *, AlignmentB,
            ElementAccum,
            TileShape,
            ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelSchedule
        >::CollectiveOp;

    // GemmUniversal with default tile scheduler (void).
    // In v3.6.0+, the array-cooperative kernel specialization resolves void to
    // PersistentTileSchedulerSm90GroupParams for grouped GEMM, which correctly
    // handles multi-group tile scheduling.
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Stride types derived from the kernel (accounts for pointer decoration)
    using StrideA = typename GemmKernel::InternalStrideA;
    using StrideB = typename GemmKernel::InternalStrideB;
    using StrideC = typename GemmKernel::InternalStrideC;
    using StrideD = typename GemmKernel::InternalStrideD;
};

// ========================= Concrete Instantiations =========================

using GroupedGemmF16_128x128 = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    GemmConfig128x128x64>;

using GroupedGemmBF16_128x128 = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    GemmConfig128x128x64>;

using GroupedGemmF16_128x256 = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    GemmConfig128x256x64>;

using GroupedGemmBF16_128x256 = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    GemmConfig128x256x64>;

using GroupedGemmF16_128x64 = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    GemmConfig128x64x64>;

using GroupedGemmBF16_128x64 = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    GemmConfig128x64x64>;

}  // namespace grouped_gemm
