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

namespace grouped_gemm {

using namespace cute;

// ==================== Tile Configurations ====================
//
// Naming: Config_{M}x{N}x{K}_{Schedule}_{ClusterMxNx1}
//   Schedule: Co = Cooperative, PP = Pingpong
//   ClusterShape: only M dimension varies, N=1, K=1 always
//
// Pingpong schedule alternates between two smem buffers, reducing pipeline
// bubble by ~15% compared to Cooperative. Requires CUTLASS >= v3.6.0.
//
// GMMA constraints for FP16/BF16:
//   - Tile M >= 128 (2 warp groups × MMA_64xN)
//   - Tile N >= 128 (MMA atom minimum N=128)
// ==============================================================

// ---------- Cooperative Schedule ----------

struct Config_128x128x64_Co {
    using TileShape    = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
};

struct Config_128x256x64_Co {
    using TileShape    = Shape<_128, _256, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
};

// ---------- Pingpong Schedule ----------
// Pingpong uses double-buffered smem with alternating producer/consumer roles.
// Typically 10-15% faster than Cooperative for grouped GEMM because it
// eliminates the pipeline drain/fill bubble between K-loop iterations.

struct Config_128x128x128_PP {
    using TileShape    = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_2, _1, _1>;
    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
};

struct Config_128x256x64_PP {
    using TileShape    = Shape<_128, _256, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    static constexpr int AlignmentA = 8;
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    using KernelSchedule   = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
};

// ==================== Kernel Type Builder ====================

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

    using ProblemShape     = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using KernelSchedule   = typename GemmConfigT::KernelSchedule;
    using EpilogueSchedule = typename GemmConfigT::EpilogueSchedule;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90,
            cutlass::arch::OpClassTensorOp,
            TileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccum, ElementAccum,
            ElementC, LayoutC *, AlignmentC,
            ElementC, LayoutC *, AlignmentC,
            EpilogueSchedule
        >::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90,
            cutlass::arch::OpClassTensorOp,
            ElementA, LayoutA *, AlignmentA,
            ElementB, LayoutB *, AlignmentB,
            ElementAccum,
            TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelSchedule
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue
    >;

    using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename GemmKernel::InternalStrideA;
    using StrideB = typename GemmKernel::InternalStrideB;
    using StrideC = typename GemmKernel::InternalStrideC;
    using StrideD = typename GemmKernel::InternalStrideD;
};

// ==================== BF16 Instantiations ====================

// Cooperative
using BF16_128x128x64_Co = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    Config_128x128x64_Co>;
using BF16_128x256x64_Co = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    Config_128x256x64_Co>;

// Pingpong
using BF16_128x128x128_PP = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    Config_128x128x128_PP>;
using BF16_128x256x64_PP = Sm90GroupedGemmKernel<
    cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    Config_128x256x64_PP>;

// ==================== FP16 Instantiations ====================

using F16_128x128x64_Co = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    Config_128x128x64_Co>;
using F16_128x256x64_Co = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    Config_128x256x64_Co>;

using F16_128x128x128_PP = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    Config_128x128x128_PP>;
using F16_128x256x64_PP = Sm90GroupedGemmKernel<
    cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
    Config_128x256x64_PP>;

}  // namespace grouped_gemm
