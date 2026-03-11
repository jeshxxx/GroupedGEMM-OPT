#include "grouped_gemm.h"
#include "grouped_gemm_sm90.cuh"

#include "cutlass/kernel_hardware_info.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace grouped_gemm {

// ---------------------------------------------------------------------------
// Cached SM count — avoid querying CUDA driver every call
// ---------------------------------------------------------------------------
static int cached_sm_count = -1;
static int cached_device_id = -1;

static int get_sm_count(int device_id) {
    if (cached_device_id != device_id) {
        cached_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
        cached_device_id = device_id;
    }
    return cached_sm_count;
}

// ---------------------------------------------------------------------------
// Per-group device arrays for CUTLASS grouped GEMM
// ---------------------------------------------------------------------------

static constexpr size_t kBufAlign = 16;
static size_t align_up(size_t x) { return (x + kBufAlign - 1) & ~(kBufAlign - 1); }

template <typename KernelType>
struct GroupedGemmArgs {
    using ElementA = typename KernelType::ElementA;
    using ElementB = typename KernelType::ElementB;
    using ElementC = typename KernelType::ElementC;
    using StrideA  = typename KernelType::StrideA;
    using StrideB  = typename KernelType::StrideB;
    using StrideC  = typename KernelType::StrideC;
    using StrideD  = typename KernelType::StrideD;
    using ProblemShape          = typename KernelType::ProblemShape;
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    int num_groups;
    std::vector<UnderlyingProblemShape> host_problem_sizes;

    // Single packed device buffer holding all 9 arrays
    torch::Tensor packed_device_buf;
    size_t off_problem_sizes;
    size_t off_ptr_A, off_ptr_B, off_ptr_C, off_ptr_D;
    size_t off_stride_A, off_stride_B, off_stride_C, off_stride_D;

    template<typename T>
    T* dev(size_t offset) const {
        return reinterpret_cast<T*>(
            static_cast<char*>(packed_device_buf.data_ptr()) + offset);
    }

    static GroupedGemmArgs prepare(
        const torch::Tensor& input,
        const torch::Tensor& weights,
        torch::Tensor& output,
        const torch::Tensor& tokens_per_expert,
        cudaStream_t stream,
        bool sort_by_m)
    {
        GroupedGemmArgs args;
        args.num_groups = tokens_per_expert.size(0);
        const int K = input.size(1);
        const int N = weights.size(1);
        const int G = args.num_groups;

        // Compute layout offsets for the packed buffer (9 arrays, 16-byte aligned)
        size_t s_ps = G * sizeof(UnderlyingProblemShape);
        size_t s_pA = G * sizeof(const ElementA*);
        size_t s_pB = G * sizeof(const ElementB*);
        size_t s_pC = G * sizeof(const ElementC*);
        size_t s_pD = G * sizeof(ElementC*);
        size_t s_sA = G * sizeof(StrideA);
        size_t s_sB = G * sizeof(StrideB);
        size_t s_sC = G * sizeof(StrideC);
        size_t s_sD = G * sizeof(StrideD);

        size_t total = 0;
        args.off_problem_sizes = total; total += align_up(s_ps);
        args.off_ptr_A  = total; total += align_up(s_pA);
        args.off_ptr_B  = total; total += align_up(s_pB);
        args.off_ptr_C  = total; total += align_up(s_pC);
        args.off_ptr_D  = total; total += align_up(s_pD);
        args.off_stride_A = total; total += align_up(s_sA);
        args.off_stride_B = total; total += align_up(s_sB);
        args.off_stride_C = total; total += align_up(s_sC);
        args.off_stride_D = total; total += align_up(s_sD);

        // Single host staging buffer
        std::vector<char> host_buf(total, 0);
        args.host_problem_sizes.resize(G);

        auto* h_pA = reinterpret_cast<const ElementA**>(host_buf.data() + args.off_ptr_A);
        auto* h_pB = reinterpret_cast<const ElementB**>(host_buf.data() + args.off_ptr_B);
        auto* h_pC = reinterpret_cast<const ElementC**>(host_buf.data() + args.off_ptr_C);
        auto* h_pD = reinterpret_cast<ElementC**>(host_buf.data() + args.off_ptr_D);
        auto* h_sA = reinterpret_cast<StrideA*>(host_buf.data() + args.off_stride_A);
        auto* h_sB = reinterpret_cast<StrideB*>(host_buf.data() + args.off_stride_B);
        auto* h_sC = reinterpret_cast<StrideC*>(host_buf.data() + args.off_stride_C);
        auto* h_sD = reinterpret_cast<StrideD*>(host_buf.data() + args.off_stride_D);

        const auto* tpe = tokens_per_expert.data_ptr<int64_t>();

        // Build group ordering: optionally sort by descending M for better
        // persistent scheduler load balance. Only reorders pointer/stride
        // arrays — zero data movement cost.
        std::vector<int> order(G);
        std::iota(order.begin(), order.end(), 0);
        if (sort_by_m && G > 1) {
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return tpe[a] > tpe[b];
            });
        }

        // Precompute per-expert input/output offsets in original order
        std::vector<int64_t> offsets(G);
        int64_t off = 0;
        for (int g = 0; g < G; ++g) {
            offsets[g] = off;
            off += tpe[g];
        }

        // Populate arrays in (potentially sorted) order
        for (int i = 0; i < G; ++i) {
            int g = order[i];
            int M_g = static_cast<int>(tpe[g]);
            int64_t a_off = offsets[g];

            args.host_problem_sizes[i] = cute::make_shape(M_g, N, K);

            h_pA[i] = reinterpret_cast<const ElementA*>(input.data_ptr()) + a_off * K;
            h_pB[i] = reinterpret_cast<const ElementB*>(weights.data_ptr()) + int64_t(g) * N * K;
            h_pC[i] = reinterpret_cast<const ElementC*>(output.data_ptr()) + a_off * N;
            h_pD[i] = reinterpret_cast<ElementC*>(output.data_ptr()) + a_off * N;

            h_sA[i] = cutlass::make_cute_packed_stride(StrideA{}, {M_g, K, 1});
            h_sB[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
            h_sC[i] = cutlass::make_cute_packed_stride(StrideC{}, {M_g, N, 1});
            h_sD[i] = cutlass::make_cute_packed_stride(StrideD{}, {M_g, N, 1});
        }

        // Copy problem sizes into the staging buffer too
        std::memcpy(host_buf.data() + args.off_problem_sizes,
                     args.host_problem_sizes.data(), s_ps);

        // Single device allocation + single H2D transfer
        args.packed_device_buf = torch::empty({int64_t(total)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
        cudaMemcpyAsync(args.packed_device_buf.data_ptr(),
                         host_buf.data(), total,
                         cudaMemcpyHostToDevice, stream);

        return args;
    }
};

// ---------------------------------------------------------------------------
// Templated launch
// ---------------------------------------------------------------------------

template <typename KernelType>
torch::Tensor launch_grouped_gemm(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& tokens_per_expert,
    cudaStream_t stream,
    bool sort_by_m)
{
    using DeviceGemm  = typename KernelType::DeviceGemm;
    using ElementA    = typename KernelType::ElementA;
    using ElementB    = typename KernelType::ElementB;
    using ElementC    = typename KernelType::ElementC;
    using StrideA     = typename KernelType::StrideA;
    using StrideB     = typename KernelType::StrideB;
    using StrideC     = typename KernelType::StrideC;
    using StrideD     = typename KernelType::StrideD;
    using UnderlyingProblemShape = typename KernelType::ProblemShape::UnderlyingProblemShape;

    const int total_tokens = input.size(0);
    const int N = weights.size(1);

    auto output = torch::empty({total_tokens, N},
        torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    auto args = GroupedGemmArgs<KernelType>::prepare(
        input, weights, output, tokens_per_expert, stream, sort_by_m);

    cutlass::KernelHardwareInfo hw_info{};
    hw_info.device_id = input.device().index();
    hw_info.sm_count = get_sm_count(hw_info.device_id);

    typename DeviceGemm::Arguments gemm_args;
    decltype(gemm_args.epilogue.thread) fusion_args{};
    fusion_args.alpha = 1.0f;
    fusion_args.beta  = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr  = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array  = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta  = {cute::_0{}, cute::_0{}, 0};

    gemm_args = typename DeviceGemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {args.num_groups,
         args.template dev<UnderlyingProblemShape>(args.off_problem_sizes),
         args.host_problem_sizes.data()},
        {args.template dev<const ElementA*>(args.off_ptr_A),
         args.template dev<StrideA>(args.off_stride_A),
         args.template dev<const ElementB*>(args.off_ptr_B),
         args.template dev<StrideB>(args.off_stride_B)},
        {fusion_args,
         args.template dev<const ElementC*>(args.off_ptr_C),
         args.template dev<StrideC>(args.off_stride_C),
         args.template dev<ElementC*>(args.off_ptr_D),
         args.template dev<StrideD>(args.off_stride_D)},
        hw_info
    };

    DeviceGemm gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(gemm_args);

    // Workspace caching: reuse if current buffer is large enough
    static torch::Tensor s_workspace;
    static size_t s_workspace_cap = 0;
    if (workspace_size > s_workspace_cap || !s_workspace.defined() ||
        s_workspace.device() != input.device()) {
        size_t alloc_size = std::max(workspace_size, size_t(1));
        s_workspace = torch::empty({static_cast<int64_t>(alloc_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
        s_workspace_cap = alloc_size;
    }

    cutlass::Status status = gemm_op.initialize(gemm_args, s_workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS grouped GEMM initialize failed: ", cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS grouped GEMM run failed: ", cutlassGetStatusString(status));

    return output;
}

// ---------------------------------------------------------------------------
// Auto tile selection
// ---------------------------------------------------------------------------

static TileConfig auto_select_tile(
    const torch::Tensor& tokens_per_expert, int K, int N)
{
    const int G = tokens_per_expert.size(0);
    const auto* tpe = tokens_per_expert.data_ptr<int64_t>();
    int64_t total_tokens = 0;
    for (int g = 0; g < G; ++g) total_tokens += tpe[g];
    int avg_m = G > 0 ? static_cast<int>(total_tokens / G) : 0;

    // For small N, narrower tile avoids waste
    if (N < 256) {
        return TileConfig::Co_128x128x64;
    }

    // For large avg M/expert, Co_128x128x64 has lower per-tile overhead
    // and more tiles → better persistent scheduler utilization
    if (avg_m >= 4096 && N <= 2048) {
        return TileConfig::Co_128x128x64;
    }

    return TileConfig::Co_128x256x64;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

torch::Tensor grouped_gemm_forward(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& tokens_per_expert,
    TileConfig tile_config,
    bool sort_by_m)
{
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(weights.is_cuda(), "weights must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [total_tokens, K]");
    TORCH_CHECK(weights.dim() == 3, "weights must be 3D [num_experts, N, K]");
    TORCH_CHECK(tokens_per_expert.is_cpu(), "tokens_per_expert must be on CPU");
    TORCH_CHECK(input.size(1) == weights.size(2),
        "K mismatch: input=", input.size(1), " vs weights=", weights.size(2));
    TORCH_CHECK(input.is_contiguous() && weights.is_contiguous());

    c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int K = input.size(1);
    const int N = weights.size(1);

    if (tile_config == TileConfig::Auto) {
        tile_config = auto_select_tile(tokens_per_expert, K, N);
    }

    if (input.scalar_type() == torch::kBFloat16) {
        switch (tile_config) {
            case TileConfig::Co_128x128x64:
                return launch_grouped_gemm<BF16_128x128x64_Co>(input, weights, tokens_per_expert, stream, sort_by_m);
            case TileConfig::Co_128x256x64:
                return launch_grouped_gemm<BF16_128x256x64_Co>(input, weights, tokens_per_expert, stream, sort_by_m);
            case TileConfig::PP_128x128x128:
                return launch_grouped_gemm<BF16_128x128x128_PP>(input, weights, tokens_per_expert, stream, sort_by_m);
            case TileConfig::PP_128x256x64:
                return launch_grouped_gemm<BF16_128x256x64_PP>(input, weights, tokens_per_expert, stream, sort_by_m);
            default:
                TORCH_CHECK(false, "Invalid tile config");
        }
    } else if (input.scalar_type() == torch::kHalf) {
        switch (tile_config) {
            case TileConfig::Co_128x128x64:
                return launch_grouped_gemm<F16_128x128x64_Co>(input, weights, tokens_per_expert, stream, sort_by_m);
            case TileConfig::Co_128x256x64:
                return launch_grouped_gemm<F16_128x256x64_Co>(input, weights, tokens_per_expert, stream, sort_by_m);
            case TileConfig::PP_128x128x128:
                return launch_grouped_gemm<F16_128x128x128_PP>(input, weights, tokens_per_expert, stream, sort_by_m);
            case TileConfig::PP_128x256x64:
                return launch_grouped_gemm<F16_128x256x64_PP>(input, weights, tokens_per_expert, stream, sort_by_m);
            default:
                TORCH_CHECK(false, "Invalid tile config");
        }
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", input.scalar_type());
    }
}

}  // namespace grouped_gemm
