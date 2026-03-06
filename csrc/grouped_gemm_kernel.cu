#include "grouped_gemm.h"
#include "grouped_gemm_sm90.cuh"

#include "cutlass/kernel_hardware_info.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cuda_runtime.h>
#include <vector>

namespace grouped_gemm {

// ---------------------------------------------------------------------------
// Helper: build device-side arrays for CUTLASS grouped GEMM.
//
// Following CUTLASS example 57_hopper_grouped_gemm, the Ptr-Array grouped
// GEMM with pointer-decorated layouts expects per-group device arrays for
// BOTH pointers AND strides.
// ---------------------------------------------------------------------------

template <typename KernelType>
struct GroupedGemmArgs {
    using ElementA = typename KernelType::ElementA;
    using ElementB = typename KernelType::ElementB;
    using ElementC = typename KernelType::ElementC;

    using StrideA = typename KernelType::StrideA;
    using StrideB = typename KernelType::StrideB;
    using StrideC = typename KernelType::StrideC;
    using StrideD = typename KernelType::StrideD;

    using ProblemShape = typename KernelType::ProblemShape;
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    int num_groups;

    // Host-side problem sizes — must stay alive through initialize() + run()
    std::vector<UnderlyingProblemShape> host_problem_sizes;

    // Device arrays (ref-counted by torch::Tensor)
    torch::Tensor problem_sizes_device;
    torch::Tensor ptr_A_device;
    torch::Tensor ptr_B_device;
    torch::Tensor ptr_C_device;
    torch::Tensor ptr_D_device;
    torch::Tensor stride_A_device;
    torch::Tensor stride_B_device;
    torch::Tensor stride_C_device;
    torch::Tensor stride_D_device;

    static GroupedGemmArgs prepare(
        const torch::Tensor& input,
        const torch::Tensor& weights,
        torch::Tensor& output,
        const torch::Tensor& tokens_per_expert,
        cudaStream_t stream)
    {
        GroupedGemmArgs args;
        args.num_groups = tokens_per_expert.size(0);

        const int K = input.size(1);
        const int N = weights.size(1);

        args.host_problem_sizes.resize(args.num_groups);

        std::vector<const ElementA*> h_ptr_A(args.num_groups);
        std::vector<const ElementB*> h_ptr_B(args.num_groups);
        std::vector<const ElementC*> h_ptr_C(args.num_groups);
        std::vector<ElementC*>       h_ptr_D(args.num_groups);

        std::vector<StrideA> h_stride_A(args.num_groups);
        std::vector<StrideB> h_stride_B(args.num_groups);
        std::vector<StrideC> h_stride_C(args.num_groups);
        std::vector<StrideD> h_stride_D(args.num_groups);

        const auto* tpe_ptr = tokens_per_expert.data_ptr<int64_t>();
        int64_t token_offset = 0;

        for (int g = 0; g < args.num_groups; ++g) {
            int M_g = static_cast<int>(tpe_ptr[g]);

            args.host_problem_sizes[g] = cute::make_shape(M_g, N, K);

            h_ptr_A[g] = reinterpret_cast<const ElementA*>(
                input.data_ptr()) + token_offset * K;
            h_ptr_B[g] = reinterpret_cast<const ElementB*>(
                weights.data_ptr()) + static_cast<int64_t>(g) * N * K;
            h_ptr_C[g] = reinterpret_cast<const ElementC*>(
                output.data_ptr()) + token_offset * N;
            h_ptr_D[g] = reinterpret_cast<ElementC*>(
                output.data_ptr()) + token_offset * N;

            h_stride_A[g] = cutlass::make_cute_packed_stride(StrideA{}, {M_g, K, 1});
            h_stride_B[g] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
            h_stride_C[g] = cutlass::make_cute_packed_stride(StrideC{}, {M_g, N, 1});
            h_stride_D[g] = cutlass::make_cute_packed_stride(StrideD{}, {M_g, N, 1});

            token_offset += M_g;
        }

        auto copy_to_device = [&](const void* src, size_t bytes) -> torch::Tensor {
            auto t = torch::empty({static_cast<int64_t>(bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
            cudaMemcpyAsync(t.data_ptr(), src, bytes, cudaMemcpyHostToDevice, stream);
            return t;
        };

        args.problem_sizes_device = copy_to_device(
            args.host_problem_sizes.data(),
            args.host_problem_sizes.size() * sizeof(UnderlyingProblemShape));

        args.ptr_A_device = copy_to_device(h_ptr_A.data(), h_ptr_A.size() * sizeof(const ElementA*));
        args.ptr_B_device = copy_to_device(h_ptr_B.data(), h_ptr_B.size() * sizeof(const ElementB*));
        args.ptr_C_device = copy_to_device(h_ptr_C.data(), h_ptr_C.size() * sizeof(const ElementC*));
        args.ptr_D_device = copy_to_device(h_ptr_D.data(), h_ptr_D.size() * sizeof(ElementC*));

        args.stride_A_device = copy_to_device(h_stride_A.data(), h_stride_A.size() * sizeof(StrideA));
        args.stride_B_device = copy_to_device(h_stride_B.data(), h_stride_B.size() * sizeof(StrideB));
        args.stride_C_device = copy_to_device(h_stride_C.data(), h_stride_C.size() * sizeof(StrideC));
        args.stride_D_device = copy_to_device(h_stride_D.data(), h_stride_D.size() * sizeof(StrideD));

        return args;
    }
};

// ---------------------------------------------------------------------------
// Templated launch function
// ---------------------------------------------------------------------------

template <typename KernelType>
torch::Tensor launch_grouped_gemm(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& tokens_per_expert,
    cudaStream_t stream)
{
    using DeviceGemm  = typename KernelType::DeviceGemm;
    using ElementA    = typename KernelType::ElementA;
    using ElementB    = typename KernelType::ElementB;
    using ElementC    = typename KernelType::ElementC;
    using StrideA     = typename KernelType::StrideA;
    using StrideB     = typename KernelType::StrideB;
    using StrideC     = typename KernelType::StrideC;
    using StrideD     = typename KernelType::StrideD;
    using ProblemShape          = typename KernelType::ProblemShape;
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    const int total_tokens = input.size(0);
    const int N = weights.size(1);

    auto output = torch::empty({total_tokens, N},
        torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    auto args = GroupedGemmArgs<KernelType>::prepare(
        input, weights, output, tokens_per_expert, stream);

    cutlass::KernelHardwareInfo hw_info{};
    hw_info.device_id = input.device().index();
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hw_info.device_id);

    // Build epilogue fusion arguments (scalar alpha=1, beta=0 for all groups)
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

    // Construct arguments exactly matching CUTLASS example 57:
    //   {mode, problem_shape, mainloop_args, epilogue_args, hw_info}
    gemm_args = typename DeviceGemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {args.num_groups,
         reinterpret_cast<UnderlyingProblemShape*>(args.problem_sizes_device.data_ptr()),
         args.host_problem_sizes.data()},
        {reinterpret_cast<const ElementA**>(args.ptr_A_device.data_ptr()),
         reinterpret_cast<StrideA*>(args.stride_A_device.data_ptr()),
         reinterpret_cast<const ElementB**>(args.ptr_B_device.data_ptr()),
         reinterpret_cast<StrideB*>(args.stride_B_device.data_ptr())},
        {fusion_args,
         reinterpret_cast<const ElementC**>(args.ptr_C_device.data_ptr()),
         reinterpret_cast<StrideC*>(args.stride_C_device.data_ptr()),
         reinterpret_cast<ElementC**>(args.ptr_D_device.data_ptr()),
         reinterpret_cast<StrideD*>(args.stride_D_device.data_ptr())},
        hw_info
    };

    DeviceGemm gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(gemm_args);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));

    cutlass::Status status = gemm_op.initialize(gemm_args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS grouped GEMM initialize failed: ",
        cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS grouped GEMM run failed: ",
        cutlassGetStatusString(status));

    return output;
}

// ---------------------------------------------------------------------------
// Auto tile selection based on average tokens per expert
// ---------------------------------------------------------------------------

static TileConfig auto_select_tile(const torch::Tensor& tokens_per_expert) {
    const auto* tpe_ptr = tokens_per_expert.data_ptr<int64_t>();
    int64_t total = 0;
    int64_t num_groups = tokens_per_expert.size(0);
    for (int64_t i = 0; i < num_groups; ++i) {
        total += tpe_ptr[i];
    }
    int64_t avg = total / std::max(num_groups, int64_t(1));

    if (avg < 128) return TileConfig::Small;
    if (avg < 512) return TileConfig::Medium;
    return TileConfig::Large;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

torch::Tensor grouped_gemm_forward(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& tokens_per_expert,
    TileConfig tile_config)
{
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(weights.is_cuda(), "weights must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [total_tokens, K]");
    TORCH_CHECK(weights.dim() == 3, "weights must be 3D [num_experts, N, K]");
    TORCH_CHECK(tokens_per_expert.is_cpu(), "tokens_per_expert must be on CPU");
    TORCH_CHECK(input.size(1) == weights.size(2),
        "K dimension mismatch: input K=", input.size(1),
        " vs weights K=", weights.size(2));
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");

    c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (tile_config == TileConfig::Auto) {
        tile_config = auto_select_tile(tokens_per_expert);
    }

    if (input.scalar_type() == torch::kHalf) {
        switch (tile_config) {
            case TileConfig::Small:
                return launch_grouped_gemm<GroupedGemmF16_64x128>(
                    input, weights, tokens_per_expert, stream);
            case TileConfig::Medium:
                return launch_grouped_gemm<GroupedGemmF16_128x128>(
                    input, weights, tokens_per_expert, stream);
            case TileConfig::Large:
                return launch_grouped_gemm<GroupedGemmF16_128x256>(
                    input, weights, tokens_per_expert, stream);
            default:
                TORCH_CHECK(false, "Invalid tile config");
        }
    } else if (input.scalar_type() == torch::kBFloat16) {
        switch (tile_config) {
            case TileConfig::Small:
                return launch_grouped_gemm<GroupedGemmBF16_64x128>(
                    input, weights, tokens_per_expert, stream);
            case TileConfig::Medium:
                return launch_grouped_gemm<GroupedGemmBF16_128x128>(
                    input, weights, tokens_per_expert, stream);
            case TileConfig::Large:
                return launch_grouped_gemm<GroupedGemmBF16_128x256>(
                    input, weights, tokens_per_expert, stream);
            default:
                TORCH_CHECK(false, "Invalid tile config");
        }
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", input.scalar_type(),
            ". Only float16 and bfloat16 are supported.");
    }
}

}  // namespace grouped_gemm
