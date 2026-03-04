#include "grouped_gemm.h"
#include "grouped_gemm_sm90.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cuda_runtime.h>
#include <vector>

namespace grouped_gemm {

// ---------------------------------------------------------------------------
// Helper: build device-side argument arrays for CUTLASS grouped GEMM
//
// CUTLASS grouped GEMM expects per-group arrays of:
//   - problem_sizes: cute::Shape<int,int,int> per group
//   - ptr_A / ptr_B / ptr_C / ptr_D: element pointers per group
//   - stride_A / stride_B / stride_C / stride_D: stride tuples per group
//
// For MoE, the input tensor is contiguous (tokens sorted by expert), so we
// compute pointer offsets from the cumulative token count.
// ---------------------------------------------------------------------------

template <typename KernelType>
struct GroupedGemmArgs {
    using ElementA = typename KernelType::ElementA;
    using ElementB = typename KernelType::ElementB;
    using ElementC = typename KernelType::ElementC;

    using StrideA = typename KernelType::StrideA;
    using StrideB = typename KernelType::StrideB;
    using StrideC = typename KernelType::StrideC;

    using ProblemShape = typename KernelType::ProblemShape;
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    int num_groups;

    // Device arrays
    torch::Tensor problem_sizes_tensor;
    torch::Tensor ptr_A_tensor;
    torch::Tensor ptr_B_tensor;
    torch::Tensor ptr_C_tensor;
    torch::Tensor ptr_D_tensor;
    torch::Tensor stride_A_tensor;
    torch::Tensor stride_B_tensor;
    torch::Tensor stride_C_tensor;
    torch::Tensor stride_D_tensor;

    static GroupedGemmArgs prepare(
        const torch::Tensor& input,       // [total_tokens, K]
        const torch::Tensor& weights,     // [num_experts, N, K] col-major (stored as N×K row)
        torch::Tensor& output,            // [total_tokens, N]
        const torch::Tensor& tokens_per_expert,  // [num_experts] on CPU
        cudaStream_t stream)
    {
        GroupedGemmArgs args;
        args.num_groups = tokens_per_expert.size(0);

        const int K = input.size(1);
        const int N = weights.size(1);

        auto opts_int = torch::TensorOptions().dtype(torch::kInt64).device(input.device());
        auto opts_ptr = torch::TensorOptions().dtype(torch::kInt64).device(input.device());

        // Build on CPU then copy to device in one shot
        auto cpu_opts_int = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);

        // Problem sizes: (M_i, N, K) per group
        std::vector<UnderlyingProblemShape> h_problem_sizes(args.num_groups);

        // Pointers
        std::vector<const ElementA*> h_ptr_A(args.num_groups);
        std::vector<const ElementB*> h_ptr_B(args.num_groups);
        std::vector<const ElementC*> h_ptr_C(args.num_groups);
        std::vector<ElementC*>       h_ptr_D(args.num_groups);

        // Strides
        std::vector<StrideA> h_stride_A(args.num_groups);
        std::vector<StrideB> h_stride_B(args.num_groups);
        std::vector<StrideC> h_stride_C(args.num_groups);
        std::vector<StrideC> h_stride_D(args.num_groups);

        const auto* tpe_ptr = tokens_per_expert.data_ptr<int64_t>();
        int64_t token_offset = 0;

        for (int g = 0; g < args.num_groups; ++g) {
            int M_g = static_cast<int>(tpe_ptr[g]);

            // Problem size for this group
            h_problem_sizes[g] = cute::make_shape(M_g, N, K);

            // A: input[token_offset : token_offset + M_g, :]  → row-major [M_g, K]
            h_ptr_A[g] = reinterpret_cast<const ElementA*>(
                input.data_ptr()) + token_offset * K;

            // B: weights[g]  → stored as [N, K] contiguous = col-major [K, N]
            h_ptr_B[g] = reinterpret_cast<const ElementB*>(
                weights.data_ptr()) + g * N * K;

            // C/D: output[token_offset : token_offset + M_g, :] → row-major [M_g, N]
            h_ptr_C[g] = reinterpret_cast<const ElementC*>(
                output.data_ptr()) + token_offset * N;
            h_ptr_D[g] = reinterpret_cast<ElementC*>(
                output.data_ptr()) + token_offset * N;

            // Strides: CUTLASS uses cute::Stride for the 3 modes (M/K, K/N, L)
            // Row-major A [M,K]: stride = (K, 1, 0)
            h_stride_A[g] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M_g, K, 1));
            // Col-major B [K,N]: stride = (1, K, 0) — but B is stored as [N,K] row-major
            h_stride_B[g] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
            // Row-major C/D [M,N]: stride = (N, 1, 0)
            h_stride_C[g] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M_g, N, 1));
            h_stride_D[g] = h_stride_C[g];

            token_offset += M_g;
        }

        // Allocate device tensors and copy
        auto copy_to_device = [&](const void* src, size_t bytes) -> torch::Tensor {
            auto t = torch::empty({static_cast<int64_t>(bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
            cudaMemcpyAsync(t.data_ptr(), src, bytes, cudaMemcpyHostToDevice, stream);
            return t;
        };

        args.problem_sizes_tensor = copy_to_device(
            h_problem_sizes.data(),
            h_problem_sizes.size() * sizeof(UnderlyingProblemShape));

        args.ptr_A_tensor = copy_to_device(
            h_ptr_A.data(), h_ptr_A.size() * sizeof(const ElementA*));
        args.ptr_B_tensor = copy_to_device(
            h_ptr_B.data(), h_ptr_B.size() * sizeof(const ElementB*));
        args.ptr_C_tensor = copy_to_device(
            h_ptr_C.data(), h_ptr_C.size() * sizeof(const ElementC*));
        args.ptr_D_tensor = copy_to_device(
            h_ptr_D.data(), h_ptr_D.size() * sizeof(ElementC*));

        args.stride_A_tensor = copy_to_device(
            h_stride_A.data(), h_stride_A.size() * sizeof(StrideA));
        args.stride_B_tensor = copy_to_device(
            h_stride_B.data(), h_stride_B.size() * sizeof(StrideB));
        args.stride_C_tensor = copy_to_device(
            h_stride_C.data(), h_stride_C.size() * sizeof(StrideC));
        args.stride_D_tensor = copy_to_device(
            h_stride_D.data(), h_stride_D.size() * sizeof(StrideC));

        return args;
    }
};

// ---------------------------------------------------------------------------
// Templated launch function for a specific kernel type
// ---------------------------------------------------------------------------

template <typename KernelType>
torch::Tensor launch_grouped_gemm(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& tokens_per_expert,
    cudaStream_t stream)
{
    using DeviceGemm = typename KernelType::DeviceGemm;
    using GemmKernel = typename KernelType::GemmKernel;
    using ElementC   = typename KernelType::ElementC;
    using StrideA    = typename KernelType::StrideA;
    using StrideB    = typename KernelType::StrideB;
    using StrideC    = typename KernelType::StrideC;
    using ProblemShape = typename KernelType::ProblemShape;
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    const int total_tokens = input.size(0);
    const int N = weights.size(1);

    auto output = torch::empty({total_tokens, N},
        torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    auto args = GroupedGemmArgs<KernelType>::prepare(
        input, weights, output, tokens_per_expert, stream);

    // Construct CUTLASS arguments
    typename DeviceGemm::Arguments gemm_args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        // Problem shape
        {args.num_groups,
         reinterpret_cast<UnderlyingProblemShape*>(args.problem_sizes_tensor.data_ptr()),
         nullptr},  // host-side problem sizes (nullptr = device only)
        // Mainloop arguments: {ptr_A, stride_A, ptr_B, stride_B}
        {reinterpret_cast<const typename KernelType::ElementA**>(args.ptr_A_tensor.data_ptr()),
         reinterpret_cast<StrideA*>(args.stride_A_tensor.data_ptr()),
         reinterpret_cast<const typename KernelType::ElementB**>(args.ptr_B_tensor.data_ptr()),
         reinterpret_cast<StrideB*>(args.stride_B_tensor.data_ptr())},
        // Epilogue arguments: {{alpha, beta}, ptr_C, stride_C, ptr_D, stride_D}
        {{1.0f, 0.0f},
         reinterpret_cast<const ElementC**>(args.ptr_C_tensor.data_ptr()),
         reinterpret_cast<StrideC*>(args.stride_C_tensor.data_ptr()),
         reinterpret_cast<ElementC**>(args.ptr_D_tensor.data_ptr()),
         reinterpret_cast<StrideC*>(args.stride_D_tensor.data_ptr())}
    };

    DeviceGemm gemm_op;

    // Query workspace size
    size_t workspace_size = gemm_op.get_workspace_size(gemm_args);
    auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));

    // Initialize (includes TMA descriptor creation on Hopper)
    cutlass::Status status = gemm_op.initialize(gemm_args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
        "CUTLASS grouped GEMM initialize failed: ",
        cutlassGetStatusString(status));

    // Run
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
// Public API: dispatch based on dtype and tile config
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

    // Dispatch based on dtype × tile config
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
