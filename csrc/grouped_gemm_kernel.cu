#include "grouped_gemm.h"
#include "grouped_gemm_sm90.cuh"

#include "cutlass/kernel_hardware_info.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cuda_runtime.h>
#include <cstring>
#include <vector>

namespace grouped_gemm {

// ---------------------------------------------------------------------------
// Packed argument buffer: all per-group arrays in ONE contiguous allocation
// and ONE cudaMemcpyAsync, eliminating 8 extra alloc+copy round trips.
// ---------------------------------------------------------------------------

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

    // Single device buffer holding all argument arrays contiguously
    torch::Tensor packed_device_buf;

    // Offsets into packed buffer (in bytes)
    size_t off_problem_sizes;
    size_t off_ptr_A, off_ptr_B, off_ptr_C, off_ptr_D;
    size_t off_stride_A, off_stride_B, off_stride_C, off_stride_D;

    template <typename T>
    T* dev_ptr(size_t byte_offset) {
        return reinterpret_cast<T*>(
            static_cast<char*>(packed_device_buf.data_ptr()) + byte_offset);
    }

    static size_t align_up(size_t x, size_t a = 16) {
        return (x + a - 1) & ~(a - 1);
    }

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
        const int G = args.num_groups;

        args.host_problem_sizes.resize(G);

        // Compute packed layout sizes (all arrays contiguous, 16B aligned)
        size_t sz_ps  = G * sizeof(UnderlyingProblemShape);
        size_t sz_pA  = G * sizeof(const ElementA*);
        size_t sz_pB  = G * sizeof(const ElementB*);
        size_t sz_pC  = G * sizeof(const ElementC*);
        size_t sz_pD  = G * sizeof(ElementC*);
        size_t sz_sA  = G * sizeof(StrideA);
        size_t sz_sB  = G * sizeof(StrideB);
        size_t sz_sC  = G * sizeof(StrideC);
        size_t sz_sD  = G * sizeof(StrideD);

        size_t cursor = 0;
        args.off_problem_sizes = cursor; cursor += align_up(sz_ps);
        args.off_ptr_A = cursor;         cursor += align_up(sz_pA);
        args.off_ptr_B = cursor;         cursor += align_up(sz_pB);
        args.off_ptr_C = cursor;         cursor += align_up(sz_pC);
        args.off_ptr_D = cursor;         cursor += align_up(sz_pD);
        args.off_stride_A = cursor;      cursor += align_up(sz_sA);
        args.off_stride_B = cursor;      cursor += align_up(sz_sB);
        args.off_stride_C = cursor;      cursor += align_up(sz_sC);
        args.off_stride_D = cursor;      cursor += align_up(sz_sD);
        size_t total_bytes = cursor;

        // Thread-local host buffer avoids re-allocation on every call.
        // For tiny buffers (few KB), pageable memory + sync copy is faster
        // than cudaMallocHost/cudaFreeHost which are expensive kernel-mode calls.
        thread_local std::vector<char> host_buf_storage;
        if (host_buf_storage.size() < total_bytes) {
            host_buf_storage.resize(total_bytes);
        }
        char* host_buf_raw = host_buf_storage.data();
        std::memset(host_buf_raw, 0, total_bytes);

        auto* h_ps = reinterpret_cast<UnderlyingProblemShape*>(host_buf_raw + args.off_problem_sizes);
        auto* h_pA = reinterpret_cast<const ElementA**>(host_buf_raw + args.off_ptr_A);
        auto* h_pB = reinterpret_cast<const ElementB**>(host_buf_raw + args.off_ptr_B);
        auto* h_pC = reinterpret_cast<const ElementC**>(host_buf_raw + args.off_ptr_C);
        auto* h_pD = reinterpret_cast<ElementC**>(host_buf_raw + args.off_ptr_D);
        auto* h_sA = reinterpret_cast<StrideA*>(host_buf_raw + args.off_stride_A);
        auto* h_sB = reinterpret_cast<StrideB*>(host_buf_raw + args.off_stride_B);
        auto* h_sC = reinterpret_cast<StrideC*>(host_buf_raw + args.off_stride_C);
        auto* h_sD = reinterpret_cast<StrideD*>(host_buf_raw + args.off_stride_D);

        const auto* tpe = tokens_per_expert.data_ptr<int64_t>();
        int64_t off = 0;

        for (int g = 0; g < G; ++g) {
            int M_g = static_cast<int>(tpe[g]);
            args.host_problem_sizes[g] = cute::make_shape(M_g, N, K);
            h_ps[g] = args.host_problem_sizes[g];

            h_pA[g] = reinterpret_cast<const ElementA*>(input.data_ptr()) + off * K;
            h_pB[g] = reinterpret_cast<const ElementB*>(weights.data_ptr()) + int64_t(g) * N * K;
            h_pC[g] = reinterpret_cast<const ElementC*>(output.data_ptr()) + off * N;
            h_pD[g] = reinterpret_cast<ElementC*>(output.data_ptr()) + off * N;

            h_sA[g] = cutlass::make_cute_packed_stride(StrideA{}, {M_g, K, 1});
            h_sB[g] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
            h_sC[g] = cutlass::make_cute_packed_stride(StrideC{}, {M_g, N, 1});
            h_sD[g] = cutlass::make_cute_packed_stride(StrideD{}, {M_g, N, 1});

            off += M_g;
        }

        // Cached device buffer — avoids torch::empty overhead on repeat calls
        thread_local torch::Tensor cached_dev_buf;
        thread_local size_t cached_dev_size = 0;
        if (total_bytes > cached_dev_size) {
            cached_dev_buf = torch::empty({static_cast<int64_t>(total_bytes)},
                torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
            cached_dev_size = total_bytes;
        }
        args.packed_device_buf = cached_dev_buf;
        cudaMemcpyAsync(args.packed_device_buf.data_ptr(), host_buf_raw,
                        total_bytes, cudaMemcpyHostToDevice, stream);

        return args;
    }
};

// ---------------------------------------------------------------------------
// Templated launch
// ---------------------------------------------------------------------------

// Per-kernel-type cached state to avoid repeated allocations and driver queries
template <typename KernelType>
struct KernelCache {
    static int sm_count;
    static size_t workspace_capacity;
    static torch::Tensor workspace;
    static bool initialized;

    static void ensure_init(int device_id, torch::Device device) {
        if (!initialized) {
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
            workspace_capacity = 0;
            initialized = true;
        }
    }

    static void* get_workspace(size_t needed, torch::Device device) {
        if (needed > workspace_capacity) {
            workspace = torch::empty({static_cast<int64_t>(std::max(needed, size_t(1)))},
                torch::TensorOptions().dtype(torch::kUInt8).device(device));
            workspace_capacity = needed;
        }
        return workspace.data_ptr();
    }
};

template <typename KT> int KernelCache<KT>::sm_count = 0;
template <typename KT> size_t KernelCache<KT>::workspace_capacity = 0;
template <typename KT> torch::Tensor KernelCache<KT>::workspace;
template <typename KT> bool KernelCache<KT>::initialized = false;

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
    using UnderlyingProblemShape = typename KernelType::ProblemShape::UnderlyingProblemShape;
    using Cache = KernelCache<KernelType>;

    const int total_tokens = input.size(0);
    const int N = weights.size(1);
    const int device_id = input.device().index();

    Cache::ensure_init(device_id, input.device());

    auto output = torch::empty({total_tokens, N},
        torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    auto args = GroupedGemmArgs<KernelType>::prepare(
        input, weights, output, tokens_per_expert, stream);

    cutlass::KernelHardwareInfo hw_info{device_id, Cache::sm_count};

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
         args.template dev_ptr<UnderlyingProblemShape>(args.off_problem_sizes),
         args.host_problem_sizes.data()},
        {args.template dev_ptr<const ElementA*>(args.off_ptr_A),
         args.template dev_ptr<StrideA>(args.off_stride_A),
         args.template dev_ptr<const ElementB*>(args.off_ptr_B),
         args.template dev_ptr<StrideB>(args.off_stride_B)},
        {fusion_args,
         args.template dev_ptr<const ElementC*>(args.off_ptr_C),
         args.template dev_ptr<StrideC>(args.off_stride_C),
         args.template dev_ptr<ElementC*>(args.off_ptr_D),
         args.template dev_ptr<StrideD>(args.off_stride_D)},
        hw_info
    };

    DeviceGemm gemm_op;

    size_t ws_size = gemm_op.get_workspace_size(gemm_args);
    void* ws_ptr = Cache::get_workspace(ws_size, input.device());

    cutlass::Status status = gemm_op.initialize(gemm_args, ws_ptr, stream);
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
    if (N < 256) {
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
    TileConfig tile_config)
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
                return launch_grouped_gemm<BF16_128x128x64_Co>(input, weights, tokens_per_expert, stream);
            case TileConfig::Co_128x256x64:
                return launch_grouped_gemm<BF16_128x256x64_Co>(input, weights, tokens_per_expert, stream);
            case TileConfig::PP_128x128x128:
                return launch_grouped_gemm<BF16_128x128x128_PP>(input, weights, tokens_per_expert, stream);
            case TileConfig::PP_128x256x64:
                return launch_grouped_gemm<BF16_128x256x64_PP>(input, weights, tokens_per_expert, stream);
            default:
                TORCH_CHECK(false, "Invalid tile config");
        }
    } else if (input.scalar_type() == torch::kHalf) {
        switch (tile_config) {
            case TileConfig::Co_128x128x64:
                return launch_grouped_gemm<F16_128x128x64_Co>(input, weights, tokens_per_expert, stream);
            case TileConfig::Co_128x256x64:
                return launch_grouped_gemm<F16_128x256x64_Co>(input, weights, tokens_per_expert, stream);
            case TileConfig::PP_128x128x128:
                return launch_grouped_gemm<F16_128x128x128_PP>(input, weights, tokens_per_expert, stream);
            case TileConfig::PP_128x256x64:
                return launch_grouped_gemm<F16_128x256x64_PP>(input, weights, tokens_per_expert, stream);
            default:
                TORCH_CHECK(false, "Invalid tile config");
        }
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", input.scalar_type());
    }
}

}  // namespace grouped_gemm
