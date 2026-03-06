#include "grouped_gemm.h"
#include <torch/extension.h>

using namespace grouped_gemm;

torch::Tensor grouped_gemm_forward_wrapper(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& tokens_per_expert,
    int64_t tile_config)
{
    return grouped_gemm_forward(
        input, weights, tokens_per_expert,
        static_cast<TileConfig>(tile_config));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_gemm_forward", &grouped_gemm_forward_wrapper,
        "CUTLASS 3.x SM90 Persistent Grouped GEMM for MoE",
        py::arg("input"),
        py::arg("weights"),
        py::arg("tokens_per_expert"),
        py::arg("tile_config") = 4  /* Auto */);
}
