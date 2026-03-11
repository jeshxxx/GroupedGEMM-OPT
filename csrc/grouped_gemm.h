#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>

namespace grouped_gemm {

enum class TileConfig {
    // Cooperative schedule
    Co_128x128x64  = 0,
    Co_128x256x64  = 1,
    // Pingpong schedule (faster pipeline, ~10-15% speedup)
    PP_128x128x128 = 2,
    PP_128x256x64  = 3,
    // Auto: selects best config based on problem dimensions
    Auto           = 4,
};

torch::Tensor grouped_gemm_forward(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& tokens_per_expert,
    TileConfig tile_config = TileConfig::Auto,
    bool sort_by_m = true);

}  // namespace grouped_gemm
