#pragma once

#include <torch/torch.h>
#include <cuda_runtime.h>

namespace grouped_gemm {

// Tile size selection strategy for MoE workloads
enum class TileConfig {
    Small  = 0,   // 64x128x64:  per-expert M < 128
    Medium = 1,   // 128x128x64: per-expert M in [128, 512) — default
    Large  = 2,   // 128x256x64: per-expert M >= 512
    Auto   = 3,   // Choose based on average tokens per expert
};

// Launch the CUTLASS 3.x SM90 persistent grouped GEMM.
//
// Args:
//   input:            [total_tokens, K] — permuted activations (tokens sorted by expert)
//   weights:          [num_experts, N, K] — expert weight matrices (column-major: stored as N×K)
//   tokens_per_expert:[num_experts] on CPU — number of tokens assigned to each expert
//   tile_config:      tile size selection (Auto recommended)
//
// Returns:
//   output:           [total_tokens, N]
torch::Tensor grouped_gemm_forward(
    const torch::Tensor& input,
    const torch::Tensor& weights,
    const torch::Tensor& tokens_per_expert,
    TileConfig tile_config = TileConfig::Auto);

}  // namespace grouped_gemm
