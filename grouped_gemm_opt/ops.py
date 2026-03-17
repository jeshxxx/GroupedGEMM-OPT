from enum import IntEnum

import torch

from grouped_gemm_opt._C import grouped_gemm_opt_forward


class TileConfig(IntEnum):
    """Tile/schedule config for grouped GEMM.

    CUTLASS Cooperative schedule:
      Co_128x128x64  — baseline, good for small avg M
      Co_128x256x64  — wider N tile

    CUTLASS Pingpong schedule:
      PP_128x128x128 — deep K tile + Cluster 2x1x1
      PP_128x256x64  — wide N tile with Pingpong

    cuBLAS sequential: one cuBLAS GEMM per expert (best for large avg M/expert)

    Auto: CUTLASS for small avg M/expert, cuBLAS for large avg M/expert
    """
    Co_128x128x64  = 0
    Co_128x256x64  = 1
    PP_128x128x128 = 2
    PP_128x256x64  = 3
    AUTO           = 4
    CuBLAS_Seq     = 5


def grouped_gemm_opt(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    tile_config: TileConfig = TileConfig.AUTO,
    sort_by_m: bool = True,
) -> torch.Tensor:
    """MoE Grouped GEMM using CUTLASS 3.x SM90 Persistent Kernel (optimized).

    Computes: output[i] = input[i] @ weights[expert_of(i)].T

    Args:
        input:  [total_tokens, K] — contiguous, tokens sorted by expert.
        weights: [num_experts, N, K] — expert weights, same shape as torch.nn.Linear (out_features, in_features).
        tokens_per_expert: [num_experts] int64 tensor (CPU or CUDA).
                           CUDA tensors use async D2H with stream-level sync,
                           avoiding the implicit cudaDeviceSynchronize that
                           .cpu() would trigger.
        tile_config: Kernel configuration. AUTO recommended.
        sort_by_m: Sort groups by descending token count before dispatch.
                   Zero-cost: only reorders pointer/stride arrays in C++,
                   no data movement. Improves persistent kernel load balance.

    Returns:
        output: [total_tokens, N]
    """
    assert input.is_cuda and weights.is_cuda
    assert input.dim() == 2 and weights.dim() == 3

    tokens_per_expert = tokens_per_expert.to(torch.int64).contiguous()

    if not tokens_per_expert.is_cuda:
        # CPU tokens_per_expert: cheap validation and zero-filtering on CPU
        assert tokens_per_expert.sum().item() == input.size(0), (
            f"sum(tokens_per_expert)={tokens_per_expert.sum().item()} "
            f"!= total_tokens={input.size(0)}"
        )
        nonzero_mask = tokens_per_expert > 0
        if not nonzero_mask.all():
            tokens_per_expert = tokens_per_expert[nonzero_mask]
            weights = weights[nonzero_mask]
        if tokens_per_expert.size(0) == 0:
            return torch.empty(0, weights.size(1) if weights.dim() == 3 else 0,
                               device=input.device, dtype=input.dtype)

    # GPU tokens_per_expert: validation and zero-filtering happen in C++
    # after async D2H, avoiding any host-device sync here.
    return grouped_gemm_opt_forward(
        input.contiguous(),
        weights.contiguous(),
        tokens_per_expert,
        int(tile_config),
        sort_by_m,
    )
