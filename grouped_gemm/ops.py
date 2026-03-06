from enum import IntEnum
from typing import Optional

import torch

from grouped_gemm._C import grouped_gemm_forward


class TileConfig(IntEnum):
    """Tile/schedule config for CUTLASS persistent grouped GEMM.

    Cooperative schedule:
      Co_128x128x64  — baseline, good for small avg M
      Co_128x256x64  — wider N tile

    Pingpong schedule (faster pipeline, ~10-15% over Cooperative):
      PP_128x128x128 — deep K tile + Cluster 2x1x1, best for large K
      PP_128x256x64  — wide N tile with Pingpong

    Auto: selects best based on K, N, avg tokens/expert
    """
    Co_128x128x64  = 0
    Co_128x256x64  = 1
    PP_128x128x128 = 2
    PP_128x256x64  = 3
    AUTO           = 4


def grouped_gemm(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    tile_config: TileConfig = TileConfig.AUTO,
    sort_by_m: bool = True,
) -> torch.Tensor:
    """MoE Grouped GEMM using CUTLASS 3.x SM90 Persistent Kernel.

    Computes: output[i] = input[i] @ weights[expert_of(i)].T

    Args:
        input:  [total_tokens, K] — contiguous, tokens sorted by expert.
        weights: [num_experts, N, K] — expert weights (transposed layout).
        tokens_per_expert: [num_experts] int64 tensor.
        tile_config: Kernel configuration. AUTO recommended.
        sort_by_m: Sort experts by descending token count before dispatch.
                   Improves persistent kernel load balance by ~2-5%.

    Returns:
        output: [total_tokens, N]
    """
    assert input.is_cuda and weights.is_cuda
    assert input.dim() == 2 and weights.dim() == 3

    if tokens_per_expert.is_cuda:
        tokens_per_expert = tokens_per_expert.cpu()
    tokens_per_expert = tokens_per_expert.to(torch.int64).contiguous()

    assert tokens_per_expert.sum().item() == input.size(0), (
        f"sum(tokens_per_expert)={tokens_per_expert.sum().item()} "
        f"!= total_tokens={input.size(0)}"
    )

    if sort_by_m and tokens_per_expert.size(0) > 1:
        input, weights, tokens_per_expert = _sort_by_descending_m(
            input, weights, tokens_per_expert)

    return grouped_gemm_forward(
        input.contiguous(),
        weights.contiguous(),
        tokens_per_expert,
        int(tile_config),
    )


def _sort_by_descending_m(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
):
    """Sort experts by descending token count for better load balance.

    The persistent tile scheduler assigns tiles round-robin. When large
    experts appear first, tiles with heavy K-dimension work are spread
    evenly across CTAs, reducing the tail effect.

    This reorders the input segments and weight matrices accordingly.
    """
    sorted_indices = torch.argsort(tokens_per_expert, descending=True)

    if torch.equal(sorted_indices, torch.arange(len(sorted_indices))):
        return input, weights, tokens_per_expert

    tpe_sorted = tokens_per_expert[sorted_indices]
    weights_sorted = weights[sorted_indices]

    offsets = torch.zeros(len(tokens_per_expert) + 1, dtype=torch.int64)
    torch.cumsum(tokens_per_expert, 0, out=offsets[1:])

    new_offsets = torch.zeros_like(offsets)
    torch.cumsum(tpe_sorted, 0, out=new_offsets[1:])

    gather_idx = torch.empty(input.size(0), dtype=torch.int64, device=input.device)
    new_off = 0
    for i, orig_idx in enumerate(sorted_indices.tolist()):
        m = tokens_per_expert[orig_idx].item()
        if m > 0:
            orig_start = offsets[orig_idx].item()
            gather_idx[new_off:new_off + m] = torch.arange(
                orig_start, orig_start + m, device=input.device)
        new_off += m

    input_sorted = input[gather_idx]
    return input_sorted, weights_sorted, tpe_sorted
