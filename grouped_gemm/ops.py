from enum import IntEnum
from typing import Optional

import torch

from grouped_gemm._C import grouped_gemm_forward


class TileConfig(IntEnum):
    """Tile size strategy for the CUTLASS persistent grouped GEMM kernel.

    Auto (default) selects based on average tokens per expert:
      - Small  (128x64x64):  avg tokens < 128
      - Medium (128x128x64): avg tokens in [128, 512)
      - Large  (128x256x64): avg tokens >= 512
    """
    SMALL  = 0
    MEDIUM = 1
    LARGE  = 2
    AUTO   = 3


def grouped_gemm(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    tile_config: TileConfig = TileConfig.AUTO,
) -> torch.Tensor:
    """MoE Grouped GEMM using CUTLASS 3.x SM90 Persistent Kernel.

    Computes: output[i] = input[i] @ weights[expert_of(i)].T
    for each token i, where tokens are sorted by expert assignment.

    Args:
        input:  [total_tokens, K] — activations, contiguous, tokens sorted by expert.
                Must be float16 or bfloat16.
        weights: [num_experts, N, K] — expert weights stored as N×K per expert.
                 This layout means each expert's weight is the transposed W matrix,
                 matching the col-major B convention.
        tokens_per_expert: [num_experts] — int64 CPU tensor.
                           sum(tokens_per_expert) must equal total_tokens.
        tile_config: Tile size selection. Auto is recommended.

    Returns:
        output: [total_tokens, N] — same dtype and device as input.
    """
    assert input.is_cuda and weights.is_cuda, "Tensors must be on CUDA"
    assert input.dim() == 2, f"input must be 2D, got {input.dim()}D"
    assert weights.dim() == 3, f"weights must be 3D, got {weights.dim()}D"

    if tokens_per_expert.is_cuda:
        tokens_per_expert = tokens_per_expert.cpu()
    tokens_per_expert = tokens_per_expert.to(torch.int64).contiguous()

    assert tokens_per_expert.sum().item() == input.size(0), (
        f"sum(tokens_per_expert)={tokens_per_expert.sum().item()} "
        f"!= total_tokens={input.size(0)}"
    )

    return grouped_gemm_forward(
        input.contiguous(),
        weights.contiguous(),
        tokens_per_expert,
        int(tile_config),
    )
