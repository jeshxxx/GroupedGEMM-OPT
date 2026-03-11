"""
Triton Fused MoE GEMM kernel — zero tile-padding, per-token expert dispatch.

Tokens are pre-sorted by expert_id. Each tile may span multiple experts.
For each expert present in a tile, the kernel masks the relevant rows,
loads that expert's weight, and accumulates via tl.dot.

This is correct for any token distribution and number of experts.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_moe_kernel(
    A_ptr, B_ptr, C_ptr,
    sorted_ids_ptr,   # [total_tokens] int32 — gather indices for A/C
    expert_ids_ptr,   # [total_tokens] int32 — expert_id per sorted position
    offsets_ptr,       # [num_experts + 1] int32 — cumulative token offsets per expert
    total_tokens,
    N, K,
    stride_an, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cn, stride_ck,
    num_experts,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_EXPERTS_PER_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    tile_start = pid_m * BLOCK_M
    m_range = tile_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < total_tokens

    token_ids = tl.load(sorted_ids_ptr + m_range, mask=m_mask, other=0)
    row_expert = tl.load(expert_ids_ptr + m_range, mask=m_mask, other=-1)

    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Find the range of experts in this tile
    first_expert = tl.load(expert_ids_ptr + tl.minimum(tile_start, total_tokens - 1))
    last_idx = tl.minimum(tile_start + BLOCK_M - 1, total_tokens - 1)
    last_expert = tl.load(expert_ids_ptr + last_idx)

    # Iterate over experts present in this tile (sorted → contiguous)
    for g in tl.static_range(MAX_EXPERTS_PER_TILE):
        e = first_expert + g
        still_active = e <= last_expert
        # Mask: rows belonging to expert e
        mask_e = (row_expert == e) & m_mask

        for k_start in range(0, K, BLOCK_K):
            k_range = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_range < K

            # Load A: gather by token_id, zero out rows not belonging to expert e
            a = tl.load(A_ptr + token_ids[:, None] * stride_an + k_range[None, :] * stride_ak,
                         mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            a = tl.where(mask_e[:, None] & still_active, a, 0.0)

            # Load B[e]: transposed view [K, N] for tl.dot
            b = tl.load(B_ptr + e * stride_be + k_range[:, None] * stride_bk + n_range[None, :] * stride_bn,
                         mask=k_mask[:, None] & n_mask[None, :] & still_active, other=0.0)

            acc += tl.dot(a, b)

    # Store C: scatter by token_id
    tl.store(C_ptr + token_ids[:, None] * stride_cn + n_range[None, :] * stride_ck,
             acc.to(C_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


def _prepare_expert_mapping(tokens_per_expert: torch.Tensor, device: torch.device):
    """Build sorted token indices, per-position expert IDs, and cumulative offsets."""
    num_experts = tokens_per_expert.size(0)
    total_tokens = int(tokens_per_expert.sum().item())

    sorted_ids = torch.empty(total_tokens, dtype=torch.int32, device=device)
    expert_ids = torch.empty(total_tokens, dtype=torch.int32, device=device)

    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    offset = 0
    for e in range(num_experts):
        m = int(tokens_per_expert[e].item())
        offsets[e] = offset
        if m > 0:
            sorted_ids[offset:offset + m] = torch.arange(
                offset, offset + m, dtype=torch.int32, device=device)
            expert_ids[offset:offset + m] = e
        offset += m
    offsets[num_experts] = offset

    return sorted_ids, expert_ids, offsets


def triton_fused_moe(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 64,
) -> torch.Tensor:
    """Triton Fused MoE GEMM — zero padding, handles multi-expert tiles correctly."""
    total_tokens = input.size(0)
    K = input.size(1)
    num_experts = weights.size(0)
    N = weights.size(1)

    if total_tokens == 0:
        return torch.empty(0, N, device=input.device, dtype=input.dtype)

    output = torch.empty(total_tokens, N, device=input.device, dtype=input.dtype)

    if tokens_per_expert.is_cuda:
        tokens_per_expert = tokens_per_expert.cpu()

    sorted_ids, expert_ids, offsets = _prepare_expert_mapping(
        tokens_per_expert, input.device)

    # Fixed at 4: covers tiles spanning up to 4 experts (BLOCK_M=128, >=32 tokens/expert).
    # Using a constant avoids Triton recompiling for every unique value.
    max_experts_per_tile = 4

    grid = (triton.cdiv(total_tokens, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _fused_moe_kernel[grid](
        input, weights, output,
        sorted_ids, expert_ids, offsets,
        total_tokens,
        N, K,
        input.stride(0), input.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        output.stride(0), output.stride(1),
        num_experts,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        MAX_EXPERTS_PER_TILE=max_experts_per_tile,
    )

    return output
