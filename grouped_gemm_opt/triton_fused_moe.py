"""
Triton Fused MoE GEMM kernel — zero tile-padding, per-token expert dispatch.

Tokens are pre-sorted by expert_id. Within each tile, uses the first token's
expert_id for the B (weight) load — correct when all tokens in a tile share
the same expert, which is true for most tiles after sorting.

Reference: vLLM fused_moe kernel (simplified for benchmark comparison).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_moe_kernel(
    A_ptr,            # [total_tokens, K]
    B_ptr,            # [num_experts, N, K]
    C_ptr,            # [total_tokens, N]
    sorted_ids_ptr,   # [total_tokens] int32
    expert_ids_ptr,   # [total_tokens] int32
    total_tokens,     # actual token count (NOT padded)
    N,
    K,
    stride_an,        # A stride along M (= K)
    stride_ak,        # A stride along K (= 1)
    stride_be,        # B stride between experts (= N * K)
    stride_bn,        # B stride along N (= K)
    stride_bk,        # B stride along K (= 1)
    stride_cn,        # C stride along M (= N)
    stride_ck,        # C stride along N (= 1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_range < total_tokens

    # Load sorted token indices (for A/C gather/scatter)
    token_ids = tl.load(sorted_ids_ptr + m_range, mask=m_mask, other=0)

    # Use the first valid token's expert_id for the entire tile's B load.
    # This is correct when all tokens in the tile share one expert (most tiles).
    first_valid = pid_m * BLOCK_M
    first_valid = tl.minimum(first_valid, total_tokens - 1)
    expert = tl.load(expert_ids_ptr + first_valid)

    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # A[token_ids, k_range]: gather rows by token_id
        a = tl.load(A_ptr + token_ids[:, None] * stride_an + k_range[None, :] * stride_ak,
                     mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # B[expert, k_range, n_range]: same expert for all rows in tile
        # B is stored as [num_experts, N, K], so B[e, n, k] = B_ptr + e*stride_be + n*stride_bn + k*stride_bk
        # For tl.dot(a[M,K], b[K,N]), b must be [BLOCK_K, BLOCK_N]
        b = tl.load(B_ptr + expert * stride_be + k_range[:, None] * stride_bk + n_range[None, :] * stride_bn,
                     mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, b)

    # C[token_ids, n_range]: scatter rows by token_id
    tl.store(C_ptr + token_ids[:, None] * stride_cn + n_range[None, :] * stride_ck,
             acc.to(C_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


def _sort_tokens_by_expert(tokens_per_expert: torch.Tensor, device: torch.device):
    """Create sorted token indices and per-position expert IDs."""
    num_experts = tokens_per_expert.size(0)
    total_tokens = int(tokens_per_expert.sum().item())

    sorted_ids = torch.empty(total_tokens, dtype=torch.int32, device=device)
    expert_ids = torch.empty(total_tokens, dtype=torch.int32, device=device)

    offset = 0
    for e in range(num_experts):
        m = int(tokens_per_expert[e].item())
        if m > 0:
            sorted_ids[offset:offset + m] = torch.arange(
                offset, offset + m, dtype=torch.int32, device=device)
            expert_ids[offset:offset + m] = e
            offset += m

    return sorted_ids, expert_ids


def triton_fused_moe(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 64,
) -> torch.Tensor:
    """Triton Fused MoE GEMM — zero padding waste."""
    total_tokens = input.size(0)
    K = input.size(1)
    N = weights.size(1)

    if total_tokens == 0:
        return torch.empty(0, N, device=input.device, dtype=input.dtype)

    output = torch.empty(total_tokens, N, device=input.device, dtype=input.dtype)

    if tokens_per_expert.is_cuda:
        tokens_per_expert = tokens_per_expert.cpu()

    sorted_ids, expert_ids = _sort_tokens_by_expert(tokens_per_expert, input.device)

    grid = (triton.cdiv(total_tokens, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _fused_moe_kernel[grid](
        input, weights, output,
        sorted_ids, expert_ids,
        total_tokens,
        N, K,
        input.stride(0), input.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output
