"""
Triton Grouped GEMM — single kernel, one expert per tile.

Each tile is pre-assigned to exactly one expert via a host-side tile map.
Advantages over CUTLASS Ptr-Array and cuBLAS sequential:
  - Single kernel launch (zero launch overhead)
  - No TMA descriptor computation per group
  - No padding waste
  - Natural L2 reuse: tiles for the same expert are consecutive,
    so expert weights stay in L2 across M-tiles.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    tile_expert_ids_ptr,
    tile_m_starts_ptr,
    offsets_ptr,
    N, K,
    stride_an, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cn, stride_ck,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert_id = tl.load(tile_expert_ids_ptr + pid_m)
    m_start = tl.load(tile_m_starts_ptr + pid_m)
    expert_end = tl.load(offsets_ptr + expert_id + 1)

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < expert_end

    n_start = pid_n * BLOCK_N
    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # C[m, n] = sum_k A[m, k] * B[expert_id, n, k]
    # B is [E, N, K] (torch.nn.Linear: out_features × in_features per expert).
    # We load B as [BLOCK_K, BLOCK_N] by transposing the (n, k) access.
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        a = tl.load(
            A_ptr + m_range[:, None] * stride_an + k_range[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        b = tl.load(
            B_ptr
            + expert_id * stride_be
            + k_range[:, None] * stride_bk
            + n_range[None, :] * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(a, b)

    tl.store(
        C_ptr + m_range[:, None] * stride_cn + n_range[None, :] * stride_ck,
        acc.to(C_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Tile map: precompute per-tile (expert_id, m_start) on CPU, upload once
# ---------------------------------------------------------------------------

_tile_map_cache: dict = {}


def _build_tile_map(tokens_per_expert: torch.Tensor, BLOCK_M: int, device: torch.device):
    """Vectorized tile map construction — no Python loops."""
    tpe = tokens_per_expert.to(dtype=torch.int64)
    num_experts = tpe.size(0)
    tiles_per_expert = (tpe + BLOCK_M - 1) // BLOCK_M
    total_m_tiles = int(tiles_per_expert.sum().item())

    if total_m_tiles == 0:
        empty = torch.zeros(0, dtype=torch.int32, device=device)
        offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        return empty, empty, offsets, 0

    offsets = torch.zeros(num_experts + 1, dtype=torch.int64)
    torch.cumsum(tpe, 0, out=offsets[1:])

    tile_expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32),
        tiles_per_expert.to(torch.int32),
    )

    tile_cumsum = torch.zeros(num_experts + 1, dtype=torch.int32)
    torch.cumsum(tiles_per_expert.to(torch.int32), 0, out=tile_cumsum[1:])

    global_tile_idx = torch.arange(total_m_tiles, dtype=torch.int64)
    local_tile_idx = global_tile_idx - tile_cumsum[tile_expert_ids].to(torch.int64)
    tile_m_starts = offsets[tile_expert_ids] + local_tile_idx * BLOCK_M

    return (
        tile_expert_ids.to(device),
        tile_m_starts.to(torch.int32).to(device),
        offsets.to(torch.int32).to(device),
        total_m_tiles,
    )


def _get_tile_map(tokens_per_expert: torch.Tensor, BLOCK_M: int, device: torch.device):
    """Cached tile map — avoids rebuilding when tokens_per_expert is unchanged."""
    key = (tuple(tokens_per_expert.tolist()), BLOCK_M)
    cached = _tile_map_cache.get(key)
    if cached is not None:
        return cached
    result = _build_tile_map(tokens_per_expert, BLOCK_M, device)
    _tile_map_cache[key] = result
    if len(_tile_map_cache) > 64:
        oldest = next(iter(_tile_map_cache))
        del _tile_map_cache[oldest]
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def triton_grouped_gemm(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Single-kernel grouped GEMM via Triton — one expert per tile.

    Args:
        input: [total_tokens, K] contiguous, tokens sorted by expert.
        weights: [num_experts, N, K] (torch.nn.Linear convention).
        tokens_per_expert: [num_experts] int64.
        BLOCK_M, BLOCK_N, BLOCK_K: tile dimensions.
        num_warps, num_stages: Triton launch parameters.

    Returns:
        output: [total_tokens, N]
    """
    total_tokens = input.size(0)
    K = input.size(1)
    N = weights.size(1)

    if total_tokens == 0:
        return torch.empty(0, N, device=input.device, dtype=input.dtype)

    if tokens_per_expert.is_cuda:
        tokens_per_expert = tokens_per_expert.cpu()
    tokens_per_expert = tokens_per_expert.to(torch.int64)

    output = torch.empty(total_tokens, N, device=input.device, dtype=input.dtype)

    tile_expert_ids, tile_m_starts, offsets, total_m_tiles = _get_tile_map(
        tokens_per_expert, BLOCK_M, input.device)

    if total_m_tiles == 0:
        return output

    num_n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (total_m_tiles, num_n_tiles)

    _grouped_gemm_kernel[grid](
        input, weights, output,
        tile_expert_ids, tile_m_starts, offsets,
        N, K,
        input.stride(0), input.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output
