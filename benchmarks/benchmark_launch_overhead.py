"""
Benchmark: async D2H launch overhead for grouped GEMM.

Compares the launch overhead of:
  1. CPU tokens_per_expert  — old path (requires .cpu() before calling, which
     triggers cudaDeviceSynchronize in Python)
  2. GPU tokens_per_expert  — new path (async D2H to pinned memory with
     cudaStreamSynchronize inside C++)

Measures:
  - Host wall time:  time.perf_counter around the Python call.
    Shows CPU-side blocking — the old path blocks longer because .cpu()
    forces a full device sync.
  - CUDA event time: actual GPU-side latency including the GEMM compute.
    Shows total kernel dispatch + compute cost.
  - Launch overhead:  (host wall time - CUDA event time), approximating
    the CPU-side preparation cost.

Run:
    python benchmarks/benchmark_launch_overhead.py
"""

import time
from typing import Tuple

import torch

from grouped_gemm_opt import grouped_gemm_opt, TileConfig


def make_inputs(
    total_tokens: int,
    num_experts: int,
    K: int,
    N: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inp = torch.randn(total_tokens, K, device=device, dtype=dtype)
    weights = torch.randn(num_experts, N, K, device=device, dtype=dtype)
    alpha = torch.ones(num_experts)
    dist = torch.distributions.Dirichlet(alpha).sample()
    tpe = (dist * total_tokens).long()
    tpe[0] += total_tokens - tpe.sum().item()
    return inp, weights, tpe


def benchmark_launch(
    inp: torch.Tensor,
    weights: torch.Tensor,
    tpe_gpu: torch.Tensor,
    tile_config: TileConfig,
    warmup: int = 20,
    repeat: int = 100,
) -> dict:
    """Benchmark both GPU-tpe (new) and CPU-tpe (old) paths."""

    tpe_cpu = tpe_gpu.cpu()
    torch.cuda.synchronize()

    results = {}

    for label, tpe in [("GPU tpe (new)", tpe_gpu), ("CPU tpe (old)", tpe_cpu)]:
        # Warmup
        for _ in range(warmup):
            _ = grouped_gemm_opt(inp, weights, tpe, tile_config, sort_by_m=False)
        torch.cuda.synchronize()

        # If using GPU tpe, also queue a small kernel before each iteration
        # to simulate a realistic MoE pipeline where the GPU is busy
        # (makes cudaDeviceSynchronize cost visible).
        fill_tensor = torch.empty(1024, device=inp.device)

        host_times = []
        cuda_times = []

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

        for i in range(repeat):
            # Simulate prior GPU work in the pipeline
            fill_tensor.fill_(1.0)

            starts[i].record()
            t0 = time.perf_counter()
            _ = grouped_gemm_opt(inp, weights, tpe, tile_config, sort_by_m=False)
            t1 = time.perf_counter()
            ends[i].record()
            host_times.append((t1 - t0) * 1e3)

        torch.cuda.synchronize()
        cuda_times = sorted(
            starts[i].elapsed_time(ends[i]) for i in range(repeat)
        )
        host_times.sort()

        mid = repeat // 2
        results[label] = {
            "host_ms": host_times[mid],
            "cuda_ms": cuda_times[mid],
            "overhead_ms": host_times[mid] - cuda_times[mid],
        }

    return results


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    tile = TileConfig.Co_128x256x64

    print("=" * 100)
    print(f"Launch Overhead Benchmark: async D2H (GPU tpe) vs .cpu() (CPU tpe)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Tile: {tile.name}, dtype: bf16")
    print(f"  Measures median over 100 iterations")
    print("=" * 100)

    configs = [
        # (total_tokens, num_experts, K, N)
        (4096,   8,  4096, 14336),
        (4096,  64,  4096, 14336),
        (8192,   8,  4096, 14336),
        (8192,  64,  4096, 14336),
        (1024,   8,  2048,  1280),
        (1024,  64,  2048,  1280),
        (16384,  8,  4096, 14336),
        (16384, 64,  4096, 14336),
    ]

    header = (
        f"  {'tokens':>6}  {'E':>3}  {'K':>5}  {'N':>5}"
        f"  │  {'CPU host':>9}  {'CPU cuda':>9}  {'CPU ovhd':>9}"
        f"  │  {'GPU host':>9}  {'GPU cuda':>9}  {'GPU ovhd':>9}"
        f"  │  {'Δ ovhd':>8}  {'speedup':>8}"
    )
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for total_tokens, num_experts, K, N in configs:
        inp, weights, tpe_gpu = make_inputs(
            total_tokens, num_experts, K, N, dtype, device
        )

        res = benchmark_launch(inp, weights, tpe_gpu, tile)

        old = res["CPU tpe (old)"]
        new = res["GPU tpe (new)"]
        delta = old["overhead_ms"] - new["overhead_ms"]
        # Avoid division by zero for speedup calculation
        spd = old["host_ms"] / new["host_ms"] if new["host_ms"] > 0 else float("inf")

        print(
            f"  {total_tokens:>6}  {num_experts:>3}  {K:>5}  {N:>5}"
            f"  │  {old['host_ms']:>8.3f}ms  {old['cuda_ms']:>8.3f}ms  {old['overhead_ms']:>8.3f}ms"
            f"  │  {new['host_ms']:>8.3f}ms  {new['cuda_ms']:>8.3f}ms  {new['overhead_ms']:>8.3f}ms"
            f"  │  {delta:>7.3f}ms  {spd:>7.2f}x"
        )

        del inp, weights, tpe_gpu
        torch.cuda.empty_cache()

    print()
    print("Legend:")
    print("  CPU host / GPU host  = wall-clock time of Python grouped_gemm_opt() call")
    print("  CPU cuda / GPU cuda  = actual GPU time (CUDA events)")
    print("  CPU ovhd / GPU ovhd  = host - cuda = CPU-side launch overhead")
    print("  Δ ovhd               = overhead saved (CPU ovhd − GPU ovhd)")
    print("  speedup              = CPU host / GPU host")


if __name__ == "__main__":
    main()
