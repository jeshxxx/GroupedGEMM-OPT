"""
Focused benchmark: Standard gmm vs CUTLASS Auto.

Sweeps total_tokens in [74400, 595200] with num_experts=64, random distribution, BF16.
Three (hidden_dim, ffn_dim) configs: (2048,1280), (2048,2560), (1280,2048).

Tests CUTLASS with both sort_by_m=False and sort_by_m=True.
"""

import torch
import torch.nn.functional as F

try:
    from grouped_gemm_opt import grouped_gemm_opt, TileConfig
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    raise RuntimeError("grouped_gemm_opt not installed. pip install -e . --no-build-isolation")

try:
    from grouped_gemm.ops import gmm as standard_gmm
    HAS_STANDARD = True
except ImportError:
    HAS_STANDARD = False
    raise RuntimeError("grouped_gemm (standard) not installed. pip install grouped_gemm")


def random_distribution(total_tokens: int, num_experts: int) -> torch.Tensor:
    alpha = torch.ones(num_experts)
    weights = torch.distributions.Dirichlet(alpha).sample()
    tpe = (weights * total_tokens).long()
    diff = total_tokens - tpe.sum().item()
    tpe[0] += diff
    return tpe


def benchmark_fn(fn, warmup: int = 10, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


def main():
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    num_experts = 64

    token_counts = [74400, 148800, 223200, 297600, 446400, 595200]
    dim_configs = [
        (2048, 1280),
        (2048, 2560),
        (1280, 2048),
    ]

    print("=" * 130)
    print(f"Standard gmm vs CUTLASS Auto  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"num_experts={num_experts}, dtype=bf16, distribution=random")
    print("=" * 130)

    for K, N in dim_configs:
        print(f"\n{'━' * 130}")
        print(f"  K={K}, N={N}")
        print(f"{'━' * 130}")
        print(f"  {'tokens':>8}  {'avg M/E':>8}"
              f"  {'Std gmm(ms)':>12} {'Std TFLOPS':>11}"
              f"  {'CUT(ms)':>10} {'CUT TFLOPS':>11}"
              f"  {'CUT+sort(ms)':>13} {'sort TFLOPS':>12}"
              f"  {'Speedup':>8}  {'sort Speedup':>13}")
        print(f"  {'─' * 120}")

        for total_tokens in token_counts:
            avg_m = total_tokens // num_experts
            tpe = random_distribution(total_tokens, num_experts)
            total_flops = int(2 * tpe.sum().item() * N * K)

            inp = torch.randn(total_tokens, K, device=device, dtype=dtype)
            w = torch.randn(num_experts, N, K, device=device, dtype=dtype)

            # Standard gmm: expects [E, K, N]
            w_kn = w.transpose(1, 2).contiguous()
            tpe_cpu = tpe.cpu()

            std_lat = benchmark_fn(
                lambda: standard_gmm(inp, w_kn, tpe_cpu, trans_b=False))
            std_tflops = total_flops / (std_lat * 1e-3) / 1e12

            # CUTLASS Auto without sort
            cut_lat = benchmark_fn(
                lambda: grouped_gemm_opt(inp, w, tpe, TileConfig.AUTO, sort_by_m=False))
            cut_tflops = total_flops / (cut_lat * 1e-3) / 1e12

            # CUTLASS Auto with sort_by_m
            cut_sort_lat = benchmark_fn(
                lambda: grouped_gemm_opt(inp, w, tpe, TileConfig.AUTO, sort_by_m=True))
            cut_sort_tflops = total_flops / (cut_sort_lat * 1e-3) / 1e12

            speedup = std_lat / cut_lat
            sort_speedup = std_lat / cut_sort_lat
            best_speedup = max(speedup, sort_speedup)
            marker = " ◀" if best_speedup > 1.0 else ""

            print(f"  {total_tokens:>8}  {avg_m:>8}"
                  f"  {std_lat:>12.3f} {std_tflops:>11.2f}"
                  f"  {cut_lat:>10.3f} {cut_tflops:>11.2f}"
                  f"  {cut_sort_lat:>13.3f} {cut_sort_tflops:>12.2f}"
                  f"  {speedup:>7.2f}x  {sort_speedup:>12.2f}x{marker}")

            del inp, w, w_kn
            torch.cuda.empty_cache()

    print(f"\n{'=' * 130}")
    print("Speedup > 1.0x means CUTLASS Auto is faster than Standard gmm.")
    print("=" * 130)


if __name__ == "__main__":
    main()
