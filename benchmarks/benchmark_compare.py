"""
Focused benchmark: Standard gmm vs our implementations.

Sweeps total_tokens in [74400, 595200] with num_experts=64, random distribution, BF16.
Three (hidden_dim, ffn_dim) configs: (2048,1280), (2048,2560), (1280,2048).

Methods compared:
  - Standard gmm (CUTLASS 2.x grouped)
  - Ours: CUTLASS persistent, cuBLASLt sequential, Triton one-kernel
"""

import torch

try:
    from grouped_gemm_opt import grouped_gemm_opt, TileConfig
    HAS_OPT = True
except ImportError:
    HAS_OPT = False
    raise RuntimeError("grouped_gemm_opt not installed. pip install -e . --no-build-isolation")

try:
    from grouped_gemm_opt.triton_grouped_gemm import triton_grouped_gemm
    HAS_TRITON_GG = True
except ImportError:
    HAS_TRITON_GG = False
    print("WARNING: triton_grouped_gemm not available")

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
    print(f"Grouped GEMM Comparison  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"num_experts={num_experts}, dtype=bf16, distribution=random")
    print("=" * 130)

    for K, N in dim_configs:
        print(f"\n{'━' * 130}")
        print(f"  K={K}, N={N}")
        print(f"{'━' * 130}")
        header = (f"  {'tokens':>8}  {'M/E':>6}"
                  f"  {'Std gmm':>10}"
                  f"  {'CUTLASS':>10}"
                  f"  {'cuBLASLt':>10}"
                  f"  {'Triton128':>10}"
                  f"  {'Triton256':>10}"
                  f"  {'Best':>10}  {'vs Std':>8}")
        print(header)
        print(f"  {'─' * 120}")

        for total_tokens in token_counts:
            avg_m = total_tokens // num_experts
            tpe = random_distribution(total_tokens, num_experts)
            total_flops = int(2 * tpe.sum().item() * N * K)

            inp = torch.randn(total_tokens, K, device=device, dtype=dtype)
            w = torch.randn(num_experts, N, K, device=device, dtype=dtype)
            w_kn = w.transpose(1, 2).contiguous()
            tpe_cpu = tpe.cpu()

            results = {}

            # Standard gmm
            std_lat = benchmark_fn(
                lambda: standard_gmm(inp, w_kn, tpe_cpu, trans_b=False))
            results["Std gmm"] = std_lat

            # CUTLASS persistent (Co 128x256x64)
            cut_lat = benchmark_fn(
                lambda: grouped_gemm_opt(inp, w, tpe,
                                         TileConfig.Co_128x256x64, sort_by_m=False))
            results["CUTLASS"] = cut_lat

            # cuBLASLt sequential
            cublas_lat = benchmark_fn(
                lambda: grouped_gemm_opt(inp, w, tpe,
                                         TileConfig.CuBLAS_Seq, sort_by_m=False))
            results["cuBLASLt"] = cublas_lat

            # Triton one-kernel 128x128x64
            if HAS_TRITON_GG:
                t128_lat = benchmark_fn(
                    lambda: triton_grouped_gemm(inp, w, tpe, 128, 128, 64,
                                                num_warps=4, num_stages=3))
                results["Triton128"] = t128_lat

                # Triton one-kernel 128x256x64
                t256_lat = benchmark_fn(
                    lambda: triton_grouped_gemm(inp, w, tpe, 128, 256, 64,
                                                num_warps=8, num_stages=3))
                results["Triton256"] = t256_lat

            # Find best non-standard method
            ours = {k: v for k, v in results.items() if k != "Std gmm"}
            best_name = min(ours, key=ours.get)
            best_lat = ours[best_name]
            speedup = std_lat / best_lat

            def fmt(lat):
                return f"{lat:>10.3f}"

            line = f"  {total_tokens:>8}  {avg_m:>6}"
            line += fmt(std_lat)
            line += fmt(results.get("CUTLASS", float('inf')))
            line += fmt(results.get("cuBLASLt", float('inf')))
            line += fmt(results.get("Triton128", float('inf')))
            line += fmt(results.get("Triton256", float('inf')))

            marker = " ◀" if speedup > 1.0 else ""
            line += f"  {best_name:>10}  {speedup:>7.2f}x{marker}"
            print(line)

            del inp, w, w_kn
            torch.cuda.empty_cache()

    print(f"\n{'=' * 130}")
    print("All latencies in ms. 'Best' = fastest of our methods. 'vs Std' = Std gmm / Best.")
    print("=" * 130)


if __name__ == "__main__":
    main()
