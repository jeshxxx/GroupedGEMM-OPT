"""
Focused benchmark: Standard gmm vs our implementations.

Sweeps total_tokens in [74400, 595200] with num_experts=64, random distribution, BF16.
Three (hidden_dim, ffn_dim) configs: (2048,1280), (2048,2560), (1280,2048).

IMPORTANT: Triton benchmarks run AFTER all CUTLASS/cuBLAS benchmarks to avoid
Triton JIT compilation and kernel execution polluting GPU state (L2 cache,
CUDA driver state) and skewing CUTLASS/cuBLAS measurements.
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
        # ── Phase 1: benchmark non-Triton methods for ALL token counts ──
        # This ensures Triton JIT/execution never contaminates these measurements.
        all_results = {}
        test_data = {}

        for total_tokens in token_counts:
            tpe = random_distribution(total_tokens, num_experts)
            inp = torch.randn(total_tokens, K, device=device, dtype=dtype)
            w = torch.randn(num_experts, N, K, device=device, dtype=dtype)
            w_kn = w.transpose(1, 2).contiguous()
            tpe_cpu = tpe.cpu()

            results = {}

            results["Std gmm"] = benchmark_fn(
                lambda: standard_gmm(inp, w_kn, tpe_cpu, trans_b=False))

            results["CUTLASS"] = benchmark_fn(
                lambda: grouped_gemm_opt(inp, w, tpe,
                                         TileConfig.Co_128x256x64, sort_by_m=False))

            results["cuBLASLt"] = benchmark_fn(
                lambda: grouped_gemm_opt(inp, w, tpe,
                                         TileConfig.CuBLAS_Seq, sort_by_m=False))

            all_results[total_tokens] = results
            test_data[total_tokens] = (inp, w, w_kn, tpe, tpe_cpu)

        # ── Phase 2: benchmark Triton methods (JIT runs here, isolated) ──
        if HAS_TRITON_GG:
            torch.cuda.synchronize()
            for total_tokens in token_counts:
                inp, w, w_kn, tpe, tpe_cpu = test_data[total_tokens]

                all_results[total_tokens]["Triton128"] = benchmark_fn(
                    lambda: triton_grouped_gemm(inp, w, tpe, 128, 128, 64,
                                                num_warps=4, num_stages=3))

                all_results[total_tokens]["Triton256"] = benchmark_fn(
                    lambda: triton_grouped_gemm(inp, w, tpe, 128, 256, 64,
                                                num_warps=8, num_stages=3))

        # ── Print results ──
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
            results = all_results[total_tokens]

            ours = {k: v for k, v in results.items() if k != "Std gmm"}
            best_name = min(ours, key=ours.get)
            best_lat = ours[best_name]
            std_lat = results["Std gmm"]
            speedup = std_lat / best_lat

            def fmt(name):
                v = results.get(name)
                return f"{v:>10.3f}" if v is not None else f"{'N/A':>10}"

            marker = " ◀" if speedup > 1.0 else ""
            print(f"  {total_tokens:>8}  {avg_m:>6}"
                  f"{fmt('Std gmm')}"
                  f"{fmt('CUTLASS')}"
                  f"{fmt('cuBLASLt')}"
                  f"{fmt('Triton128')}"
                  f"{fmt('Triton256')}"
                  f"  {best_name:>10}  {speedup:>7.2f}x{marker}")

        # Cleanup
        for v in test_data.values():
            del v
        test_data.clear()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 130}")
    print("All latencies in ms. 'Best' = fastest of our methods. 'vs Std' = Std gmm / Best.")
    print("=" * 130)


if __name__ == "__main__":
    main()
