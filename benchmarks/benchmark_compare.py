"""
Focused benchmark: Standard gmm vs our implementations.

Sweeps total_tokens in [74400, 595200] with num_experts=64, random distribution, BF16.
Three (hidden_dim, ffn_dim) configs: (2048,1280), (2048,2560), (1280,2048).

IMPORTANT: Triton benchmarks run AFTER all CUTLASS/cuBLAS benchmarks to avoid
Triton JIT compilation and kernel execution polluting GPU state (L2 cache,
CUDA driver state) and skewing CUTLASS/cuBLAS measurements.
"""

import torch
import torch.nn.functional as F

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

try:
    from transformer_engine.pytorch import GroupedLinear as TEGroupedLinear
    HAS_TE = True
except ImportError:
    HAS_TE = False
    print("INFO: transformer_engine not installed. Skipping TE GroupedLinear benchmark.")


def random_distribution(total_tokens: int, num_experts: int, seed: int = 42) -> torch.Tensor:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed + total_tokens * 31 + num_experts * 7)
    alpha = torch.ones(num_experts)
    weights = torch.distributions.Dirichlet(alpha).sample()
    torch.random.set_rng_state(rng_state)
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

    token_counts = [111600, 223200, 334800, 446400, 669600, 892800]
    dim_configs = [
        (2048, 1280),
        (2048, 2560),
        (1280, 2048),
    ]

    print("=" * 145)
    print(f"Grouped GEMM Comparison  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"num_experts={num_experts}, dtype=bf16, distribution=random")
    print("=" * 145)

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

            # Dense cuBLAS: single large GEMM (upper bound reference)
            dense_w = torch.randn(N, K, device=device, dtype=dtype)
            results["Dense"] = benchmark_fn(
                lambda: F.linear(inp, dense_w))
            del dense_w

            results["Std gmm"] = benchmark_fn(
                lambda: standard_gmm(inp, w_kn, tpe_cpu, trans_b=False))

            results["CUTLASS"] = benchmark_fn(
                lambda: grouped_gemm_opt(inp, w, tpe,
                                         TileConfig.Co_128x256x64, sort_by_m=False))

            results["cuBLASLt"] = benchmark_fn(
                lambda: grouped_gemm_opt(inp, w, tpe,
                                         TileConfig.CuBLAS_Seq, sort_by_m=False))

            if HAS_TE:
                te_mod = TEGroupedLinear(num_gemms=num_experts,
                                         in_features=K, out_features=N,
                                         bias=False, device=device,
                                         params_dtype=dtype)
                m_splits = tpe.tolist()
                results["TE GrpLin"] = benchmark_fn(
                    lambda: te_mod(inp, m_splits=m_splits))
                del te_mod

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
        print(f"\n{'━' * 145}")
        print(f"  K={K}, N={N}")
        print(f"{'━' * 145}")
        cols = ["Dense", "Std gmm", "TE GrpLin", "CUTLASS", "cuBLASLt",
                "Triton128", "Triton256"]
        header = f"  {'tokens':>8}  {'M/E':>6}"
        for c in cols:
            header += f"  {c:>10}"
        header += f"  {'Best':>10}  {'vs Std':>8}  {'vs Dense':>9}"
        print(header)
        print(f"  {'─' * (len(cols) * 12 + 50)}")

        for total_tokens in token_counts:
            avg_m = total_tokens // num_experts
            results = all_results[total_tokens]

            baselines = ("Std gmm", "Dense", "TE GrpLin")
            ours = {k: v for k, v in results.items() if k not in baselines}
            best_name = min(ours, key=ours.get)
            best_lat = ours[best_name]
            std_lat = results["Std gmm"]
            dense_lat = results["Dense"]
            sp_std = std_lat / best_lat
            sp_dense = dense_lat / best_lat

            def fmt(name):
                v = results.get(name)
                return f"{v:>10.3f}" if v is not None else f"{'N/A':>10}"

            marker = " ◀" if sp_std > 1.0 else ""
            line = f"  {total_tokens:>8}  {avg_m:>6}"
            for c in cols:
                line += fmt(c)
            line += f"  {best_name:>10}  {sp_std:>7.2f}x{marker}  {sp_dense:>8.1%}"
            print(line)

        # Cleanup
        for v in test_data.values():
            del v
        test_data.clear()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 145}")
    print("All latencies in ms. 'Best' = fastest of our methods.")
    print("'vs Std' = Std gmm / Best.  'vs Dense' = Dense / Best (theoretical ceiling).")
    print("=" * 145)


if __name__ == "__main__":
    main()
