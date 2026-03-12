"""
Head-to-head benchmark: TE GroupedLinear vs Standard gmm.

Same config as benchmark_std_vs_te.py but with REVERSED phase order:
  Phase 1: TE GroupedLinear (runs first)
  Phase 2: Standard gmm (runs second)

Compare results with benchmark_std_vs_te.py to detect ordering bias.
"""

import torch

from grouped_gemm.ops import gmm as standard_gmm
from transformer_engine.pytorch import GroupedLinear as TEGroupedLinear


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

    print("=" * 110)
    print(f"TE GroupedLinear vs Standard gmm (TE first)  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"num_experts={num_experts}, dtype=bf16, distribution=random (seed=42)")
    print("=" * 110)

    for K, N in dim_configs:
        std_results = {}
        te_results = {}
        test_data = {}

        # ── Phase 1: TE GroupedLinear (runs FIRST) ──
        for total_tokens in token_counts:
            tpe = random_distribution(total_tokens, num_experts)
            inp = torch.randn(total_tokens, K, device=device, dtype=dtype)
            tpe_cpu = tpe.cpu()
            m_splits = tpe.tolist()

            te_mod = TEGroupedLinear(num_gemms=num_experts,
                                     in_features=K, out_features=N,
                                     bias=False, device=device,
                                     params_dtype=dtype)

            te_results[total_tokens] = benchmark_fn(
                lambda: te_mod(inp, m_splits=m_splits))
            del te_mod

            test_data[total_tokens] = (inp, tpe, tpe_cpu)

        # ── Phase 2: Standard gmm (runs SECOND) ──
        torch.cuda.synchronize()
        for total_tokens in token_counts:
            inp, tpe, tpe_cpu = test_data[total_tokens]
            w_kn = torch.randn(num_experts, K, N, device=device, dtype=dtype)

            std_results[total_tokens] = benchmark_fn(
                lambda: standard_gmm(inp, w_kn, tpe_cpu, trans_b=False))
            del w_kn

        # ── Print results ──
        print(f"\n{'━' * 110}")
        print(f"  K={K}, N={N}")
        print(f"{'━' * 110}")
        print(f"  {'tokens':>8}  {'avg M/E':>8}"
              f"  {'TE(ms)':>10} {'TE TFLOPS':>10}"
              f"  {'Std gmm(ms)':>12} {'Std TFLOPS':>11}"
              f"  {'TE/Std':>8}")
        print(f"  {'─' * 95}")

        for total_tokens in token_counts:
            avg_m = total_tokens // num_experts
            total_flops = int(2 * total_tokens * N * K)

            std_lat = std_results[total_tokens]
            te_lat = te_results[total_tokens]
            std_tflops = total_flops / (std_lat * 1e-3) / 1e12
            te_tflops = total_flops / (te_lat * 1e-3) / 1e12
            ratio = std_lat / te_lat

            winner = " ◀TE" if ratio > 1.0 else " ◀Std"
            print(f"  {total_tokens:>8}  {avg_m:>8}"
                  f"  {te_lat:>10.3f} {te_tflops:>10.2f}"
                  f"  {std_lat:>12.3f} {std_tflops:>11.2f}"
                  f"  {ratio:>7.2f}x{winner}")

        for v in test_data.values():
            del v
        test_data.clear()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 110}")
    print("TE/Std > 1.0x means TE GroupedLinear is faster.")
    print("Compare with benchmark_std_vs_te.py to check ordering bias.")
    print("=" * 110)


if __name__ == "__main__":
    main()
