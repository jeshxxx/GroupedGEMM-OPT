"""
Comprehensive benchmark: CUTLASS Persistent Grouped GEMM vs baselines.

Compares:
  1. Dense GEMM (cuBLAS) — upper bound throughput reference
  2. Sequential cuBLAS — one GEMM per expert, sequential launches
  3. Batched cuBLAS — padded to max-M, batched launch
  4. CUTLASS Persistent Grouped GEMM — our optimized kernel

Metrics: latency (ms), TFLOPS, efficiency vs dense GEMM, numerical accuracy
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

try:
    from grouped_gemm_opt import grouped_gemm_opt, TileConfig
    HAS_CUTLASS_GROUPED = True
except ImportError:
    HAS_CUTLASS_GROUPED = False
    print("WARNING: grouped_gemm_opt not installed. Skipping optimized CUTLASS benchmark.")
    print("  Install with: cd /path/to/groupedgemm && pip install -e . --no-build-isolation")

try:
    from grouped_gemm_opt.triton_fused_moe import triton_fused_moe
    HAS_TRITON_FUSED = True
except ImportError:
    HAS_TRITON_FUSED = False

try:
    from grouped_gemm.ops import gmm as standard_gmm
    HAS_STANDARD_GMM = True
except ImportError:
    HAS_STANDARD_GMM = False


# ---------------------------------------------------------------------------
# Token distribution generators (simulate real MoE routing patterns)
# ---------------------------------------------------------------------------

def uniform_distribution(total_tokens: int, num_experts: int) -> torch.Tensor:
    """Each expert gets equal tokens (best case)."""
    base = total_tokens // num_experts
    remainder = total_tokens % num_experts
    tpe = torch.full((num_experts,), base, dtype=torch.int64)
    tpe[:remainder] += 1
    return tpe


def skewed_distribution(total_tokens: int, num_experts: int,
                        skew_factor: float = 3.0) -> torch.Tensor:
    """Power-law skew: a few experts get most tokens (worst case)."""
    weights = torch.pow(torch.arange(1, num_experts + 1, dtype=torch.float64),
                        -skew_factor)
    weights /= weights.sum()
    tpe = (weights * total_tokens).long()
    # Fix rounding to match total
    diff = total_tokens - tpe.sum().item()
    tpe[0] += diff
    return tpe


def random_distribution(total_tokens: int, num_experts: int) -> torch.Tensor:
    """Random Dirichlet distribution (realistic case)."""
    alpha = torch.ones(num_experts)
    weights = torch.distributions.Dirichlet(alpha).sample()
    tpe = (weights * total_tokens).long()
    diff = total_tokens - tpe.sum().item()
    tpe[0] += diff
    return tpe


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def dense_gemm(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Single dense GEMM via cuBLAS. weight is [N, K]."""
    return F.linear(input, weight)


def sequential_gemm(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Per-expert sequential cuBLAS GEMMs."""
    outputs = []
    offset = 0
    for g in range(weights.size(0)):
        m = tokens_per_expert[g].item()
        if m > 0:
            out = F.linear(input[offset:offset + m], weights[g])
            outputs.append(out)
        offset += m
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, weights.size(1),
                                                                   device=input.device,
                                                                   dtype=input.dtype)


def batched_gemm_padded(
    input: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Padded batched GEMM: pad all experts to max-M, use torch.bmm."""
    num_experts = weights.size(0)
    K = input.size(1)
    N = weights.size(1)
    max_m = tokens_per_expert.max().item()

    batched_input = torch.zeros(num_experts, max_m, K,
                                device=input.device, dtype=input.dtype)
    offset = 0
    for g in range(num_experts):
        m = tokens_per_expert[g].item()
        if m > 0:
            batched_input[g, :m] = input[offset:offset + m]
        offset += m

    # bmm: [E, max_m, K] × [E, K, N] → [E, max_m, N]
    batched_output = torch.bmm(batched_input, weights.transpose(1, 2))

    # Unpad
    outputs = []
    for g in range(num_experts):
        m = tokens_per_expert[g].item()
        if m > 0:
            outputs.append(batched_output[g, :m])
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, N,
                                                                   device=input.device,
                                                                   dtype=input.dtype)


# ---------------------------------------------------------------------------
# Accuracy verification
# ---------------------------------------------------------------------------

@dataclass
class AccuracyResult:
    name: str
    max_abs_err: float
    mean_abs_err: float
    max_rel_err: float
    cos_sim: float
    passed: bool


def verify_accuracy(
    total_tokens: int,
    num_experts: int,
    K: int,
    N: int,
    dtype: torch.dtype,
) -> List[AccuracyResult]:
    """Compare CUTLASS grouped GEMM outputs against sequential cuBLAS (ground truth)."""

    device = torch.device("cuda:0")
    results = []

    tpe = uniform_distribution(total_tokens, num_experts)
    input_tensor = torch.randn(total_tokens, K, device=device, dtype=dtype)
    expert_weights = torch.randn(num_experts, N, K, device=device, dtype=dtype)

    ref_output = sequential_gemm(input_tensor, expert_weights, tpe)
    torch.cuda.synchronize()

    # Standard grouped_gemm accuracy
    if HAS_STANDARD_GMM:
        try:
            weights_kn = expert_weights.transpose(1, 2).contiguous()
            tpe_cpu = tpe.cpu()
            test_output = standard_gmm(input_tensor, weights_kn, tpe_cpu, trans_b=False)
            torch.cuda.synchronize()

            diff = (test_output.float() - ref_output.float()).abs()
            ref_abs = ref_output.float().abs().clamp(min=1e-6)
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            max_rel = (diff / ref_abs).max().item()
            flat_test = test_output.float().flatten()
            flat_ref = ref_output.float().flatten()
            cos = torch.nn.functional.cosine_similarity(
                flat_test.unsqueeze(0), flat_ref.unsqueeze(0)).item()
            atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
            passed = torch.allclose(test_output.float(), ref_output.float(),
                                    atol=atol, rtol=1e-2)
            results.append(AccuracyResult("Standard gmm", max_abs, mean_abs,
                                          max_rel, cos, passed))
        except Exception as e:
            results.append(AccuracyResult("Standard gmm", float('inf'), float('inf'),
                                          float('inf'), 0.0, False))
            print(f"  WARNING: Standard gmm accuracy check failed: {e}")

    if HAS_TRITON_FUSED:
        try:
            test_output = triton_fused_moe(input_tensor, expert_weights, tpe)
            torch.cuda.synchronize()
            diff = (test_output.float() - ref_output.float()).abs()
            ref_abs = ref_output.float().abs().clamp(min=1e-6)
            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            max_rel = (diff / ref_abs).max().item()
            flat_test = test_output.float().flatten()
            flat_ref = ref_output.float().flatten()
            cos = torch.nn.functional.cosine_similarity(
                flat_test.unsqueeze(0), flat_ref.unsqueeze(0)).item()
            atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
            passed = torch.allclose(test_output.float(), ref_output.float(),
                                    atol=atol, rtol=1e-2)
            results.append(AccuracyResult("Triton Fused", max_abs, mean_abs,
                                          max_rel, cos, passed))
        except Exception as e:
            results.append(AccuracyResult("Triton Fused", float('inf'), float('inf'),
                                          float('inf'), 0.0, False))
            print(f"  WARNING: Triton Fused accuracy check failed: {e}")

    if not HAS_CUTLASS_GROUPED:
        return results

    configs = [
        ("Co 128x128x64",  TileConfig.Co_128x128x64),
        ("Co 128x256x64",  TileConfig.Co_128x256x64),
        ("PP 128x128x128", TileConfig.PP_128x128x128),
        ("PP 128x256x64",  TileConfig.PP_128x256x64),
    ]

    for tc_name, tc in configs:
        try:
            test_output = grouped_gemm_opt(input_tensor, expert_weights, tpe, tc,
                                           sort_by_m=False)
            torch.cuda.synchronize()

            diff = (test_output.float() - ref_output.float()).abs()
            ref_abs = ref_output.float().abs().clamp(min=1e-6)

            max_abs = diff.max().item()
            mean_abs = diff.mean().item()
            max_rel = (diff / ref_abs).max().item()

            flat_test = test_output.float().flatten()
            flat_ref = ref_output.float().flatten()
            cos = torch.nn.functional.cosine_similarity(
                flat_test.unsqueeze(0), flat_ref.unsqueeze(0)).item()

            atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
            rtol = 1e-2
            passed = torch.allclose(test_output.float(), ref_output.float(),
                                    atol=atol, rtol=rtol)

            results.append(AccuracyResult(tc_name, max_abs, mean_abs, max_rel,
                                          cos, passed))
        except Exception as e:
            results.append(AccuracyResult(tc_name, float('inf'), float('inf'),
                                          float('inf'), 0.0, False))
            print(f"  WARNING: {tc_name} failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    latency_ms: float
    tflops: float
    efficiency: float  # vs dense GEMM


def benchmark_fn(fn, warmup: int = 10, repeat: int = 50) -> float:
    """Returns median latency in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]  # median


def compute_flops(tokens_per_expert: torch.Tensor, K: int, N: int) -> int:
    """Total FLOPs for grouped GEMM: sum(2 * M_i * N * K)."""
    return int(2 * tokens_per_expert.sum().item() * N * K)


def run_benchmark(
    total_tokens: int,
    num_experts: int,
    K: int,
    N: int,
    dtype: torch.dtype,
    distribution: str = "uniform",
) -> List[BenchResult]:

    device = torch.device("cuda:0")
    results = []

    # Generate token distribution
    if distribution == "uniform":
        tpe = uniform_distribution(total_tokens, num_experts)
    elif distribution == "skewed":
        tpe = skewed_distribution(total_tokens, num_experts)
    elif distribution == "random":
        tpe = random_distribution(total_tokens, num_experts)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    total_flops = compute_flops(tpe, K, N)

    # Allocate tensors
    input_tensor = torch.randn(total_tokens, K, device=device, dtype=dtype)
    expert_weights = torch.randn(num_experts, N, K, device=device, dtype=dtype)

    # 1. Dense GEMM baseline (same total compute as single large matmul)
    dense_weight = torch.randn(N, K, device=device, dtype=dtype)
    dense_latency = benchmark_fn(lambda: dense_gemm(input_tensor, dense_weight))
    dense_tflops = total_flops / (dense_latency * 1e-3) / 1e12
    results.append(BenchResult("Dense cuBLAS", dense_latency, dense_tflops, 1.0))

    # 2. Sequential cuBLAS (one launch per expert)
    seq_latency = benchmark_fn(
        lambda: sequential_gemm(input_tensor, expert_weights, tpe))
    seq_tflops = total_flops / (seq_latency * 1e-3) / 1e12
    results.append(BenchResult("Sequential cuBLAS", seq_latency, seq_tflops,
                               dense_latency / seq_latency))

    # 3. Batched cuBLAS (padded)
    batch_latency = benchmark_fn(
        lambda: batched_gemm_padded(input_tensor, expert_weights, tpe))
    batch_tflops = total_flops / (batch_latency * 1e-3) / 1e12
    results.append(BenchResult("Batched cuBLAS (pad)", batch_latency, batch_tflops,
                               dense_latency / batch_latency))

    # 4. Standard grouped_gemm (tgale96) — CUTLASS 2.x grouped kernel
    if HAS_STANDARD_GMM:
        # standard gmm expects: a=[total_tokens, K], b=[E, K, N], batch_sizes CPU
        # with trans_b=False: computes a @ b, so b must be [E, K, N]
        weights_kn = expert_weights.transpose(1, 2).contiguous()  # [E, N, K] → [E, K, N]
        tpe_cpu = tpe.cpu()
        try:
            std_latency = benchmark_fn(
                lambda: standard_gmm(input_tensor, weights_kn, tpe_cpu, trans_b=False))
            std_tflops = total_flops / (std_latency * 1e-3) / 1e12
            results.append(BenchResult("Standard gmm", std_latency, std_tflops,
                                       dense_latency / std_latency))
        except Exception as e:
            results.append(BenchResult("Standard gmm", float('inf'), 0.0, 0.0))
            print(f"  WARNING: Standard gmm failed: {e}")

    # 5. Triton Fused MoE (zero padding)
    if HAS_TRITON_FUSED:
        triton_configs = [
            ("M128 N128 K64",  128, 128, 64),
            ("M128 N256 K64",  128, 256, 64),
        ]
        for tc_name, bm, bn, bk in triton_configs:
            try:
                triton_latency = benchmark_fn(
                    lambda bm=bm, bn=bn, bk=bk: triton_fused_moe(
                        input_tensor, expert_weights, tpe, bm, bn, bk))
                triton_tflops = total_flops / (triton_latency * 1e-3) / 1e12
                results.append(BenchResult(
                    f"Triton Fused ({tc_name})",
                    triton_latency, triton_tflops,
                    dense_latency / triton_latency))
            except Exception as e:
                results.append(BenchResult(
                    f"Triton Fused ({tc_name})",
                    float('inf'), 0.0, 0.0))
                print(f"  WARNING: Triton Fused {tc_name} failed: {e}")

    # 6. CUTLASS Persistent Grouped GEMM (opt) — each config independently
    if HAS_CUTLASS_GROUPED:
        # Determine which config Auto would select (for annotation only)
        auto_output = grouped_gemm_opt(input_tensor, expert_weights, tpe,
                                       TileConfig.AUTO, sort_by_m=False)
        auto_pick = "Co_128x256x64" if N >= 256 else "Co_128x128x64"

        configs = [
            ("Co 128x128x64",  TileConfig.Co_128x128x64),
            ("Co 128x256x64",  TileConfig.Co_128x256x64),
            ("PP 128x128x128", TileConfig.PP_128x128x128),
            ("PP 128x256x64",  TileConfig.PP_128x256x64),
        ]
        for tc_name, tc in configs:
            suffix = " ★Auto" if tc_name.replace(" ", "_") == auto_pick else ""
            try:
                cutlass_latency = benchmark_fn(
                    lambda tc=tc: grouped_gemm_opt(input_tensor, expert_weights, tpe, tc,
                                                   sort_by_m=False))
                cutlass_tflops = total_flops / (cutlass_latency * 1e-3) / 1e12
                results.append(BenchResult(
                    f"CUTLASS ({tc_name}){suffix}",
                    cutlass_latency, cutlass_tflops,
                    dense_latency / cutlass_latency))
            except Exception as e:
                results.append(BenchResult(
                    f"CUTLASS ({tc_name}){suffix}",
                    float('inf'), 0.0, 0.0))
                print(f"  WARNING: {tc_name} failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grouped GEMM Benchmark")
    parser.add_argument("--total-tokens", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=4096,
                        help="K dimension (hidden size)")
    parser.add_argument("--ffn-dim", type=int, default=14336,
                        help="N dimension (FFN intermediate size)")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--distributions", nargs="+",
                        default=["uniform", "skewed", "random"])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print("=" * 90)
    print(f"Grouped GEMM Benchmark for MoE")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Total tokens: {args.total_tokens}")
    print(f"  Experts: {args.num_experts}")
    print(f"  Dimensions: M_total={args.total_tokens}, K={args.hidden_dim}, N={args.ffn_dim}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Avg tokens/expert: {args.total_tokens // args.num_experts}")
    print("=" * 90)

    for dist in args.distributions:
        print(f"\n{'─' * 90}")
        print(f"  Distribution: {dist}")
        print(f"{'─' * 90}")
        print(f"  {'Method':<30} {'Latency(ms)':>12} {'TFLOPS':>10} {'vs Dense':>10}")
        print(f"  {'─' * 66}")

        results = run_benchmark(
            args.total_tokens, args.num_experts,
            args.hidden_dim, args.ffn_dim,
            dtype, dist)

        for r in results:
            eff_str = f"{r.efficiency:.1%}" if r.efficiency > 0 else "N/A"
            lat_str = f"{r.latency_ms:.3f}" if r.latency_ms < float('inf') else "FAIL"
            print(f"  {r.name:<30} {lat_str:>12} {r.tflops:>10.2f} {eff_str:>10}")

    # Accuracy verification
    if HAS_CUTLASS_GROUPED:
        print(f"\n\n{'=' * 90}")
        print("Accuracy: CUTLASS vs Sequential cuBLAS (ground truth)")
        print(f"  Reference: per-expert cuBLAS GEMM (FP32 accumulate)")
        print(f"{'=' * 90}")
        print(f"  {'Config':<20} {'MaxAbs':>10} {'MeanAbs':>10} {'MaxRel':>10} "
              f"{'CosSim':>10} {'Status':>8}")
        print(f"  {'─' * 72}")

        acc_results = verify_accuracy(
            min(args.total_tokens, 4096), args.num_experts,
            args.hidden_dim, args.ffn_dim, dtype)

        for r in acc_results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  {r.name:<20} {r.max_abs_err:>10.6f} {r.mean_abs_err:>10.6f} "
                  f"{r.max_rel_err:>10.6f} {r.cos_sim:>10.8f} {status:>8}")

    # Sweep over different MoE configurations
    print(f"\n\n{'=' * 90}")
    print("Sweep: varying total tokens (uniform distribution)")
    print(f"{'=' * 90}")

    token_counts = [1024, 2048, 4096, 8192, 16384]
    for total_tokens in token_counts:
        print(f"\n  Total tokens = {total_tokens} "
              f"(avg {total_tokens // args.num_experts}/expert)")
        results = run_benchmark(
            total_tokens, args.num_experts,
            args.hidden_dim, args.ffn_dim,
            dtype, "uniform")
        print(f"  {'Method':<30} {'Latency(ms)':>12} {'TFLOPS':>10} {'vs Dense':>10}")
        for r in results:
            eff_str = f"{r.efficiency:.1%}" if r.efficiency > 0 else "N/A"
            lat_str = f"{r.latency_ms:.3f}" if r.latency_ms < float('inf') else "FAIL"
            print(f"  {r.name:<30} {lat_str:>12} {r.tflops:>10.2f} {eff_str:>10}")


if __name__ == "__main__":
    main()
