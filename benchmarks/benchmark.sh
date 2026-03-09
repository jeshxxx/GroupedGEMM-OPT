#!/bin/bash
set -e

echo "============================================================"
echo " Grouped GEMM Benchmark Suite"
echo " 3 configs × random distribution × BF16"
echo "============================================================"
echo ""

echo ">>> Test 1/3: K=4096, N=1280 (large K, small N)"
echo ""
python3 benchmarks/benchmark.py \
    --total-tokens=37200 --num-experts=32 \
    --hidden-dim=4096 --ffn-dim=1280 \
    --dtype=bf16 --distributions=random

echo ""
echo ">>> Test 2/3: K=4096, N=2560 (large K, medium N)"
echo ""
python3 benchmarks/benchmark.py \
    --total-tokens=37200 --num-experts=32 \
    --hidden-dim=4096 --ffn-dim=2560 \
    --dtype=bf16 --distributions=random

echo ""
echo ">>> Test 3/3: K=1280, N=4096 (small K, large N)"
echo ""
python3 benchmarks/benchmark.py \
    --total-tokens=37200 --num-experts=32 \
    --hidden-dim=1280 --ffn-dim=4096 \
    --dtype=bf16 --distributions=random

echo ""
echo ">>> Test 4/5: 64 experts, small M/expert (persistent advantage)"
echo ""
python3 benchmarks/benchmark.py \
    --total-tokens=4096 --num-experts=64 \
    --hidden-dim=4096 --ffn-dim=4096 \
    --dtype=bf16 --distributions=random

echo ""
echo ">>> Test 5/5: 128 experts, very small M/expert (persistent advantage)"
echo ""
python3 benchmarks/benchmark.py \
    --total-tokens=4096 --num-experts=128 \
    --hidden-dim=4096 --ffn-dim=4096 \
    --dtype=bf16 --distributions=random

echo ""
echo "============================================================"
echo " All benchmarks complete."
echo "============================================================"
