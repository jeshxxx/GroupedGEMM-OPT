#!/bin/bash
set -e

CUTLASS_VERSION="v3.6.0"
CUTLASS_DIR="third_party/cutlass"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -d "$CUTLASS_DIR" ]; then
    echo "CUTLASS already exists at $CUTLASS_DIR"
    echo "  To upgrade, run:  rm -rf $CUTLASS_DIR && bash fetch_cutlass.sh"
    exit 0
fi

echo "Fetching CUTLASS $CUTLASS_VERSION ..."
mkdir -p third_party
git clone --depth 1 --branch "$CUTLASS_VERSION" \
    https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"

# Apply grouped GEMM TMA descriptor fix (CUTLASS v3.6.0 bug):
# The Ptr-Array cooperative mainloop creates a template TMA descriptor with
# shape (1,1,1) for grouped GEMM, but cuTensorMapEncodeTiled rejects it
# because the box dimensions exceed the global tensor dimensions.
# Fix: use tile dimensions instead of 1, and compute valid packed strides.
MAINLOOP_FILE="$CUTLASS_DIR/include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp"
if [ -f "$MAINLOOP_FILE" ]; then
    echo "Applying grouped GEMM TMA descriptor fix ..."

    # Replace init_M/N/K = 1 with tile dimensions
    sed -i 's/int32_t init_M = 1;/int32_t init_M = size<0>(TileShape{});/' "$MAINLOOP_FILE"
    sed -i 's/int32_t init_N = 1;/int32_t init_N = size<1>(TileShape{});/' "$MAINLOOP_FILE"
    sed -i 's/int32_t init_K = 1;/int32_t init_K = size<2>(TileShape{});/' "$MAINLOOP_FILE"

    # Replace default-initialized strides with valid packed strides.
    # InternalStrideA/B are Stride<int64_t, Int<1>, int64_t> for both row/col major.
    # For A [M,K] and B [N,K], leading dimension stride = K.
    # Use cute::make_stride which is already in scope (no extra includes needed).
    sed -i 's/stride_a = InternalStrideA{};/stride_a = cute::make_stride(int64_t(init_K), cute::Int<1>{}, int64_t(0));/' "$MAINLOOP_FILE"
    sed -i 's/stride_b = InternalStrideB{};/stride_b = cute::make_stride(int64_t(init_K), cute::Int<1>{}, int64_t(0));/' "$MAINLOOP_FILE"

    echo "  Patched: $MAINLOOP_FILE"
fi

echo "Done. CUTLASS $CUTLASS_VERSION installed at $CUTLASS_DIR"
