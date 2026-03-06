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

# ---------------------------------------------------------------------------
# Patch: CUTLASS v3.6.0 grouped GEMM TMA descriptor bug
#
# The Ptr-Array cooperative mainloop creates a template TMA descriptor with
# shape (1,1,1) and default-zero strides for grouped GEMM. CUDA 12.5+
# validates cuTensorMapEncodeTiled inputs strictly: boxDim must be <=
# globalDim, and non-innermost strides must be 16B-aligned.
#
# Fix: use tile dimensions for shape and valid packed strides.
# We use Python to avoid sed escaping issues with C++ template syntax.
# ---------------------------------------------------------------------------
MAINLOOP_FILE="$CUTLASS_DIR/include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp"
if [ -f "$MAINLOOP_FILE" ]; then
    echo "Applying grouped GEMM TMA descriptor fix ..."
    python3 -c "
import sys
path = sys.argv[1]
with open(path, 'r') as f:
    src = f.read()

# Replace init_M/N/K = 1 with tile dimensions
src = src.replace('int32_t init_M = 1;', 'int32_t init_M = size<0>(TileShape{});')
src = src.replace('int32_t init_N = 1;', 'int32_t init_N = size<1>(TileShape{});')
src = src.replace('int32_t init_K = 1;', 'int32_t init_K = size<2>(TileShape{});')

# Replace default-zero strides with valid packed strides
src = src.replace(
    'stride_a = InternalStrideA{};',
    'stride_a = cute::make_stride(int64_t(init_K), cute::Int<1>{}, cute::Int<0>{});')
src = src.replace(
    'stride_b = InternalStrideB{};',
    'stride_b = cute::make_stride(int64_t(init_K), cute::Int<1>{}, cute::Int<0>{});')

with open(path, 'w') as f:
    f.write(src)
print(f'  Patched: {path}')
" "$MAINLOOP_FILE"
fi

echo "Done. CUTLASS $CUTLASS_VERSION installed at $CUTLASS_DIR"
