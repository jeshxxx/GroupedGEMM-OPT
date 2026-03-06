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
# In to_underlying_arguments(), grouped GEMM creates a template TMA descriptor
# with shape (1,1,1) and zero strides. cuTensorMapEncodeTiled (CUDA 12.5+)
# rejects this because boxDim > globalDim.
#
# Fix: use the FIRST group's actual dimensions and strides from the host-side
# problem shapes. This creates a fully valid TMA descriptor that passes all
# CUDA driver validation. The kernel replaces it per-group at runtime anyway.
# ---------------------------------------------------------------------------
MAINLOOP_FILE="$CUTLASS_DIR/include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_ss_warpspecialized.hpp"
if [ -f "$MAINLOOP_FILE" ]; then
    echo "Applying grouped GEMM TMA descriptor fix ..."
    python3 -c "
import sys

path = sys.argv[1]
with open(path, 'r') as f:
    src = f.read()

# The original code for grouped GEMM:
#   int32_t init_M = 1;
#   int32_t init_N = 1;
#   int32_t init_K = 1;
#   ...
#   if constexpr (IsGroupedGemmKernel) {
#       stride_a = InternalStrideA{};
#       stride_b = InternalStrideB{};
#   }
#
# Replace the entire grouped GEMM branch to use the first group's
# real dimensions and compute valid strides from them.

old_block = '''      stride_a = InternalStrideA{};
      stride_b = InternalStrideB{};'''

new_block = '''      if (problem_shapes.is_host_problem_shape_available()) {
        auto host_ps = problem_shapes.get_host_problem_shape(0);
        init_M = cute::get<0>(host_ps);
        init_N = cute::get<1>(host_ps);
        init_K = cute::get<2>(host_ps);
      } else {
        init_M = cute::size<0>(TileShape{});
        init_N = cute::size<1>(TileShape{});
        init_K = cute::size<2>(TileShape{});
      }
      stride_a = cute::make_stride(int64_t(init_K), cute::Int<1>{}, cute::Int<0>{});
      stride_b = cute::make_stride(int64_t(init_K), cute::Int<1>{}, cute::Int<0>{});'''

if old_block in src:
    src = src.replace(old_block, new_block)
    print(f'  Patched: {path}')
else:
    print(f'  WARNING: Could not find patch target in {path}')
    print(f'  File may already be patched or has a different format.')

with open(path, 'w') as f:
    f.write(src)
" "$MAINLOOP_FILE"
fi

echo "Done. CUTLASS $CUTLASS_VERSION installed at $CUTLASS_DIR"
