#!/bin/bash
set -e

CUTLASS_VERSION="v3.6.0"
CUTLASS_DIR="third_party/cutlass"

if [ -d "$CUTLASS_DIR" ]; then
    echo "CUTLASS already exists at $CUTLASS_DIR"
    echo "  To upgrade, run:  rm -rf $CUTLASS_DIR && bash fetch_cutlass.sh"
    exit 0
fi

echo "Fetching CUTLASS $CUTLASS_VERSION ..."
mkdir -p third_party
git clone --depth 1 --branch "$CUTLASS_VERSION" \
    https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
echo "Done. CUTLASS installed at $CUTLASS_DIR"
