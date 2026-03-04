import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages

# Defer torch import so that a helpful message is shown when torch is missing
# (e.g. when pip's build-isolation creates a temp venv without torch).
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ModuleNotFoundError:
    print(
        "\n"
        "ERROR: PyTorch is not found in the current Python environment.\n"
        "\n"
        "  This usually happens because pip creates an isolated build\n"
        "  environment (PEP 517) that does not inherit your installed\n"
        "  packages.  Fix it with ONE of the following:\n"
        "\n"
        "    pip install -e . --no-build-isolation\n"
        "\n"
        "  or, if you use an older pip that ignores pyproject.toml:\n"
        "\n"
        "    python setup.py develop\n"
        "\n"
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
CUTLASS_DIR = ROOT / "third_party" / "cutlass"

if not CUTLASS_DIR.exists():
    print("CUTLASS not found. Running fetch_cutlass.sh ...")
    subprocess.check_call(["bash", str(ROOT / "fetch_cutlass.sh")], cwd=str(ROOT))

cutlass_include = str(CUTLASS_DIR / "include")
cute_include = str(CUTLASS_DIR / "tools" / "util" / "include")

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    f"-I{cutlass_include}",
    f"-I{cute_include}",
]

ext_modules = [
    CUDAExtension(
        name="grouped_gemm._C",
        sources=[
            "csrc/grouped_gemm_kernel.cu",
            "csrc/torch_ext.cpp",
        ],
        include_dirs=[
            cutlass_include,
            cute_include,
            str(ROOT / "csrc"),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags,
        },
    ),
]

setup(
    name="grouped_gemm",
    version="0.1.0",
    description="High-performance Grouped GEMM for MoE using CUTLASS 3.x Persistent Kernels",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.1.0"],
)
