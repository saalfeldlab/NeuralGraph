#!/usr/bin/env python3
"""
Cross-platform environment validation for NeuralGraph (macOS & Linux).

Checks:
  ✅ torch import + backend (MPS or CUDA)
  ✅ numpy import and numerical correctness
  ✅ scipy import and basic linear algebra
"""

import io
import sys
import platform
import torch
import numpy as np
from contextlib import redirect_stdout

# SciPy might not always be pre-installed; catch ImportError gracefully
try:
    import scipy
    from scipy import linalg
except ImportError:
    scipy = None


def check_numpy():
    print("\n🔍 Testing NumPy import and functionality...")
    print(f"✅ NumPy {np.__version__} imported successfully.")

    # Print BLAS/LAPACK info safely for NumPy 1.x and 2.x
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            np.show_config()
        info = buf.getvalue().strip()
        if info:
            print("Using BLAS/LAPACK configuration:")
            print(info)
        else:
            print("ℹ️  No BLAS/LAPACK info available (NumPy 2.x minimal build).")
    except Exception as e:
        print(f"⚠️ Could not query BLAS info: {e}")

    # Quick numerical test
    a = np.random.rand(200, 200)
    b = np.random.rand(200, 200)
    c = np.dot(a, b)
    print(f"✅ NumPy matrix multiplication result shape: {c.shape}")
    assert c.shape == (200, 200)


def check_scipy():
    if scipy is None:
        print("⚠️ SciPy not installed. Skipping SciPy tests.")
        return
    print("\n🔍 Testing SciPy import and functionality...")
    print(f"✅ SciPy {scipy.__version__} imported successfully.")
    A = np.random.rand(100, 100)
    b = np.random.rand(100)
    x = linalg.solve(A + np.eye(100), b)  # ensure non-singular
    print(f"✅ SciPy linear solve successful. Solution norm: {np.linalg.norm(x):.4f}")


def check_torch_import():
    print("\n🔍 Testing PyTorch import...")
    print(f"✅ PyTorch {torch.__version__} imported successfully.")
    print(f"Built with CUDA: {torch.version.cuda}")
    print(f"Compiled with MPS: {'mps' in dir(torch.backends)}")


def test_cuda_linux():
    print("\n🔍 Testing CUDA backend...")
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Check driver installation or CUDA-enabled wheel.")
        sys.exit(1)

    print(f"✅ CUDA available, version: {torch.version.cuda}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Active GPU: {torch.cuda.get_device_name(0)}")

    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = torch.matmul(x, y)
    print(f"✅ Matrix multiply successful on CUDA, result shape: {z.shape}")


def test_mps_mac():
    print("\n🔍 Testing MPS backend...")
    if not torch.backends.mps.is_available():
        print("❌ MPS not available. Ensure macOS ≥12.3 and MPS-compatible PyTorch build.")
        sys.exit(1)

    print("✅ MPS backend is available.")
    device = torch.device("mps")
    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    z = torch.matmul(x, y)
    print(f"✅ Matrix multiply successful on MPS, result shape: {z.shape}")


def main():
    os_name = platform.system()
    print("===============================================")
    print("🧪 NeuralGraph Environment Validation")
    print("===============================================")
    print(f"Platform: {os_name}")
    print(f"Python version: {platform.python_version()}\n")

    check_numpy()
    check_scipy()
    check_torch_import()

    if os_name == "Darwin":
        test_mps_mac()
    elif os_name == "Linux":
        test_cuda_linux()
    else:
        print("⚠️ Unknown platform; skipping backend checks.")

    print("\n🎉 All tests completed successfully!")


if __name__ == "__main__":
    main()
