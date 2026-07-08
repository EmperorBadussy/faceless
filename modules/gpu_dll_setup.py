"""Register CUDA / cuDNN / TensorRT DLL directories.

Import this BEFORE onnxruntime (directly or via any modules.* that imports it).
It adds the pip-installed NVIDIA library directories (nvidia-*-cu12 wheels and
tensorrt_libs) to the DLL search path so onnxruntime's CUDA and TensorRT execution
providers can resolve cudnn64_9.dll, nvinfer_10.dll, etc.

pip ships these DLLs with versioned names (nvinfer_10.dll), so the usual
ctypes.util.find_library('nvinfer') check does not find them. HAS_TENSORRT_LIBS
below reflects whether the TensorRT libs are actually present.
"""
from __future__ import annotations

import os
import glob
import importlib.util

HAS_TENSORRT_LIBS = False


def _site_packages() -> str | None:
    spec = importlib.util.find_spec("numpy")
    if spec is None or spec.origin is None:
        return None
    return os.path.dirname(os.path.dirname(spec.origin))


def _add_dir(path: str) -> None:
    if os.path.isdir(path):
        try:
            os.add_dll_directory(path)
        except (OSError, AttributeError):
            pass
        if path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]


def setup() -> None:
    global HAS_TENSORRT_LIBS
    sp = _site_packages()
    if not sp:
        return

    # nvidia-*-cu12 wheels: <sp>/nvidia/<pkg>/{bin,lib}
    for pkg in glob.glob(os.path.join(sp, "nvidia", "*")):
        for sub in ("bin", "lib"):
            _add_dir(os.path.join(pkg, sub))

    # TensorRT libs: <sp>/tensorrt_libs
    trt = os.path.join(sp, "tensorrt_libs")
    if os.path.isdir(trt) and glob.glob(os.path.join(trt, "nvinfer_*.dll")):
        _add_dir(trt)
        HAS_TENSORRT_LIBS = True


setup()
