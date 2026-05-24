"""GPU library paths for ONNX Runtime and PyTorch on DGX Spark (GB10 / CUDA 13)."""

from __future__ import annotations

import os
import site
from pathlib import Path


def configure_gpu_environment() -> None:
    """Expose cuDNN/CUDA libs from the venv and toolkit."""
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    candidates: list[Path] = [
        cuda_home / "lib64",
        cuda_home / "targets" / "sbsa-linux" / "lib",
    ]
    try:
        sp = Path(site.getsitepackages()[0])
        for sub in ("cudnn", "cublas", "cuda_runtime", "nvrtc"):
            candidates.append(sp / "nvidia" / sub / "lib")
    except IndexError:
        pass

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts: list[str] = []
    for p in candidates:
        if p.is_dir() and str(p) not in parts:
            parts.append(str(p))
    if existing:
        for p in existing.split(":"):
            if p and p not in parts:
                parts.append(p)
    if parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

    # Avoid invalid CUDA_VISIBLE_DEVICES (e.g. literal "cuda") breaking PyTorch/boxmot.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip().lower() in ("cuda", "gpu", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
