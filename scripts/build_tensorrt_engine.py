#!/usr/bin/env python3
"""Build model.trt from model.onnx for TensorRT inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ENGINE_PATH, ONNX_PATH  # noqa: E402
from backends.tensorrt import GB10_BUILD_HINT, build_engine  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build TensorRT engine from RT-DETR ONNX")
    p.add_argument("--onnx", type=Path, default=ONNX_PATH)
    p.add_argument("--engine", type=Path, default=ENGINE_PATH)
    p.add_argument("--fp16", action="store_true", default=True, help="Use FP16 (default)")
    p.add_argument("--fp32", action="store_true", help="Build FP32 engine")
    p.add_argument("--workspace-gb", type=int, default=4)
    args = p.parse_args()

    try:
        out = build_engine(
            args.onnx,
            args.engine,
            fp16=not args.fp32,
            workspace_gb=args.workspace_gb,
        )
        print(f"Saved {out} ({out.stat().st_size // (1024 * 1024)} MiB)")
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
