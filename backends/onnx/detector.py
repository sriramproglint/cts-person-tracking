"""ONNX Runtime RT-DETR person detector."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from backends.decode import NET_H, NET_W, decode_detections

if TYPE_CHECKING:
    import onnxruntime as ort

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]


def resolve_ort_device(requested: str, *, gpu_only: bool = False) -> str:
    if ort is None:
        raise RuntimeError("onnxruntime not installed")
    available = ort.get_available_providers()
    if gpu_only or requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise SystemExit("CUDAExecutionProvider required (install onnxruntime-gpu).")
        return "cuda"
    if requested == "auto":
        return "cuda" if "CUDAExecutionProvider" in available else "cpu"
    if requested == "cuda" and "CUDAExecutionProvider" not in available:
        print("WARNING: CUDA unavailable, falling back to CPU.", file=sys.stderr)
        return "cpu"
    return requested


def _probe_cuda_session(sess: ort.InferenceSession) -> None:
    images = np.zeros((1, 3, NET_H, NET_W), dtype=np.float32)
    ots = np.array([[480, 640]], dtype=np.int64)
    sess.run(None, {"images": images, "orig_target_sizes": ots})


def create_ort_session(onnx_path: Path, device: str, *, gpu_only: bool = False) -> tuple[ort.InferenceSession, str]:
    if ort is None:
        raise SystemExit("Install onnxruntime: pip install onnxruntime-gpu")
    resolved = resolve_ort_device(device, gpu_only=gpu_only)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True

    providers: list[str | tuple[str, dict]] = []
    if resolved == "cuda":
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                    "enable_cuda_graph": "0",
                },
            )
        )
        if not gpu_only:
            providers.append("CPUExecutionProvider")
    else:
        so.intra_op_num_threads = os.cpu_count() or 4
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = True
        providers.append("CPUExecutionProvider")

    try:
        sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        active = sess.get_providers()[0]
        if resolved == "cuda":
            _probe_cuda_session(sess)
    except Exception as exc:
        if resolved != "cuda" or gpu_only:
            raise
        print(f"WARNING: CUDA session failed ({exc}). Falling back to CPU.", file=sys.stderr)
        so_cpu = ort.SessionOptions()
        so_cpu.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so_cpu.intra_op_num_threads = os.cpu_count() or 4
        sess = ort.InferenceSession(str(onnx_path), sess_options=so_cpu, providers=["CPUExecutionProvider"])
        active = "CPUExecutionProvider"
    return sess, active


class OrtRtdetrDetector:
    """RT-DETR inference via ONNX Runtime."""

    def __init__(self, sess: ort.InferenceSession, conf: float, orig_mode: str, infer_max_size: int = 0) -> None:
        self.sess = sess
        self.conf = conf
        self.orig_mode = orig_mode
        self.infer_max_size = infer_max_size
        self._images = np.zeros((1, 3, NET_H, NET_W), dtype=np.float32)
        self._ots_frame = np.zeros((1, 2), dtype=np.int64)
        self._ots_network = np.array([[NET_H, NET_W]], dtype=np.int64)
        self._warmup()

    def _warmup(self) -> None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.infer(dummy)

    def _maybe_downscale(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame_bgr.shape[:2]
        if self.infer_max_size <= 0 or max(w, h) <= self.infer_max_size:
            return frame_bgr, 1.0, 1.0
        scale = self.infer_max_size / float(max(w, h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA), w / nw, h / nh

    def infer(self, frame_bgr: np.ndarray) -> list[tuple[int, float, int, int, int, int]]:
        full_h, full_w = frame_bgr.shape[:2]
        infer_frame, sx_back, sy_back = self._maybe_downscale(frame_bgr)
        blob = cv2.dnn.blobFromImage(infer_frame, 1.0 / 255.0, (NET_W, NET_H), swapRB=True, crop=False)
        np.copyto(self._images, blob)
        h, w = infer_frame.shape[:2]
        if self.orig_mode == "network":
            ots = self._ots_network
        else:
            self._ots_frame[0, 0], self._ots_frame[0, 1] = h, w
            ots = self._ots_frame

        labels, boxes, scores = self.sess.run(None, {"images": self._images, "orig_target_sizes": ots})
        infer_h, infer_w = infer_frame.shape[:2]
        return decode_detections(
            labels, boxes, scores, self.conf, self.orig_mode,
            full_h, full_w, infer_h, infer_w, sx_back, sy_back,
        )
