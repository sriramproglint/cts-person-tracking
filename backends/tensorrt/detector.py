"""TensorRT engine build and RT-DETR inference on GPU."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import tensorrt as trt
except ImportError as exc:
    raise ImportError("Install TensorRT: pip install tensorrt") from exc

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except ImportError as exc:
    raise ImportError("Install PyCUDA: pip install pycuda") from exc

from backends.decode import NET_H, NET_W, decode_detections

GB10_BUILD_HINT = (
    "TensorRT engine build failed on GB10 (sm_121). The pip tensorrt package may lack "
    "CASK kernels for this GPU (createCaskHardwareInfo / no kernel image). "
    "Use ONNX Runtime with a GB10 CUDA build: python run.py --backend ort --device cuda"
)


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool = True,
    workspace_gb: int = 4,
) -> Path:
    """Compile ONNX to a serialized TensorRT engine (batch=1, 640x640)."""
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            raise RuntimeError(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, NET_H, NET_W), (1, 3, NET_H, NET_W), (1, 3, NET_H, NET_W))
    profile.set_shape("orig_target_sizes", (1, 2), (1, 2), (1, 2))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(GB10_BUILD_HINT)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return engine_path


class TensorRTRtdetrDetector:
    """RT-DETR inference via a prebuilt TensorRT engine."""

    def __init__(
        self,
        engine_path: Path,
        conf: float,
        orig_mode: str,
        infer_max_size: int = 0,
    ) -> None:
        self.conf = conf
        self.orig_mode = orig_mode
        self.infer_max_size = infer_max_size
        self._images = np.zeros((1, 3, NET_H, NET_W), dtype=np.float32)
        self._ots_frame = np.zeros((1, 2), dtype=np.int64)
        self._ots_network = np.array([[NET_H, NET_W]], dtype=np.int64)
        self._stream = cuda.Stream()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        data = Path(engine_path).read_bytes()
        self._engine = runtime.deserialize_cuda_engine(data)
        if self._engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self._io: dict[str, dict[str, Any]] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            self._io[name] = {"mode": mode, "dtype": dtype, "host": None, "device": None}

        self._warmup()

    def _set_input_shape(self) -> None:
        self._context.set_input_shape("images", (1, 3, NET_H, NET_W))
        self._context.set_input_shape("orig_target_sizes", (1, 2))

    def _bind_buffers(self) -> None:
        for name, meta in self._io.items():
            shape = tuple(self._context.get_tensor_shape(name))
            size = int(np.prod(shape))
            dtype = meta["dtype"]
            if meta["host"] is None or meta["host"].size != size:
                meta["host"] = cuda.pagelocked_empty(size, dtype)
                meta["device"] = cuda.mem_alloc(meta["host"].nbytes)
            self._context.set_tensor_address(name, int(meta["device"]))

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

        self._set_input_shape()
        self._bind_buffers()

        images_host = self._io["images"]["host"]
        images_host[:] = self._images.ravel()
        cuda.memcpy_htod_async(self._io["images"]["device"], images_host, self._stream)

        ots_host = self._io["orig_target_sizes"]["host"]
        ots_host[:] = ots.ravel()
        cuda.memcpy_htod_async(self._io["orig_target_sizes"]["device"], ots_host, self._stream)

        self._context.execute_async_v3(self._stream.handle)

        for name, meta in self._io.items():
            if meta["mode"] == trt.TensorIOMode.OUTPUT:
                cuda.memcpy_dtoh_async(meta["host"], meta["device"], self._stream)
        self._stream.synchronize()

        labels = self._io["labels"]["host"].reshape(tuple(self._context.get_tensor_shape("labels")))
        boxes = self._io["boxes"]["host"].reshape(tuple(self._context.get_tensor_shape("boxes")))
        scores = self._io["scores"]["host"].reshape(tuple(self._context.get_tensor_shape("scores")))

        infer_h, infer_w = infer_frame.shape[:2]
        return decode_detections(
            labels, boxes, scores, self.conf, self.orig_mode,
            full_h, full_w, infer_h, infer_w, sx_back, sy_back,
        )
