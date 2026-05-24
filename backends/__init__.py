"""Inference backends: ONNX Runtime and TensorRT."""

from backends.decode import NET_H, NET_W, decode_detections

__all__ = ["NET_H", "NET_W", "decode_detections"]
