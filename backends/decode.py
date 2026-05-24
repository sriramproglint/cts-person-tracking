"""Shared RT-DETR output decoding (ONNX Runtime and TensorRT)."""

from __future__ import annotations

import numpy as np

NET_W, NET_H = 640, 640


def decode_detections(
    labels: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    conf: float,
    orig_mode: str,
    full_h: int,
    full_w: int,
    infer_h: int,
    infer_w: int,
    sx_back: float,
    sy_back: float,
) -> list[tuple[int, float, int, int, int, int]]:
    mask = scores[0] >= conf
    if not np.any(mask):
        return []

    boxes_m = boxes[0][mask]
    scores_m = scores[0][mask]
    labels_m = labels[0][mask]

    bx1 = boxes_m[:, 0].astype(np.float64)
    by1 = boxes_m[:, 1].astype(np.float64)
    bx2 = boxes_m[:, 2].astype(np.float64)
    by2 = boxes_m[:, 3].astype(np.float64)

    not_xyxy = (bx2 <= bx1) | (by2 <= by1)
    if np.any(not_xyxy):
        cx, cy, bw, bh = bx1[not_xyxy], by1[not_xyxy], bx2[not_xyxy], by2[not_xyxy]
        bx1[not_xyxy] = cx - bw / 2.0
        by1[not_xyxy] = cy - bh / 2.0
        bx2[not_xyxy] = bx1[not_xyxy] + bw
        by2[not_xyxy] = by1[not_xyxy] + bh

    max_abs = np.maximum(np.maximum(np.abs(bx1), np.abs(by1)), np.maximum(np.abs(bx2), np.abs(by2)))
    norm = max_abs <= 2.0
    if np.any(norm):
        bx1[norm] *= NET_W
        bx2[norm] *= NET_W
        by1[norm] *= NET_H
        by2[norm] *= NET_H

    valid = (bx2 > bx1) & (by2 > by1)
    if not np.any(valid):
        return []

    bx1, by1, bx2, by2 = bx1[valid], by1[valid], bx2[valid], by2[valid]
    scores_m = scores_m[valid]
    labels_m = labels_m[valid]

    if orig_mode == "frame":
        map_sx, map_sy = sx_back, sy_back
    else:
        map_sx = full_w / float(infer_w)
        map_sy = full_h / float(infer_h)

    out: list[tuple[int, float, int, int, int, int]] = []
    for i in range(len(scores_m)):
        x1 = int(round(float(bx1[i]) * map_sx))
        y1 = int(round(float(by1[i]) * map_sy))
        x2 = int(round(float(bx2[i]) * map_sx))
        y2 = int(round(float(by2[i]) * map_sy))
        x1 = max(0, min(x1, full_w - 1))
        y1 = max(0, min(y1, full_h - 1))
        x2 = max(0, min(x2, full_w))
        y2 = max(0, min(y2, full_h))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((int(labels_m[i]), float(scores_m[i]), x1, y1, x2, y2))
    return out
