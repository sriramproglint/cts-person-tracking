#!/usr/bin/env python3
"""
Standalone RT-DETR person detection (same ONNX as DeepStream PGIE).

Reads image / video / webcam, runs config/model/model.onnx, draws person boxes,
shows result with cv2.imshow (press q to quit, space to pause).

Usage:
  python3 tools/rtdetr_person_detect.py --image frame.jpg
  python3 tools/rtdetr_person_detect.py --video /path/to.mp4
  python3 tools/rtdetr_person_detect.py --video clip.mp4 --device cuda
  python3 tools/rtdetr_person_detect.py --video clip.mp4 --fast
  python3 tools/rtdetr_person_detect.py --video clip.mp4 --stride 2

Speed tips:
  - GPU: pip install onnxruntime-gpu  (CUDA must match; use --device cuda)
  - CPU: --fast or --stride 2 (infer every Nth frame, reuse boxes in between)
  - Headless benchmark: --no-display

Dependencies:
  pip install opencv-python onnxruntime
  # GPU: pip install onnxruntime-gpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Install onnxruntime: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = ROOT / "scripts" / "model.onnx"
NET_W, NET_H = 640, 640
DEFAULT_CONF = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RT-DETR person detection with bbox overlay (cv2.imshow)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Single image path")
    src.add_argument("--video", type=str, help="Video file path")
    src.add_argument("--camera", type=int, help="Webcam device index (e.g. 0)")
    p.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX), help="RT-DETR ONNX path")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Score threshold (PGIE pre-cluster-threshold)")
    p.add_argument(
        "--orig-size-mode",
        choices=("frame", "network"),
        default="frame",
        help="orig_target_sizes: frame=[H,W] of source; network=[640,640] (DeepStream style)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX Runtime provider (auto: CUDA if available, else CPU)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run inference every N frames; skipped frames reuse last detections (>=1)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Preset: --stride 2, network orig-size, lighter drawing",
    )
    p.add_argument("--no-display", action="store_true", help="Skip cv2.imshow (use with --save)")
    p.add_argument("--save", type=str, default="", help="Save annotated image or video to this path")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    p.add_argument("--frame", type=int, default=0, help="Start frame index for --video")
    p.add_argument(
        "--max-display-size",
        type=int,
        default=1280,
        help="Fit imshow preview so max(width,height) <= this (0 = native resolution)",
    )
    p.add_argument(
        "--display-scale",
        type=float,
        default=0.0,
        help="Fixed preview scale (e.g. 0.5). 0 = auto from --max-display-size",
    )
    p.add_argument(
        "--infer-max-size",
        type=int,
        default=0,
        help="Downscale frames before infer when max(w,h) exceeds this (0=off). "
        "Faster resize/draw; boxes mapped back to full resolution.",
    )
    return p.parse_args()


def resolve_device(requested: str) -> tuple[str, list[str]]:
    available = ort.get_available_providers()
    if requested == "auto":
        if "CUDAExecutionProvider" in available:
            return "cuda", available
        return "cpu", available
    if requested == "cuda" and "CUDAExecutionProvider" not in available:
        print(
            "WARNING: CUDA requested but CUDAExecutionProvider not available.\n"
            "  Install: pip install onnxruntime-gpu  (CUDA version must match your driver)\n"
            "  Falling back to CPU.",
            file=sys.stderr,
        )
        return "cpu", available
    return requested, available


def create_session(onnx_path: Path, device: str) -> tuple[ort.InferenceSession, str]:
    resolved, available = resolve_device(device)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = os.cpu_count() or 4
    so.inter_op_num_threads = 1
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True

    providers: list[str | tuple[str, dict]] = []
    if resolved == "cuda":
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                },
            )
        )
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    active = sess.get_providers()[0]
    return sess, active


class RtdetrDetector:
    """Reusable ORT session with preallocated buffers and vectorized decode."""

    def __init__(self, sess: ort.InferenceSession, conf: float, orig_mode: str, infer_max_size: int = 0):
        self.sess = sess
        self.conf = conf
        self.orig_mode = orig_mode
        self.infer_max_size = infer_max_size
        self._images = np.zeros((1, 3, NET_H, NET_W), dtype=np.float32)
        self._ots_frame = np.zeros((1, 2), dtype=np.int64)
        self._ots_network = np.array([[NET_H, NET_W]], dtype=np.int64)
        self._input_names = [i.name for i in sess.get_inputs()]
        self._warmup()

    def _warmup(self) -> None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.infer(dummy)

    def _maybe_downscale(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame_bgr.shape[:2]
        if self.infer_max_size <= 0:
            return frame_bgr, 1.0, 1.0
        longest = max(w, h)
        if longest <= self.infer_max_size:
            return frame_bgr, 1.0, 1.0
        scale = self.infer_max_size / float(longest)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        return small, w / float(nw), h / float(nh)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[int, int]:
        blob = cv2.dnn.blobFromImage(frame_bgr, scalefactor=1.0 / 255.0, size=(NET_W, NET_H), swapRB=True, crop=False)
        np.copyto(self._images, blob)
        h, w = frame_bgr.shape[:2]
        if self.orig_mode == "network":
            ots = self._ots_network
        else:
            self._ots_frame[0, 0] = h
            self._ots_frame[0, 1] = w
            ots = self._ots_frame
        return h, w

    def infer(self, frame_bgr: np.ndarray) -> list[tuple[int, float, int, int, int, int]]:
        full_h, full_w = frame_bgr.shape[:2]
        infer_frame, sx_back, sy_back = self._maybe_downscale(frame_bgr)
        self._preprocess(infer_frame)
        ots = self._ots_network if self.orig_mode == "network" else self._ots_frame
        labels, boxes, scores = self.sess.run(None, {"images": self._images, "orig_target_sizes": ots})

        mask = scores[0] >= self.conf
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

        infer_h, infer_w = infer_frame.shape[:2]
        if self.orig_mode == "frame":
            map_sx, map_sy = sx_back, sy_back
        else:
            map_sx = full_w / float(NET_W)
            map_sy = full_h / float(NET_H)
            if infer_frame is not frame_bgr:
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


def preview_scale(frame_w: int, frame_h: int, max_display_size: int, display_scale: float) -> float:
    if display_scale > 0.0:
        return display_scale
    if max_display_size <= 0:
        return 1.0
    longest = max(frame_w, frame_h)
    if longest <= max_display_size:
        return 1.0
    return max_display_size / float(longest)


def scale_for_display(vis: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return vis
    w = max(1, int(round(vis.shape[1] * scale)))
    h = max(1, int(round(vis.shape[0] * scale)))
    return cv2.resize(vis, (w, h), interpolation=cv2.INTER_AREA)


def setup_display_window(window: str, preview_w: int, preview_h: int) -> None:
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, preview_w, preview_h)


def draw_detections(
    frame_bgr: np.ndarray,
    dets: list[tuple[int, float, int, int, int, int]],
    *,
    lite: bool = False,
) -> np.ndarray:
    if not dets:
        return frame_bgr
    vis = frame_bgr if lite else frame_bgr.copy()
    for _cid, score, x1, y1, x2, y2 in dets:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if lite:
            continue
        text = f"person {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), (0, 255, 0), -1)
        cv2.putText(vis, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return vis


def show_or_save(
    window: str,
    vis: np.ndarray,
    *,
    display: bool,
    writer: cv2.VideoWriter | None,
    display_scale: float = 1.0,
    window_ready: bool = False,
) -> int:
    if writer is not None:
        writer.write(vis)
    if not display:
        return -1
    preview = scale_for_display(vis, display_scale)
    if not window_ready:
        setup_display_window(window, preview.shape[1], preview.shape[0])
    cv2.imshow(window, preview)
    return cv2.waitKey(1) & 0xFF


def apply_fast_preset(args: argparse.Namespace) -> None:
    if not args.fast:
        return
    if args.stride == 1:
        args.stride = 2
    args.orig_size_mode = "network"
    if args.infer_max_size == 0:
        args.infer_max_size = 1280


def run_image(detector: RtdetrDetector, path: Path, args: argparse.Namespace) -> None:
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Failed to read image: {path}")
    dets = detector.infer(frame)
    vis = draw_detections(frame, dets, lite=args.fast)
    print(f"{path.name}: {len(dets)} person(s)")
    for _c, sc, x1, y1, x2, y2 in dets:
        print(f"  score={sc:.3f} bbox=({x1},{y1})-({x2},{y2})")

    if args.save:
        cv2.imwrite(args.save, vis)
        print(f"Saved {args.save}")
    if not args.no_display:
        dscale = preview_scale(vis.shape[1], vis.shape[0], args.max_display_size, args.display_scale)
        preview = scale_for_display(vis, dscale)
        setup_display_window("rtdetr_person", preview.shape[1], preview.shape[0])
        cv2.imshow("rtdetr_person", preview)
        print("Press any key to close…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(
    detector: RtdetrDetector,
    source: str | int,
    args: argparse.Namespace,
    *,
    is_camera: bool,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video source: {source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not is_camera and args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    writer: cv2.VideoWriter | None = None
    save_path = Path(args.save) if args.save else None
    save_as_image = save_path is not None and save_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if args.save and not save_as_image:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"Failed to open video writer: {args.save}")

    display = not args.no_display
    display_scale = 1.0
    window_ready = False
    if display:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        display_scale = preview_scale(w, h, args.max_display_size, args.display_scale)
        if display_scale < 0.999:
            pw = max(1, int(round(w * display_scale)))
            ph = max(1, int(round(h * display_scale)))
            print(f"Preview ~{pw}x{ph} (source {w}x{h})")
        print("Controls: q=quit, space=pause/resume")

    stride = max(1, args.stride)
    last_dets: list[tuple[int, float, int, int, int, int]] = []
    paused = False
    frame_idx = 0
    n_infer = 0
    t0 = time.perf_counter()

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            if stride == 1 or (frame_idx - 1) % stride == 0:
                last_dets = detector.infer(frame)
                n_infer += 1

            vis = draw_detections(frame, last_dets, lite=args.fast)

            if frame_idx == 1 or frame_idx % 30 == 0:
                elapsed = time.perf_counter() - t0
                infer_fps = n_infer / elapsed if elapsed > 0 else 0.0
                loop_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"frame {frame_idx}: {len(last_dets)} person(s)  "
                    f"(infer {infer_fps:.1f} fps, loop {loop_fps:.1f} fps, stride={stride})"
                )

            if save_as_image and save_path is not None:
                cv2.imwrite(str(save_path), vis)
            key = show_or_save(
                "rtdetr_person",
                vis,
                display=display,
                writer=writer,
                display_scale=display_scale,
                window_ready=window_ready,
            )
            if display:
                window_ready = True
        else:
            key = cv2.waitKey(30) & 0xFF if display else 255

        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
    if args.save:
        print(f"Saved {args.save}")
    if display:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    apply_fast_preset(args)
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX not found: {onnx_path}")

    print(f"Loading {onnx_path} (requested device={args.device})…")
    sess, active_provider = create_session(onnx_path, args.device)
    print(f"ONNX Runtime provider: {active_provider}")
    if active_provider == "CPUExecutionProvider" and args.device in ("auto", "cuda"):
        print("Tip: for GPU speed, install onnxruntime-gpu matching your CUDA version.")

    detector = RtdetrDetector(sess, args.conf, args.orig_size_mode, args.infer_max_size)
    print(
        f"orig_size={args.orig_size_mode}  conf={args.conf}  stride={args.stride}"
        + (f"  infer_max_size={args.infer_max_size}" if args.infer_max_size > 0 else "")
        + ("  [fast]" if args.fast else "")
    )

    if args.image:
        run_image(detector, Path(args.image), args)
    elif args.video:
        run_video(detector, args.video, args, is_camera=False)
    else:
        run_video(detector, int(args.camera), args, is_camera=True)


if __name__ == "__main__":
    main()
