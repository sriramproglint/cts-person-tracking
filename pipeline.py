"""Unified detection + BoTSORT pipeline (ORT or TensorRT)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

import gpu_runtime

gpu_runtime.configure_gpu_environment()

from config import (
    TRACK_MERGE_DISTANCE_PX,
    TRACK_MERGE_MAX_GAP,
    AppConfig,
)
from person_tracker import Detection, StrongSortPersonTracker, Track, TrackConfig
from video_overlay import draw_frame_hud, draw_tracks_bold
from tracking_report import TrackingReport


class Detector(Protocol):
    def infer(self, frame_bgr: np.ndarray) -> list[Detection]: ...


def resolve_track_device(requested: str, *, gpu_only: bool = False) -> str:
    if requested == "cpu" and not gpu_only:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
    except ImportError:
        pass
    if gpu_only:
        raise SystemExit("GPU-only mode requires CUDA for tracker ReID.")
    return "cpu"


def create_detector(cfg: AppConfig, backend: str) -> tuple[Any, str]:
    """Return (detector, runtime_info string)."""
    if backend == "tensorrt":
        from backends.tensorrt import TensorRTRtdetrDetector, build_engine

        engine = cfg.engine_path
        if not engine.is_file():
            if cfg.build_trt_if_missing:
                if not cfg.onnx_path.is_file():
                    raise SystemExit(f"ONNX not found: {cfg.onnx_path}")
                print(f"Building TensorRT engine from {cfg.onnx_path} …", file=sys.stderr)
                build_engine(cfg.onnx_path, engine)
            else:
                raise SystemExit(
                    f"TensorRT engine not found: {engine}\n"
                    "  Build: python scripts/build_tensorrt_engine.py"
                )
        det = TensorRTRtdetrDetector(engine, cfg.conf, cfg.orig_size_mode, cfg.infer_max_size)
        return det, f"TensorRT  engine={engine}"

    from backends.onnx import OrtRtdetrDetector, create_ort_session

    if not cfg.onnx_path.is_file():
        raise SystemExit(f"ONNX not found: {cfg.onnx_path}")
    sess, provider = create_ort_session(cfg.onnx_path, cfg.ort_device, gpu_only=cfg.gpu_only)
    det = OrtRtdetrDetector(sess, cfg.conf, cfg.orig_size_mode, cfg.infer_max_size)
    try:
        import onnxruntime as ort

        ver = ort.__version__
    except ImportError:
        ver = "?"
    return det, f"ONNX Runtime {ver}  provider={provider}"


def draw_detections(frame_bgr: np.ndarray, dets: list[Detection]) -> np.ndarray:
    if not dets:
        return frame_bgr
    vis = frame_bgr.copy()
    for _cid, score, x1, y1, x2, y2 in dets:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return vis


class PersonPipeline:
    """Detect persons and optionally track with stable BoTSORT IDs."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        backend = cfg.resolve_backend()
        self.backend = backend
        self.detector, self.runtime_info = create_detector(cfg, backend)
        self.report: TrackingReport | None = TrackingReport() if cfg.track and cfg.report else None
        self._trails: dict[int, list[tuple[int, int]]] = {}
        self._tracker: StrongSortPersonTracker | None = None
        self._last_dets: list[Detection] = []

        if cfg.track:
            device = resolve_track_device(cfg.track_device, gpu_only=cfg.gpu_only)
            self._tracker = StrongSortPersonTracker(
                TrackConfig(
                    reid_weights=cfg.reid_weights,
                    fastreid_enabled=cfg.fastreid_enabled,
                    fastreid_weights=cfg.fastreid_weights,
                    embed_smooth=cfg.embed_smooth,
                    embed_history=cfg.embed_history,
                    embed_ema_alpha=cfg.embed_ema_alpha,
                    device=device,
                    det_conf=cfg.conf,
                    track_high_thresh=cfg.track_high_thresh,
                    track_low_thresh=cfg.track_low_thresh,
                    new_track_thresh=cfg.new_track_thresh,
                    track_buffer=cfg.track_buffer,
                    match_thresh=cfg.match_thresh,
                    proximity_thresh=cfg.proximity_thresh,
                    appearance_thresh=cfg.appearance_thresh,
                    merge_max_gap=TRACK_MERGE_MAX_GAP,
                    merge_distance_px=TRACK_MERGE_DISTANCE_PX,
                    half=(device != "cpu"),
                )
            )

    def process_frame(self, frame_bgr: np.ndarray, *, frame_idx: int, run_infer: bool) -> list[Track] | list[Detection]:
        if run_infer:
            self._last_dets = self.detector.infer(frame_bgr)
        if self._tracker is None:
            return self._last_dets
        tracks = self._tracker.update(frame_bgr, self._last_dets, frame_idx=frame_idx)
        if self.report is not None:
            self.report.observe(frame_idx, self._last_dets, tracks)
        if self.cfg.show_trails:
            self._update_trails(tracks)
        return tracks

    def _update_trails(self, tracks: list[Track]) -> None:
        seen = {t[0] for t in tracks}
        for track_id, _cls, _sc, x1, y1, x2, y2 in tracks:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trail = self._trails.setdefault(track_id, [])
            trail.append((cx, cy))
            if len(trail) > self.cfg.trail_length:
                del trail[: len(trail) - self.cfg.trail_length]
        if self._tracker:
            for tid in list(self._trails):
                if tid not in seen and tid not in self._tracker.active_ids:
                    self._trails.pop(tid, None)

    def render(
        self,
        frame_bgr: np.ndarray,
        results: list[Track] | list[Detection],
        *,
        frame_idx: int = 0,
    ) -> np.ndarray:
        if self._tracker is None:
            vis = draw_detections(frame_bgr, results)  # type: ignore[arg-type]
        else:
            trails = self._trails if self.cfg.show_trails else None
            vis = draw_tracks_bold(frame_bgr, results, trails=trails)  # type: ignore[arg-type]
        if self.cfg.show_frame_id and frame_idx > 0:
            ids = sorted({t[0] for t in results}) if self._tracker else []
            vis = draw_frame_hud(vis, frame_idx, num_tracked=len(results), active_ids=ids)
        return vis


def _preview_scale(w: int, h: int, max_size: int) -> float:
    if max_size <= 0 or max(w, h) <= max_size:
        return 1.0
    return max_size / float(max(w, h))


def run_on_source(cfg: AppConfig, pipeline: PersonPipeline) -> None:
    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open: {cfg.source}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(cfg.source, str) and cfg.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    scale = _preview_scale(w, h, cfg.max_display_size)

    save_path = cfg.save or cfg.resolve_output_video()
    writer: cv2.VideoWriter | None = None
    if save_path and not save_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"Failed to open video writer: {save_path}")
        print(f"Saving annotated video → {save_path}")

    if cfg.display:
        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.window_name, int(w * scale), int(h * scale))
        print("Controls: q=quit, space=pause")

    stride = max(1, cfg.stride)
    frame_idx = 0
    n_infer = 0
    t0 = time.perf_counter()
    paused = False
    mode = "tracked" if pipeline._tracker else "detected"

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if cfg.max_frames > 0 and frame_idx > cfg.max_frames:
                break

            run_infer = stride == 1 or (frame_idx - 1) % stride == 0
            if run_infer:
                n_infer += 1
            results = pipeline.process_frame(frame, frame_idx=frame_idx, run_infer=run_infer)
            vis = pipeline.render(frame, results, frame_idx=frame_idx)

            if frame_idx == 1 or frame_idx % 30 == 0:
                elapsed = time.perf_counter() - t0
                loop_fps = frame_idx / elapsed if elapsed else 0.0
                infer_fps = n_infer / elapsed if elapsed else 0.0
                if pipeline._tracker:
                    ids = sorted({t[0] for t in results})  # type: ignore[union-attr]
                    print(f"frame {frame_idx}: {len(results)} {mode}  IDs={ids}  ({loop_fps:.1f} fps)")
                else:
                    print(f"frame {frame_idx}: {len(results)} {mode}  (infer {infer_fps:.1f} fps)")

            if writer:
                writer.write(vis)
            if cfg.display:
                show = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 0.999 else vis
                cv2.imshow(cfg.window_name, show)
            key = cv2.waitKey(1) & 0xFF if cfg.display else 255
        else:
            key = cv2.waitKey(30) & 0xFF if cfg.display else 255

        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video: {save_path}")
    if cfg.display:
        cv2.destroyAllWindows()
    save_reports(cfg, pipeline)


def save_reports(cfg: AppConfig, pipeline: PersonPipeline) -> None:
    """Save tracking report (JSON + HTML + optional CSV). Called after every run."""
    if not pipeline._tracker or pipeline.report is None:
        return
    print(f"\n{pipeline._tracker.merge_stats}")
    print(pipeline.report.format_text())
    json_path = Path(cfg.report_json) if cfg.report_json else Path("tracking_report.json")
    html_path = Path(cfg.report_html) if cfg.report_html else json_path.with_suffix(".html")
    pipeline.report.save_json(json_path)
    print(f"Report JSON saved: {json_path}")
    pipeline.report.save_html(html_path)
    print(f"Report HTML saved: {html_path}")
    print(f"Open in browser: file://{html_path.resolve()}")
    if cfg.report_csv:
        pipeline.report.save_csv(cfg.report_csv)
        print(f"Report CSV saved: {cfg.report_csv}")
