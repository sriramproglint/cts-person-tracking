#!/usr/bin/env python3
"""Run person detection and tracking on video, image, or webcam."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from config import AppConfig, DEFAULT_VIDEO, ENGINE_PATH, ONNX_PATH
from pipeline import PersonPipeline, run_on_source


def parse_args() -> AppConfig:
    p = argparse.ArgumentParser(description="RT-DETR person detection + StrongSORT tracking")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", type=str, help=f"Video file (default: {DEFAULT_VIDEO.name})")
    src.add_argument("--image", type=str, help="Single image")
    src.add_argument("--camera", type=int, help="Webcam index (e.g. 0)")

    p.add_argument("--backend", choices=("auto", "ort", "tensorrt"), default="auto")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="ORT device")
    p.add_argument("--onnx", type=Path, default=ONNX_PATH)
    p.add_argument("--engine", type=Path, default=ENGINE_PATH)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--no-track", action="store_true", help="Detection only (no StrongSORT)")
    p.add_argument("--gpu-only", action="store_true", help="CUDA only; prefer TensorRT if engine exists")
    p.add_argument("--build-trt", action="store_true", help="Build TensorRT engine from ONNX if missing")
    p.add_argument("--save", type=str, default="", help="Output video/image path (default: <video>_tracked.mp4)")
    p.add_argument("--no-auto-save", action="store_true", help="Do not auto-save annotated video")
    p.add_argument("--no-frame-id", action="store_true", help="Hide frame number overlay on video")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--frame", type=int, default=0, help="Start frame for video")
    p.add_argument("--no-report", action="store_true", help="Disable tracking quality report")
    p.add_argument("--report-json", type=str, default="", help="Save JSON report to this path")
    p.add_argument("--report-html", type=str, default="", help="Save HTML report (default: same name as JSON, .html)")
    p.add_argument("--report-csv", type=str, default="", help="Save per-frame CSV log")
    args = p.parse_args()

    cfg = AppConfig()
    if args.video:
        cfg.source = args.video
    elif args.image:
        cfg.source = args.image
    elif args.camera is not None:
        cfg.source = args.camera
    else:
        cfg.source = str(DEFAULT_VIDEO)

    cfg.backend = args.backend
    cfg.ort_device = args.device
    cfg.onnx_path = args.onnx
    cfg.engine_path = args.engine
    if args.conf is not None:
        cfg.conf = args.conf
    cfg.track = not args.no_track
    cfg.gpu_only = args.gpu_only
    cfg.build_trt_if_missing = args.build_trt
    cfg.save = args.save
    cfg.auto_save_video = not args.no_auto_save
    cfg.show_frame_id = not args.no_frame_id
    cfg.display = not args.no_display
    cfg.max_frames = args.max_frames
    cfg.start_frame = args.frame
    cfg.report = not args.no_report
    cfg.report_json = args.report_json
    cfg.report_html = args.report_html
    cfg.report_csv = args.report_csv
    return cfg


def run_image(cfg: AppConfig, pipeline: PersonPipeline) -> None:
    path = Path(cfg.source)
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Failed to read image: {path}")
    results = pipeline.process_frame(frame, frame_idx=1, run_infer=True)
    vis = pipeline.render(frame, results, frame_idx=1)
    print(f"{path.name}: {len(results)} person(s)")
    if cfg.save:
        cv2.imwrite(cfg.save, vis)
        print(f"Saved {cfg.save}")
    if cfg.display:
        cv2.imshow(cfg.window_name, vis)
        print("Press any key to close…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    cfg = parse_args()
    cfg.apply_presets()

    if cfg.track:
        try:
            import boxmot  # noqa: F401
        except ImportError:
            raise SystemExit("Tracking requires: pip install boxmot torch torchvision") from None

    backend = cfg.resolve_backend()
    if backend == "tensorrt" and cfg.track:
        try:
            import pycuda  # noqa: F401
            import tensorrt  # noqa: F401
        except ImportError:
            raise SystemExit("TensorRT requires: pip install tensorrt pycuda") from None

    pipeline = PersonPipeline(cfg)
    print(pipeline.runtime_info)
    if cfg.track:
        print(f"StrongSORT  max_age={cfg.track_max_age}  max_cos_dist={cfg.track_max_cos_dist}")
    save_out = cfg.save or cfg.resolve_output_video()
    print(f"source={cfg.source}  backend={backend}  conf={cfg.conf}  track={cfg.track}")
    if save_out:
        print(f"  annotated output: {save_out}")

    if isinstance(cfg.source, str) and Path(cfg.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        run_image(cfg, pipeline)
    else:
        run_on_source(cfg, pipeline)


if __name__ == "__main__":
    main()
