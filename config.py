"""Central configuration for person detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- paths ---
ONNX_PATH = ROOT / "model.onnx"
ENGINE_PATH = ROOT / "model.trt"
DEFAULT_VIDEO = ROOT / "output_front.mp4"

# --- detection ---
CONF_THRESHOLD = 0.30
ORIG_SIZE_MODE = "frame"  # "frame" or "network"
INFER_MAX_SIZE = 0  # downscale before infer when max(w,h) exceeds this (0=off)

# --- backend ---
BACKEND = "auto"  # auto | ort | tensorrt
ORT_DEVICE = "auto"  # auto | cpu | cuda
BUILD_TRT_IF_MISSING = False

# --- tracking (StrongSORT stable IDs) ---
TRACK_ENABLED = True
REID_WEIGHTS = "osnet_x0_25_msmt17.pt"
TRACK_DEVICE = "auto"  # auto | cpu | cuda
TRACK_MAX_AGE = 120
TRACK_N_INIT = 4
TRACK_MAX_COS_DIST = 0.28
TRACK_MAX_IOU_DIST = 0.62  # looser IoU → re-match after 1-frame miss via Kalman prediction
TRACK_NN_BUDGET = 250
TRACK_MC_LAMBDA = 0.998  # favor motion prediction during brief occlusion
TRACK_EMA_ALPHA = 0.82
# Short-gap ID merge: if StrongSORT creates new ID within this many frames, keep old ID
TRACK_MERGE_MAX_GAP = 2  # merge when gap < 2 frames (0 or 1 frame gap)
TRACK_MERGE_DISTANCE_PX = 120.0
SHOW_TRAILS = True
TRAIL_LENGTH = 40

# --- runtime ---
STRIDE = 1  # forced to 1 when tracking
GPU_ONLY = False

# --- display / I/O ---
WINDOW_NAME = "person_tracking"
MAX_DISPLAY_SIZE = 1280
DISPLAY = True
SAVE_PATH = ""
SHOW_FRAME_ID = True
AUTO_SAVE_VIDEO = True  # save annotated video when tracking a file (stem_tracked.mp4)
OUTPUT_VIDEO_SUFFIX = "_tracked"
MAX_FRAMES = 0
START_FRAME = 0

# --- tracking report ---
REPORT_ENABLED = True
REPORT_JSON = ""  # auto: tracking_report.json when empty
REPORT_HTML = ""  # auto: tracking_report.html when empty
REPORT_CSV = ""


@dataclass
class AppConfig:
    """Runtime settings (defaults from module constants; CLI overrides)."""

    source: str | int = field(default_factory=lambda: str(DEFAULT_VIDEO))
    onnx_path: Path = field(default_factory=lambda: ONNX_PATH)
    engine_path: Path = field(default_factory=lambda: ENGINE_PATH)
    backend: str = BACKEND
    ort_device: str = ORT_DEVICE
    conf: float = CONF_THRESHOLD
    orig_size_mode: str = ORIG_SIZE_MODE
    infer_max_size: int = INFER_MAX_SIZE
    stride: int = STRIDE
    track: bool = TRACK_ENABLED
    reid_weights: str = REID_WEIGHTS
    track_device: str = TRACK_DEVICE
    track_max_age: int = TRACK_MAX_AGE
    track_max_cos_dist: float = TRACK_MAX_COS_DIST
    show_trails: bool = SHOW_TRAILS
    trail_length: int = TRAIL_LENGTH
    gpu_only: bool = GPU_ONLY
    build_trt_if_missing: bool = BUILD_TRT_IF_MISSING
    display: bool = DISPLAY
    max_display_size: int = MAX_DISPLAY_SIZE
    window_name: str = WINDOW_NAME
    save: str = SAVE_PATH
    show_frame_id: bool = SHOW_FRAME_ID
    auto_save_video: bool = AUTO_SAVE_VIDEO
    max_frames: int = MAX_FRAMES
    start_frame: int = START_FRAME
    report: bool = REPORT_ENABLED
    report_json: str = REPORT_JSON
    report_csv: str = REPORT_CSV
    report_html: str = REPORT_HTML

    def resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if self.gpu_only and self.engine_path.is_file():
            return "tensorrt"
        if self.engine_path.is_file():
            return "tensorrt"
        return "ort"

    def apply_presets(self) -> None:
        if self.gpu_only:
            self.ort_device = "cuda"
            self.track_device = "cuda"
            self.track = True
            if self.engine_path.is_file():
                self.backend = "tensorrt"
        if self.track and self.stride > 1:
            self.stride = 1

    def resolve_output_video(self) -> str:
        """Default annotated output path for video files."""
        if self.save:
            return self.save
        if not self.track or not self.auto_save_video or not isinstance(self.source, str):
            return ""
        p = Path(self.source)
        if p.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return ""
        return str(p.with_name(f"{p.stem}{OUTPUT_VIDEO_SUFFIX}.mp4"))
