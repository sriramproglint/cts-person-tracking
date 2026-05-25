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

# --- tracking (BoTSORT — IoU-first matching prevents swaps during close crossings) ---
TRACK_ENABLED = True
REID_WEIGHTS = "osnet_x0_25_msmt17.pt"  # fallback when FastReID disabled
TRACK_DEVICE = "auto"  # auto | cpu | cuda
# BoTSORT association thresholds
TRACK_HIGH_THRESH = 0.40   # detections above this get appearance+IoU matching (first pass)
TRACK_LOW_THRESH = 0.1     # detections between low/high get IoU-only matching (second pass)
NEW_TRACK_THRESH = 0.55    # higher bar for new IDs — prefer re-association over new ID
TRACK_BUFFER = 900         # frames to keep lost tracks alive (30 sec — retail dwell time)
MATCH_THRESH = 0.85        # Hungarian assignment cost threshold
PROXIMITY_THRESH = 0.5     # IoU gate: block appearance matching for close pairs (anti-swap)
APPEARANCE_THRESH = 0.35   # max embedding distance for appearance re-identification
CMC_METHOD = None           # disabled for static surveillance camera (avoids noise)
FRAME_RATE = 30            # used to scale track_buffer internally
FUSE_FIRST_ASSOCIATE = True  # fuse detection confidence into IoU distance (prefer high-conf)
# Short-gap ID merge: spatial re-association for brief tracker misses
TRACK_MERGE_MAX_GAP = 60   # merge across gaps up to 2 seconds (at 30fps)
TRACK_MERGE_DISTANCE_PX = 250.0  # spatial radius for re-association (dedup prevents false merges)
# Appearance-based re-identification: catches long-gap re-entries the spatial merger misses
# ONLY merges into IDs that are currently LOST (not visible on screen)
REID_MERGE_COSINE_THRESH = 0.22  # cosine distance threshold (0.22 = 78% similarity required)
REID_MERGE_MAX_AGE = 900   # keep features for up to 30 seconds after track lost
REID_MERGE_MIN_LOST = 3    # target track must be lost for at least this many frames

# --- FastReID (replaces OSNet for much stronger person re-identification) ---
FASTREID_ENABLED = True
FASTREID_WEIGHTS = ""  # empty → auto-download SBS R50-IBN MSMT17
FASTREID_INPUT_SIZE = (384, 128)  # (H, W)
# Embedding smoothing: EMA over last N frames instead of noisy single-frame features
EMBED_SMOOTH = True
EMBED_HISTORY = 20   # number of raw embeddings to keep per detection slot
EMBED_EMA_ALPHA = 0.9  # higher → smoother / slower to adapt (0.0 = raw, 1.0 = never update)
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
    fastreid_enabled: bool = FASTREID_ENABLED
    fastreid_weights: str = FASTREID_WEIGHTS
    embed_smooth: bool = EMBED_SMOOTH
    embed_history: int = EMBED_HISTORY
    embed_ema_alpha: float = EMBED_EMA_ALPHA
    track_device: str = TRACK_DEVICE
    track_high_thresh: float = TRACK_HIGH_THRESH
    track_low_thresh: float = TRACK_LOW_THRESH
    new_track_thresh: float = NEW_TRACK_THRESH
    track_buffer: int = TRACK_BUFFER
    match_thresh: float = MATCH_THRESH
    proximity_thresh: float = PROXIMITY_THRESH
    appearance_thresh: float = APPEARANCE_THRESH
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
