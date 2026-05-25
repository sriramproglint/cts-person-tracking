# Person Tracking (RT-DETR + StrongSORT + FastReID)

Detect and track people in video with **stable IDs**. Uses **FastReID** (SBS R50-IBN, MSMT17) for strong re-identification and **StrongSORT** for robust multi-object tracking. Supports **ONNX Runtime** or **TensorRT** detection backends.

## Folder structure

```
CTS-Person-Tracking/
├── run.py                    # Entry point — run this
├── config.py                 # All defaults (paths, detection, tracking, display)
├── pipeline.py               # Detection + tracking loop
├── person_tracker.py         # StrongSORT + FastReID (shared by both backends)
├── fastreid_backend.py       # FastReID SBS R50-IBN — pure PyTorch (no fastreid dep)
├── id_merge.py               # Short-gap ID merge + anti-swap guard
├── tracking_report.py        # ID stability / swap / fragmentation report
├── report/
│   ├── generate_html.py      # HTML dashboard generator
│   ├── json_to_html.py       # JSON → HTML converter
│   └── serve.py              # localhost static server
├── gpu_runtime.py            # CUDA library paths (DGX Spark)
│
├── backends/
│   ├── decode.py             # Shared RT-DETR output decode
│   ├── onnx/
│   │   └── detector.py       # ONNX Runtime inference
│   └── tensorrt/
│       └── detector.py       # TensorRT engine build + inference
│
├── scripts/
│   └── build_tensorrt_engine.py
│
├── model.onnx                # RT-DETR model (place in repo root)
├── model.trt                 # TensorRT engine (build or copy)
└── output_front.mp4          # Sample video
```

## Setup

```bash
cd CTS-Person-Tracking
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place **`model.onnx`** in the repo root.

### Optional: TensorRT backend

```bash
pip install tensorrt pycuda onnx
python scripts/build_tensorrt_engine.py
```

### Optional: ONNX Runtime GPU (DGX Spark / GB10)

Standard `onnxruntime-gpu` may not include sm_121 kernels. Build a custom wheel:

```bash
bash scripts/build_onnxruntime_gpu_gb10.sh   # if available
pip install wheels/onnxruntime_gpu-*-linux_aarch64.whl
```

---

## FastReID integration

By default, tracking uses **FastReID SBS R50-IBN** trained on MSMT17 instead of OSNet. This provides dramatically better re-identification accuracy (83.9% Rank-1 vs ~52%) which directly prevents ID swaps during person crossings.

**How it works:**

- `fastreid_backend.py` is a standalone PyTorch re-implementation of FastReID's ResNet50-IBN-a + Non-local + GeM pooling + BNNeck architecture
- Pre-trained weights are auto-downloaded on first run (~94 MB from GitHub)
- The backend plugs into boxmot's StrongSORT via the `reid_model` parameter (zero changes to boxmot)
- An `AntiSwapGuard` in `id_merge.py` provides an additional spatial check to catch and reverse ID swaps

**Key guarantees:**

| Guarantee | Mechanism |
|-----------|-----------|
| One ID per customer | FastReID 2048-dim features + tight cosine threshold (0.20) |
| No ID repetition | StrongSORT monotonic ID counter + ShortGapIdMerger canonical mapping |
| Swap prevention | AntiSwapGuard detects crossed displacement vectors and reverses swaps |
| Brief occlusion survival | ShortGapIdMerger re-associates IDs lost for 0-1 frames |

**CLI flags:**

```bash
# Default: FastReID enabled
python run.py --video output_front.mp4

# Disable FastReID, use OSNet fallback
python run.py --video output_front.mp4 --no-fastreid

# Custom FastReID weights path
python run.py --video output_front.mp4 --fastreid-weights /path/to/weights.pth
```

---

## Run commands (future reference)

Activate the venv first:

```bash
source .venv/bin/activate
```

### Quick start

```bash
# Default: track people in output_front.mp4 (auto-picks backend)
python run.py

# Specific video (auto-saves output_front_tracked.mp4 with frame # + person IDs)
python run.py --video output_front.mp4

# Custom output path
python run.py --video output_front.mp4 --save my_annotated.mp4
```

### ONNX Runtime

```bash
# CPU — works end-to-end on GB10 today (~1–2 fps on 3072×3072 video)
python run.py --video output_front.mp4 --backend ort --device cpu

# GPU (requires GB10-compatible onnxruntime-gpu build)
python run.py --video output_front.mp4 --backend ort --device cuda

# Detection only (no StrongSORT IDs)
python run.py --video output_front.mp4 --backend ort --device cpu --no-track
```

### TensorRT

```bash
# Build engine from ONNX (once)
python scripts/build_tensorrt_engine.py --onnx model.onnx --engine model.trt

# Track with TensorRT + GPU ReID (when model.trt exists)
python run.py --video output_front.mp4 --backend tensorrt

# Auto-build engine if missing
python run.py --video output_front.mp4 --backend tensorrt --build-trt
```

### GPU-only mode

Uses TensorRT when `model.trt` exists, otherwise ORT CUDA + GPU ReID:

```bash
python run.py --video output_front.mp4 --gpu-only
```

### Save / headless

```bash
# Save annotated video, no preview window
python run.py --video output_front.mp4 --save tracked.mp4 --no-display

# Limit frames (testing)
python run.py --video output_front.mp4 --max-frames 100 --no-display
```

### Webcam / image

```bash
python run.py --camera 0
python run.py --image frame.jpg --save out.jpg
```

**Preview controls:** `q` = quit, `space` = pause

### Tracking quality report (HTML dashboard)

After a tracked run, reports are saved automatically:

- **`tracking_report.json`** — raw data
- **`tracking_report.html`** — visual dashboard (open in browser)

```bash
# Run video and generate reports
python run.py --video output_front.mp4 --no-display

# Custom report names (creates my_report.json + my_report.html)
python run.py --video output_front.mp4 --report-json my_report.json --no-display

# Open the HTML file directly
xdg-open tracking_report.html
# or
firefox tracking_report.html
```

**Localhost server** (optional):

```bash
python report/serve.py --port 8765
# Then open http://localhost:8765/my_report.html
```

**Rebuild HTML from existing JSON:**

```bash
python report/json_to_html.py my_report.json
```

The HTML dashboard shows:

| Section | Content |
|---------|---------|
| Summary cards | Stability score, stable IDs, swaps, fragmentation |
| Metrics legend | What each metric means |
| Chart | Detections vs tracks per frame |
| Swap table | Frame + ID pairs where crossings may have swapped IDs |
| Fragmentation table | New ID born where another ID recently vanished |
| Per-ID journey | Coverage bars, status badges (Stable / Has gaps / Short-lived) |

---

## Configuration

Edit **`config.py`** for defaults. CLI flags override at runtime.

| Setting | Default | Description |
|---------|---------|-------------|
| `ONNX_PATH` | `model.onnx` | ONNX model path |
| `ENGINE_PATH` | `model.trt` | TensorRT engine path |
| `CONF_THRESHOLD` | 0.30 | Detection score threshold |
| `TRACK_MAX_AGE` | 120 | Frames to keep lost IDs |
| `TRACK_MAX_COS_DIST` | 0.20 | ReID strictness (lower = fewer ID swaps) |
| `FASTREID_ENABLED` | True | Use FastReID instead of OSNet for ReID |
| `BACKEND` | auto | `tensorrt` if engine exists, else `ort` |

---

## GB10 / DGX Spark notes

| Backend | Status |
|---------|--------|
| ORT CPU + tracking | Works |
| ORT CUDA (custom GB10 build) | Fails at postprocessor `Cast` until patched |
| TensorRT pip 10.16 | ONNX parses; engine build may fail (sm_121 CASK) |
| TensorRT + tracking (when engine exists) | ~14–19 fps on sample video |

Recommended today on GB10:

```bash
python run.py --video output_front.mp4 --backend ort --device cpu
```

When `model.trt` is available:

```bash
python run.py --video output_front.mp4 --backend tensorrt
```
```bash
# Rerun and update reports
python run.py --video output_front.mp4 --report-json my_report.json
# → overwrites my_report.json + my_report.html

python run.py --video output_front.mp4 --report-json my_report.json --conf 0.25
# → my_report.json + my_report.html reflect the new run
```
