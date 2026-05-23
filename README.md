# RT-DETR person detection — Setup & Run

This repository contains a standalone RT-DETR person detection script: `rtdetr_person_detect.py`.

**Prerequisites**
- Linux or macOS (Windows should work with minor path changes)
- Python 3.8+ (3.10 recommended)
- Optional GPU: CUDA and matching `onnxruntime-gpu`

**Quick setup (recommended: virtualenv)**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If you have CUDA and want GPU acceleration:
# pip install onnxruntime-gpu
```

**Place the ONNX model**
The script expects `model.onnx` by default at `scripts/model.onnx`. You can also specify a different path with `--onnx`.

**Run (examples)**
- Run on a single image:
```bash
python3 rtdetr_person_detect.py --image /path/to/frame.jpg
```
- Run on a video file (CPU or GPU selection shown):
```bash
python3 rtdetr_person_detect.py --video clip.mp4 --device auto
# or force GPU
python3 rtdetr_person_detect.py --video clip.mp4 --device cuda
```
- Run webcam (device index 0):
```bash
python3 rtdetr_person_detect.py --camera 0
```

**Useful flags**
- `--onnx PATH` : path to ONNX model
- `--device {auto,cpu,cuda}` : runtime provider (auto prefers CUDA if available)
- `--fast` : preset for faster inference (`--stride 2`, network orig-size)
- `--stride N` : run inference every N frames (reduces CPU/GPU load)
- `--no-display` : skip cv2.imshow (useful for headless benchmarking)
- `--save PATH` : save annotated image/video to PATH

**Notes**
- For GPU acceleration install `onnxruntime-gpu` matching your CUDA driver.
- `requirements.txt` lists core deps: `opencv-python`, `numpy`, `onnxruntime`.

If you want, I can also add a small script to download a sample `model.onnx` or an example video.
