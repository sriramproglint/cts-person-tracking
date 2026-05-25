"""
Standalone FastReID (SBS R50-IBN) backend for boxmot BoTSORT / StrongSORT.

Architecture: ResNet50 + IBN-a + Non-local + GeM pooling + BNNeck
Pre-trained on MSMT17 (Rank@1 83.9%, mAP 60.6% — much stronger than OSNet_x0_25).

Implements ``get_features()`` / ``warmup()`` expected by boxmot trackers.
No fastreid library dependency — pure PyTorch re-implementation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FASTREID_WEIGHTS_URL = (
    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50-ibn.pth"
)
FASTREID_WEIGHTS_NAME = "msmt_sbs_R50-ibn.pth"
INPUT_SIZE = (384, 128)  # (H, W) matching SBS config
FEAT_DIM = 2048
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]


# ── layers ────────────────────────────────────────────────────────────────

class _IBN(nn.Module):
    """Instance-Batch Normalization (IBN-a variant)."""

    def __init__(self, planes: int) -> None:
        super().__init__()
        half = planes // 2
        self.half = half
        self.IN = nn.InstanceNorm2d(half, affine=True)
        self.BN = nn.BatchNorm2d(planes - half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = torch.split(x, self.half, 1)
        return torch.cat(
            [self.IN(split[0].contiguous()), self.BN(split[1].contiguous())], 1
        )


class _NonLocal(nn.Module):
    """Lightweight non-local self-attention (matches FastReID checkpoint)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # FastReID uses inter_channels = reduc_ratio // reduc_ratio = 1
        ic = 1
        self.inter_channels = ic
        self.g = nn.Conv2d(in_channels, ic, 1)
        self.W = nn.Sequential(
            nn.Conv2d(ic, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.theta = nn.Conv2d(in_channels, ic, 1)
        self.phi = nn.Conv2d(in_channels, ic, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        ic = self.inter_channels
        g = self.g(x).view(B, ic, -1).permute(0, 2, 1)
        theta = self.theta(x).view(B, ic, -1).permute(0, 2, 1)
        phi = self.phi(x).view(B, ic, -1)
        f = torch.matmul(theta, phi)
        y = torch.matmul(f / f.size(-1), g)
        y = y.permute(0, 2, 1).contiguous().view(B, ic, *x.shape[2:])
        return self.W(y) + x


class _GeMP(nn.Module):
    """Generalized Mean Pooling with trainable power."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        ).pow(1.0 / self.p)


# ── ResNet50-IBN-a bottleneck ─────────────────────────────────────────────

class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        with_ibn: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1: nn.Module = _IBN(planes) if with_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


# ── full backbone ─────────────────────────────────────────────────────────

class _ResNetIBNa(nn.Module):
    """ResNet50 with IBN-a (layers 1-3) and Non-local blocks (layers 2-3).

    Mirrors the exact architecture of FastReID ``build_resnet_backbone`` with
    ``depth=50x, with_ibn=True, with_nl=True, last_stride=1, bn_norm=BN``.
    """

    def __init__(self) -> None:
        super().__init__()
        layers = [3, 4, 6, 3]
        nl_counts = [0, 2, 3, 0]
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(64, layers[0], 1, with_ibn=True)
        self.layer2 = self._make_layer(128, layers[1], 2, with_ibn=True)
        self.layer3 = self._make_layer(256, layers[2], 2, with_ibn=True)
        self.layer4 = self._make_layer(512, layers[3], 1, with_ibn=False)

        # non-local blocks injected after specific residual blocks
        self.NL_1_idx: list[int] = []
        self.NL_2 = nn.ModuleList([_NonLocal(512) for _ in range(nl_counts[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(nl_counts[1])])
        self.NL_3 = nn.ModuleList([_NonLocal(1024) for _ in range(nl_counts[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(nl_counts[2])])
        self.NL_4_idx: list[int] = []

        self._init_weights()

    # ── helpers ──

    def _make_layer(
        self, planes: int, blocks: int, stride: int, with_ibn: bool
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * _Bottleneck.expansion,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * _Bottleneck.expansion),
            )
        layer_list: list[nn.Module] = [
            _Bottleneck(self.inplanes, planes, stride, downsample, with_ibn)
        ]
        self.inplanes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layer_list.append(
                _Bottleneck(self.inplanes, planes, with_ibn=with_ibn)
            )
        return nn.Sequential(*layer_list)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _run_layer_with_nl(
        self,
        layer: nn.Sequential,
        nl_list: nn.ModuleList,
        nl_idx: list[int],
        x: torch.Tensor,
    ) -> torch.Tensor:
        counter = 0
        sentinel = nl_idx if nl_idx else [-1]
        for i in range(len(layer)):
            x = layer[i](x)
            if counter < len(sentinel) and i == sentinel[counter]:
                x = nl_list[counter](x)
                counter += 1
        return x

    # ── forward ──

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for blk in self.layer1:
            x = blk(x)
        x = self._run_layer_with_nl(self.layer2, self.NL_2, self.NL_2_idx, x)
        x = self._run_layer_with_nl(self.layer3, self.NL_3, self.NL_3_idx, x)
        for blk in self.layer4:
            x = blk(x)
        return x


# ── full model (backbone + head) ─────────────────────────────────────────

class _FastReIDModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ResNetIBNa()
        self.pool_layer = _GeMP(3.0)
        # BNNeck (bias frozen during training, we keep it frozen)
        self.bottleneck = nn.Sequential(nn.BatchNorm2d(FEAT_DIM))
        nn.init.constant_(self.bottleneck[0].weight, 1.0)
        nn.init.constant_(self.bottleneck[0].bias, 0.0)
        self.bottleneck[0].bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = self.pool_layer(feat)  # (B, 2048, 1, 1)
        neck = self.bottleneck(pooled)
        return neck[..., 0, 0]  # (B, 2048)


# ── checkpoint remapping ──────────────────────────────────────────────────

def _remap_checkpoint(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap FastReID checkpoint keys to our ``_FastReIDModel`` structure.

    FastReID uses ``heads.pool_layer.p`` and ``heads.bnneck.*`` (or
    ``heads.bottleneck.0.*``) while our model uses ``pool_layer.p``
    and ``bottleneck.0.*``.  Backbone keys already match.  Classifier
    weights (``heads.weight``, ``heads.classifier.*``) are skipped.
    """
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("heads.pool_layer."):
            nk = nk.replace("heads.pool_layer.", "pool_layer.", 1)
        elif nk.startswith("heads.bnneck."):
            nk = nk.replace("heads.bnneck.", "bottleneck.0.", 1)
        elif nk.startswith("heads.bottleneck."):
            nk = nk.replace("heads.bottleneck.", "bottleneck.", 1)
        elif nk.startswith("heads.weight") or nk.startswith("heads.classifier"):
            continue
        remapped[nk] = v
    return remapped


def _download_weights(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("FastReID weights cached: %s", dest)
        return
    print(f"Downloading FastReID SBS R50-IBN (MSMT17) → {dest}")
    try:
        import gdown

        gdown.download(FASTREID_WEIGHTS_URL, str(dest), quiet=False)
    except Exception:
        import urllib.request

        urllib.request.urlretrieve(FASTREID_WEIGHTS_URL, str(dest))
    print(f"FastReID weights saved: {dest}")


# ── public backend (boxmot-compatible) ────────────────────────────────────

class FastReIDBackend:
    """Drop-in ReID backend for boxmot BoTSORT / StrongSORT.

    Provides ``get_features(xyxys, img)`` and ``warmup()`` matching the
    interface that boxmot trackers call on ``self.model``.
    """

    def __init__(
        self,
        device: torch.device,
        half: bool = False,
        weights: Path | str | None = None,
    ) -> None:
        self.device = device
        self.half = half
        self.input_shape = INPUT_SIZE  # (H, W) = (384, 128)

        dtype = torch.float16 if half else torch.float32
        self.mean_array = torch.tensor(PIXEL_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std_array = torch.tensor(PIXEL_STD, device=device, dtype=dtype).view(1, 3, 1, 1)

        # resolve / download weights
        if weights is None:
            from boxmot.utils import WEIGHTS as _W

            weights = _W / FASTREID_WEIGHTS_NAME
        weights = Path(weights)
        _download_weights(weights)

        # build model & load
        self.model = _FastReIDModel()
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        remapped = _remap_checkpoint(state)
        info = self.model.load_state_dict(remapped, strict=False)
        if info.missing_keys:
            print(f"FATAL: FastReID missing keys: {info.missing_keys}")
            raise RuntimeError(f"FastReID weight loading incomplete: {info.missing_keys}")
        loaded = len(remapped) - len(info.unexpected_keys)
        logger.info("FastReID: loaded %d/%d params (all keys matched)", loaded, len(remapped))

        self.model.to(device).eval()
        if half:
            self.model.half()

    # ── cropping / preprocessing ──

    def get_crops(self, xyxys: np.ndarray, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        xyxys = np.asarray(xyxys, dtype=np.float32)
        if xyxys.ndim == 1:
            xyxys = xyxys.reshape(1, -1)
        n = len(xyxys)
        dtype = torch.float16 if self.half else torch.float32
        crops = torch.empty((n, 3, *self.input_shape), dtype=dtype, device=self.device)

        for i, box in enumerate(xyxys):
            x1, y1, x2, y2 = box[:4].astype(int)
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            if cx2 > cx1 and cy2 > cy1:
                crop = img[cy1:cy2, cx1:cx2]
            else:
                crop = np.zeros((*self.input_shape, 3), dtype=np.uint8)
            crop = cv2.resize(crop, (self.input_shape[1], self.input_shape[0]))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(crop).to(self.device, dtype=dtype)
            crops[i] = t.permute(2, 0, 1)

        crops = (crops / 255.0 - self.mean_array) / self.std_array
        return crops

    def inference_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.half and x.dtype != torch.float16:
            x = x.half()
        return x

    def inference_postprocess(self, features: torch.Tensor) -> np.ndarray:
        return features.cpu().float().numpy()

    @torch.no_grad()
    def get_features(self, xyxys: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Extract L2-normalised appearance features for each box."""
        if xyxys.size == 0:
            return np.empty((0, FEAT_DIM), dtype=np.float32)
        crops = self.get_crops(xyxys, img)
        crops = self.inference_preprocess(crops)
        features = self.model(crops)
        features = self.inference_postprocess(features)
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return features / norms

    def warmup(self) -> None:
        if self.device.type != "cpu":
            dtype = torch.float16 if self.half else torch.float32
            dummy = torch.randn(2, 3, *self.input_shape, dtype=dtype, device=self.device)
            with torch.no_grad():
                self.model(dummy)

    def forward(self, im_batch: torch.Tensor) -> torch.Tensor:
        return self.model(im_batch)


# ── embedding smoother (wraps any ReID backend) ──────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


class SmoothedReIDBackend:
    """Wraps any ReID backend and returns EMA-smoothed embeddings.

    Instead of passing a noisy single-frame embedding to the tracker, this
    wrapper maintains a sliding window of recent embeddings per spatial
    position (associated via IoU across frames) and returns the smoothed
    version.  This makes the appearance signal much more stable during
    brief pose changes, partial occlusions, and motion blur.

    Parameters
    ----------
    backend : object
        Underlying ReID backend (e.g. ``FastReIDBackend``) with a
        ``get_features(xyxys, img)`` method.
    history_size : int
        Maximum number of raw embeddings stored per slot (default 20).
    ema_alpha : float
        EMA decay factor.  Higher → the smoothed embedding reacts more
        slowly to new observations (default 0.9).
    iou_thresh : float
        Minimum IoU to associate a current detection with an existing
        slot from the previous frame (default 0.25).
    max_stale : int
        Remove slots not matched for this many consecutive frames.
    """

    def __init__(
        self,
        backend: object,
        *,
        history_size: int = 20,
        ema_alpha: float = 0.9,
        iou_thresh: float = 0.25,
        max_stale: int = 30,
    ) -> None:
        self._backend = backend
        self._history_size = history_size
        self._ema_alpha = ema_alpha
        self._iou_thresh = iou_thresh
        self._max_stale = max_stale

        # per-slot state: box, smoothed embedding, raw ring buffer, staleness
        self._slots: list[dict] = []

    # ── proxy attributes so the tracker sees a normal backend ──

    @property
    def input_shape(self):  # type: ignore[override]
        return getattr(self._backend, "input_shape", (256, 128))

    @property
    def half(self):  # type: ignore[override]
        return getattr(self._backend, "half", False)

    @property
    def device(self):  # type: ignore[override]
        return getattr(self._backend, "device", None)

    @property
    def mean_array(self):  # type: ignore[override]
        return getattr(self._backend, "mean_array", None)

    @property
    def std_array(self):  # type: ignore[override]
        return getattr(self._backend, "std_array", None)

    # ── core ──

    def get_features(self, xyxys: np.ndarray, img: np.ndarray) -> np.ndarray:
        raw = self._backend.get_features(xyxys, img)
        if raw.size == 0:
            self._age_slots()
            return raw

        xyxys = np.asarray(xyxys, dtype=np.float32)
        if xyxys.ndim == 1:
            xyxys = xyxys.reshape(1, -1)
        boxes = xyxys[:, :4]

        smoothed = np.empty_like(raw)
        matched_slots: set[int] = set()

        # greedy IoU association: for each detection find best existing slot
        if self._slots:
            slot_boxes = np.array([s["box"] for s in self._slots])
            for i in range(len(boxes)):
                best_j = -1
                best_iou = self._iou_thresh
                for j in range(len(slot_boxes)):
                    if j in matched_slots:
                        continue
                    v = _iou(boxes[i], slot_boxes[j])
                    if v > best_iou:
                        best_iou = v
                        best_j = j
                if best_j >= 0:
                    matched_slots.add(best_j)
                    s = self._slots[best_j]
                    # EMA update
                    s["smooth"] = (
                        self._ema_alpha * s["smooth"]
                        + (1.0 - self._ema_alpha) * raw[i]
                    )
                    # ring buffer
                    buf = s["history"]
                    if len(buf) >= self._history_size:
                        buf.pop(0)
                    buf.append(raw[i].copy())
                    s["box"] = boxes[i].copy()
                    s["stale"] = 0
                    # return L2-normalised smoothed feature
                    feat = s["smooth"]
                    n = np.linalg.norm(feat)
                    smoothed[i] = feat / n if n > 0 else feat
                else:
                    # no match → new slot, return raw for this frame
                    self._new_slot(boxes[i], raw[i])
                    smoothed[i] = raw[i]
        else:
            # first frame — seed all slots
            for i in range(len(boxes)):
                self._new_slot(boxes[i], raw[i])
                smoothed[i] = raw[i]

        self._age_slots()
        return smoothed

    def _new_slot(self, box: np.ndarray, feat: np.ndarray) -> None:
        self._slots.append(
            {
                "box": box.copy(),
                "smooth": feat.copy(),
                "history": [feat.copy()],
                "stale": 0,
            }
        )

    def _age_slots(self) -> None:
        for s in self._slots:
            s["stale"] += 1
        self._slots = [s for s in self._slots if s["stale"] <= self._max_stale]

    # ── passthrough / proxy ──

    def get_crops(self, xyxys: np.ndarray, img: np.ndarray):  # type: ignore[override]
        return self._backend.get_crops(xyxys, img)

    def warmup(self) -> None:
        self._backend.warmup()

    def inference_preprocess(self, x):  # type: ignore[override]
        return self._backend.inference_preprocess(x)

    def inference_postprocess(self, x):  # type: ignore[override]
        return self._backend.inference_postprocess(x)

    def forward(self, im_batch):  # type: ignore[override]
        return self._backend.forward(im_batch)

    def reset(self) -> None:
        self._slots.clear()
