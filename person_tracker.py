"""BoTSORT tracking with FastReID — IoU-first matching prevents swaps during close crossings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from config import (
    APPEARANCE_THRESH,
    CMC_METHOD,
    EMBED_EMA_ALPHA,
    EMBED_HISTORY,
    EMBED_SMOOTH,
    FASTREID_ENABLED,
    FASTREID_WEIGHTS,
    FRAME_RATE,
    FUSE_FIRST_ASSOCIATE,
    MATCH_THRESH,
    NEW_TRACK_THRESH,
    PROXIMITY_THRESH,
    REID_MERGE_COSINE_THRESH,
    REID_MERGE_MAX_AGE,
    REID_MERGE_MIN_LOST,
    REID_WEIGHTS,
    TRACK_BUFFER,
    TRACK_HIGH_THRESH,
    TRACK_LOW_THRESH,
    TRACK_MERGE_DISTANCE_PX,
    TRACK_MERGE_MAX_GAP,
)
from id_merge import AppearanceReIDMerger, ShortGapIdMerger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Track = tuple[int, int, float, int, int, int, int]  # id, cls, score, x1, y1, x2, y2
Detection = tuple[int, float, int, int, int, int]  # cls, score, x1, y1, x2, y2


@dataclass
class TrackConfig:
    reid_weights: str = REID_WEIGHTS
    fastreid_enabled: bool = FASTREID_ENABLED
    fastreid_weights: str = FASTREID_WEIGHTS
    embed_smooth: bool = EMBED_SMOOTH
    embed_history: int = EMBED_HISTORY
    embed_ema_alpha: float = EMBED_EMA_ALPHA
    device: str = "0"
    det_conf: float = 0.30
    track_high_thresh: float = TRACK_HIGH_THRESH
    track_low_thresh: float = TRACK_LOW_THRESH
    new_track_thresh: float = NEW_TRACK_THRESH
    track_buffer: int = TRACK_BUFFER
    match_thresh: float = MATCH_THRESH
    proximity_thresh: float = PROXIMITY_THRESH
    appearance_thresh: float = APPEARANCE_THRESH
    cmc_method: str = CMC_METHOD
    frame_rate: int = FRAME_RATE
    fuse_first_associate: bool = FUSE_FIRST_ASSOCIATE
    merge_max_gap: int = TRACK_MERGE_MAX_GAP
    merge_distance_px: float = TRACK_MERGE_DISTANCE_PX
    reid_cosine_thresh: float = REID_MERGE_COSINE_THRESH
    reid_max_age: int = REID_MERGE_MAX_AGE
    reid_min_lost: int = REID_MERGE_MIN_LOST
    half: bool = True


def dets_to_xyxy6(dets: list[Detection]) -> NDArray[np.float32]:
    if not dets:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(
        [[x1, y1, x2, y2, score, cls_id] for cls_id, score, x1, y1, x2, y2 in dets],
        dtype=np.float32,
    )


def tracks_from_boxmot(raw: NDArray[np.float32]) -> list[Track]:
    if raw is None or len(raw) == 0:
        return []
    return [
        (int(r[4]), int(r[6]), float(r[5]), int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        for r in raw
    ]


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    bgr = np.random.default_rng(track_id * 9973 + 17).integers(64, 255, size=3, dtype=np.int32)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class BotSortPersonTracker:
    """BoTSORT tracker with FastReID.

    BoTSORT does IoU-first matching: each detection goes to the spatially
    nearest track.  Appearance features are used only as a secondary
    tie-breaker.  This prevents ID swaps during close crossings.

    Lost tracks are kept in a separate pool for up to ``track_buffer``
    frames and can be re-identified by appearance when they reappear.
    Only detections with confidence >= ``new_track_thresh`` can create
    new track IDs, preventing ID proliferation from noisy detections.
    """

    def __init__(self, config: TrackConfig | None = None, **kwargs: object) -> None:
        from boxmot.trackers.tracker_zoo import create_tracker

        config = config or TrackConfig(**kwargs)  # type: ignore[arg-type]
        self.config = config

        reid_model = self._build_reid_backend(config)
        self._tracker = create_tracker(
            "botsort",
            reid_weights=Path(config.reid_weights),
            device=config.device,
            half=config.half,
            reid_model=reid_model,
            evolve_param_dict={
                "det_thresh": config.det_conf,
                "track_high_thresh": config.track_high_thresh,
                "track_low_thresh": config.track_low_thresh,
                "new_track_thresh": config.new_track_thresh,
                "track_buffer": config.track_buffer,
                "match_thresh": config.match_thresh,
                "proximity_thresh": config.proximity_thresh,
                "appearance_thresh": config.appearance_thresh,
                "cmc_method": config.cmc_method,
                "frame_rate": config.frame_rate,
                "fuse_first_associate": config.fuse_first_associate,
                "with_reid": True,
            },
        )
        self._merger = ShortGapIdMerger(
            max_gap=config.merge_max_gap,
            max_distance_px=config.merge_distance_px,
        )
        self._reid_merger = AppearanceReIDMerger(
            cosine_thresh=config.reid_cosine_thresh,
            max_age=config.reid_max_age,
            min_lost=config.reid_min_lost,
        )
        self._frame_idx = 0
        self._active_ids: set[int] = set()

    @staticmethod
    def _build_reid_backend(config: TrackConfig) -> object | None:
        """Build FastReID backend (optionally smoothed), else return None."""
        if not config.fastreid_enabled:
            return None
        try:
            import torch
            from fastreid_backend import FastReIDBackend, SmoothedReIDBackend

            device_str = config.device
            if device_str.isdigit():
                device = torch.device(f"cuda:{device_str}")
            elif device_str in ("cpu",):
                device = torch.device("cpu")
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            weights = Path(config.fastreid_weights) if config.fastreid_weights else None
            backend = FastReIDBackend(device=device, half=config.half, weights=weights)

            if config.embed_smooth:
                backend = SmoothedReIDBackend(
                    backend,
                    history_size=config.embed_history,
                    ema_alpha=config.embed_ema_alpha,
                )
                print(
                    f"ReID: FastReID SBS R50-IBN (MSMT17) — 2048-dim features, "
                    f"EMA-smoothed (history={config.embed_history}, α={config.embed_ema_alpha})"
                )
            else:
                print("ReID: FastReID SBS R50-IBN (MSMT17) — 2048-dim features")

            logger.info("FastReID backend loaded for BoTSORT (SBS R50-IBN, MSMT17)")
            return backend
        except Exception as exc:
            logger.warning("FastReID init failed (%s), falling back to OSNet", exc)
            print(f"FastReID unavailable ({exc}), falling back to OSNet")
            return None

    def update(self, frame_bgr: np.ndarray, dets: list[Detection], *, frame_idx: int = 0) -> list[Track]:
        self._frame_idx = frame_idx
        raw = self._tracker.update(dets_to_xyxy6(dets), frame_bgr)
        tracks = tracks_from_boxmot(raw)

        feat_map = self._extract_strack_features()

        tracks = self._merger.merge(frame_idx, tracks)
        tracks = self._reid_merger.merge(
            frame_idx, tracks, feat_map, self._merger.raw_to_canonical,
        )
        tracks = self._dedup(tracks)
        self._active_ids.update(t[0] for t in tracks)
        return tracks

    @staticmethod
    def _dedup(tracks: list[Track]) -> list[Track]:
        """Guarantee no two tracks share the same ID in a single frame."""
        seen: set[int] = set()
        out: list[Track] = []
        for t in tracks:
            if t[0] not in seen:
                seen.add(t[0])
                out.append(t)
        return out

    def _extract_strack_features(self) -> dict[int, np.ndarray]:
        """Extract smoothed appearance features from BoTSORT's internal STracks."""
        feat_map: dict[int, np.ndarray] = {}
        for strack in self._tracker.active_tracks:
            if strack.smooth_feat is not None:
                feat_map[strack.id] = strack.smooth_feat.copy()
        for strack in getattr(self._tracker, "lost_stracks", []):
            if strack.smooth_feat is not None and strack.id not in feat_map:
                feat_map[strack.id] = strack.smooth_feat.copy()
        return feat_map

    @property
    def active_ids(self) -> set[int]:
        return set(self._active_ids)

    @property
    def merge_stats(self) -> str:
        return (
            f"Short-gap merges: {self._merger.merge_count}  |  "
            f"ReID merges: {self._reid_merger.merge_count}"
        )

    def reset(self) -> None:
        self._active_ids.clear()
        self._merger.reset()
        self._reid_merger.reset()
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()


# backward-compatible alias used by pipeline.py
StrongSortPersonTracker = BotSortPersonTracker


class TrackJourney:
    def __init__(self) -> None:
        self._first: dict[int, int] = {}
        self._last: dict[int, int] = {}

    def observe(self, frame_idx: int, tracks: list[Track]) -> None:
        for track_id, *_ in tracks:
            if track_id not in self._first:
                self._first[track_id] = frame_idx
            self._last[track_id] = frame_idx

    def summary(self) -> str:
        if not self._first:
            return "no tracks"
        parts = [f"ID {tid} frames {self._first[tid]}-{self._last[tid]}" for tid in sorted(self._first)]
        return "; ".join(parts[:8]) + (" …" if len(parts) > 8 else "")


def draw_tracks(
    frame_bgr: np.ndarray,
    tracks: list[Track],
    *,
    lite: bool = False,
    trails: dict[int, list[tuple[int, int]]] | None = None,
) -> np.ndarray:
    if not tracks and not trails:
        return frame_bgr
    vis = frame_bgr if lite else frame_bgr.copy()
    if trails:
        for track_id, points in trails.items():
            if len(points) < 2:
                continue
            color = _color_for_id(track_id)
            for i in range(1, len(points)):
                cv2.line(vis, points[i - 1], points[i], color, 2, cv2.LINE_AA)
    for track_id, _cls, score, x1, y1, x2, y2 in tracks:
        color = _color_for_id(track_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if lite:
            continue
        label = f"ID {track_id} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
        cv2.putText(vis, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return vis
