"""StrongSORT tracking — works with ORT or TensorRT detections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from config import (
    REID_WEIGHTS,
    TRACK_EMA_ALPHA,
    TRACK_MAX_AGE,
    TRACK_MAX_COS_DIST,
    TRACK_MAX_IOU_DIST,
    TRACK_MC_LAMBDA,
    TRACK_MERGE_DISTANCE_PX,
    TRACK_MERGE_MAX_GAP,
    TRACK_N_INIT,
    TRACK_NN_BUDGET,
)
from id_merge import ShortGapIdMerger

if TYPE_CHECKING:
    from numpy.typing import NDArray

Track = tuple[int, int, float, int, int, int, int]  # id, cls, score, x1, y1, x2, y2
Detection = tuple[int, float, int, int, int, int]  # cls, score, x1, y1, x2, y2


@dataclass
class TrackConfig:
    reid_weights: str = REID_WEIGHTS
    device: str = "0"
    det_conf: float = 0.30
    max_age: int = TRACK_MAX_AGE
    n_init: int = TRACK_N_INIT
    max_cos_dist: float = TRACK_MAX_COS_DIST
    max_iou_dist: float = TRACK_MAX_IOU_DIST
    nn_budget: int = TRACK_NN_BUDGET
    mc_lambda: float = TRACK_MC_LAMBDA
    ema_alpha: float = TRACK_EMA_ALPHA
    merge_max_gap: int = TRACK_MERGE_MAX_GAP
    merge_distance_px: float = TRACK_MERGE_DISTANCE_PX
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


class StrongSortPersonTracker:
    def __init__(self, config: TrackConfig | None = None, **kwargs: object) -> None:
        from boxmot.trackers.tracker_zoo import create_tracker

        config = config or TrackConfig(**kwargs)  # type: ignore[arg-type]
        self.config = config
        self._tracker = create_tracker(
            "strongsort",
            reid_weights=Path(config.reid_weights),
            device=config.device,
            half=config.half,
            evolve_param_dict={
                "min_conf": config.det_conf,
                "max_age": config.max_age,
                "n_init": config.n_init,
                "max_cos_dist": config.max_cos_dist,
                "max_iou_dist": config.max_iou_dist,
                "nn_budget": config.nn_budget,
                "mc_lambda": config.mc_lambda,
                "ema_alpha": config.ema_alpha,
            },
        )
        self._merger = ShortGapIdMerger(
            max_gap=config.merge_max_gap,
            max_distance_px=config.merge_distance_px,
        )
        self._frame_idx = 0
        self._active_ids: set[int] = set()

    def update(self, frame_bgr: np.ndarray, dets: list[Detection], *, frame_idx: int = 0) -> list[Track]:
        self._frame_idx = frame_idx
        raw = self._tracker.update(dets_to_xyxy6(dets), frame_bgr)
        tracks = tracks_from_boxmot(raw)
        tracks = self._merger.merge(frame_idx, tracks)
        self._active_ids.update(t[0] for t in tracks)
        return tracks

    @property
    def active_ids(self) -> set[int]:
        return set(self._active_ids)

    def reset(self) -> None:
        self._active_ids.clear()
        self._merger.reset()
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()


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
