"""Short-gap ID merge: keep old IDs when StrongSORT creates a new one within 1 frame."""

from __future__ import annotations

from dataclasses import dataclass

Track = tuple[int, int, float, int, int, int, int]


def _centroid(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@dataclass
class _LostTrack:
    canonical_id: int
    lost_frame: int
    centroid: tuple[float, float]


class ShortGapIdMerger:
    """
    When StrongSORT assigns a new raw ID within ``max_gap`` frames of a lost ID
    at a nearby position, remap the new ID to the old canonical ID.

    Default max_gap=2 → merge when gap is 0 or 1 frame (gap < 2f).
    """

    def __init__(self, *, max_gap: int = 2, max_distance_px: float = 120.0) -> None:
        self.max_gap = max_gap
        self.max_distance_px = max_distance_px
        self._raw_to_canonical: dict[int, int] = {}
        self._prev_canonical: set[int] = set()
        self._last_centroid: dict[int, tuple[float, float]] = {}
        self._lost: list[_LostTrack] = []
        self._merge_count = 0

    @property
    def merge_count(self) -> int:
        return self._merge_count

    def merge(self, frame_idx: int, tracks: list[Track]) -> list[Track]:
        raw_centroids: dict[int, tuple[float, float]] = {}
        for track_id, _cls, _sc, x1, y1, x2, y2 in tracks:
            raw_centroids[track_id] = _centroid(x1, y1, x2, y2)

        # Register new raw IDs: try to inherit a recently lost canonical ID
        for raw_id, pos in raw_centroids.items():
            if raw_id in self._raw_to_canonical:
                continue
            merged = self._find_merge_target(frame_idx, pos)
            if merged is not None:
                self._raw_to_canonical[raw_id] = merged
                self._merge_count += 1
            else:
                self._raw_to_canonical[raw_id] = raw_id

        # Remap output IDs
        remapped: list[Track] = []
        this_canonical: set[int] = set()
        for track_id, cls_id, score, x1, y1, x2, y2 in tracks:
            canon = self._raw_to_canonical[track_id]
            this_canonical.add(canon)
            self._last_centroid[canon] = raw_centroids[track_id]
            remapped.append((canon, cls_id, score, x1, y1, x2, y2))

        # IDs that disappeared this frame → lost pool for short-gap re-merge
        for canon_id in self._prev_canonical - this_canonical:
            if canon_id in self._last_centroid:
                self._lost.append(
                    _LostTrack(canon_id, frame_idx, self._last_centroid[canon_id])
                )

        self._lost = [
            lt for lt in self._lost if frame_idx - lt.lost_frame < self.max_gap
        ]
        self._prev_canonical = this_canonical
        return remapped

    def _find_merge_target(self, frame_idx: int, pos: tuple[float, float]) -> int | None:
        best_id: int | None = None
        best_dist = self.max_distance_px + 1.0
        for lt in self._lost:
            gap = frame_idx - lt.lost_frame
            if gap <= 0 or gap >= self.max_gap:
                continue
            d = _dist(pos, lt.centroid)
            if d <= self.max_distance_px and d < best_dist:
                best_dist = d
                best_id = lt.canonical_id
        if best_id is not None:
            self._lost = [lt for lt in self._lost if lt.canonical_id != best_id]
        return best_id

    def reset(self) -> None:
        self._raw_to_canonical.clear()
        self._prev_canonical.clear()
        self._last_centroid.clear()
        self._lost.clear()
        self._merge_count = 0
