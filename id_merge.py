"""ID stability helpers for robust person tracking.

ShortGapIdMerger       — re-associates IDs lost for a few frames at a nearby position.
AppearanceReIDMerger   — feature-gallery re-identification across long gaps.
TrajectoryGuard        — rejects impossible identity jumps using velocity prediction.
AntiSwapGuard          — detects and reverses ID swaps when two tracks cross paths.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

Track = tuple[int, int, float, int, int, int, int]

Pt = tuple[float, float]


def _centroid(x1: int, y1: int, x2: int, y2: int) -> Pt:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(a: Pt, b: Pt) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ── Short-gap ID merge ───────────────────────────────────────────────────

Bbox = tuple[int, int, int, int]


@dataclass
class _LostTrack:
    canonical_id: int
    lost_frame: int
    centroid: Pt
    bbox: Bbox


def _size_ratio(a: Bbox, b: Bbox) -> float:
    """Return the ratio of the smaller box area to the larger (0..1)."""
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return min(area_a, area_b) / max(area_a, area_b)


class ShortGapIdMerger:
    """Re-map new raw IDs to recently lost canonical IDs when gap < *max_gap*
    frames, centroid distance is within *max_distance_px*, AND box sizes are
    consistent (prevents merging a different person who walks through the
    same area).
    """

    MAX_SIZE_RATIO = 0.30  # reject merge if area ratio < 30% (allow pose/cropping variance)

    def __init__(self, *, max_gap: int = 15, max_distance_px: float = 250.0) -> None:
        self.max_gap = max_gap
        self.max_distance_px = max_distance_px
        self._raw_to_canonical: dict[int, int] = {}
        self._prev_canonical: set[int] = set()
        self._last_centroid: dict[int, Pt] = {}
        self._last_bbox: dict[int, Bbox] = {}
        self._lost: list[_LostTrack] = []
        self._merge_count = 0

    @property
    def merge_count(self) -> int:
        return self._merge_count

    @property
    def raw_to_canonical(self) -> dict[int, int]:
        return self._raw_to_canonical

    def merge(self, frame_idx: int, tracks: list[Track]) -> list[Track]:
        raw_centroids: dict[int, Pt] = {}
        raw_bboxes: dict[int, Bbox] = {}
        for track_id, _cls, _sc, x1, y1, x2, y2 in tracks:
            raw_centroids[track_id] = _centroid(x1, y1, x2, y2)
            raw_bboxes[track_id] = (x1, y1, x2, y2)

        # Collect canonical IDs already claimed by existing mappings this frame
        claimed: set[int] = set()
        for raw_id in raw_centroids:
            if raw_id in self._raw_to_canonical:
                claimed.add(self._raw_to_canonical[raw_id])

        for raw_id, pos in raw_centroids.items():
            if raw_id in self._raw_to_canonical:
                continue
            merged = self._find_merge_target(frame_idx, pos, raw_bboxes[raw_id], claimed)
            if merged is not None:
                self._raw_to_canonical[raw_id] = merged
                claimed.add(merged)
                self._merge_count += 1
            else:
                self._raw_to_canonical[raw_id] = raw_id
                claimed.add(raw_id)

        remapped: list[Track] = []
        this_canonical: set[int] = set()
        for track_id, cls_id, score, x1, y1, x2, y2 in tracks:
            canon = self._raw_to_canonical[track_id]
            if canon in this_canonical:
                # Break stale mapping to avoid duplicate IDs in the same frame
                self._raw_to_canonical[track_id] = track_id
                canon = track_id
            this_canonical.add(canon)
            self._last_centroid[canon] = raw_centroids[track_id]
            self._last_bbox[canon] = raw_bboxes[track_id]
            remapped.append((canon, cls_id, score, x1, y1, x2, y2))

        for canon_id in self._prev_canonical - this_canonical:
            if canon_id in self._last_centroid:
                self._lost.append(
                    _LostTrack(
                        canon_id, frame_idx,
                        self._last_centroid[canon_id],
                        self._last_bbox.get(canon_id, (0, 0, 1, 1)),
                    )
                )
        self._lost = [lt for lt in self._lost if frame_idx - lt.lost_frame < self.max_gap]
        self._prev_canonical = this_canonical
        return remapped

    def _find_merge_target(
        self, frame_idx: int, pos: Pt, bbox: Bbox, claimed: set[int],
    ) -> int | None:
        best_id: int | None = None
        best_dist = self.max_distance_px + 1.0
        for lt in self._lost:
            if lt.canonical_id in claimed:
                continue
            gap = frame_idx - lt.lost_frame
            if gap <= 0 or gap >= self.max_gap:
                continue
            d = _dist(pos, lt.centroid)
            if d > self.max_distance_px or d >= best_dist:
                continue
            if _size_ratio(bbox, lt.bbox) < self.MAX_SIZE_RATIO:
                continue
            best_dist = d
            best_id = lt.canonical_id
        if best_id is not None:
            self._lost = [lt for lt in self._lost if lt.canonical_id != best_id]
        return best_id

    def reset(self) -> None:
        self._raw_to_canonical.clear()
        self._prev_canonical.clear()
        self._last_centroid.clear()
        self._last_bbox.clear()
        self._lost.clear()
        self._merge_count = 0


# ── Appearance-based re-identification across long gaps ───────────────────

@dataclass
class _GalleryEntry:
    canon_id: int
    feature: np.ndarray
    last_frame: int
    centroid: Pt
    bbox: Bbox


class AppearanceReIDMerger:
    """Re-identify tracks across long gaps using stored appearance features.

    When a new canonical ID appears, this merger checks whether the person's
    appearance matches any previously known person **that is currently LOST**
    (not visible on screen).  Only very strong appearance matches are accepted
    (cosine distance < *cosine_thresh*) to avoid merging different people.

    Critical constraints:
    - NEVER merge into an ID that is currently active on screen
    - Target must have been lost for at least *min_lost* frames
    - Box size must be consistent (prevents merging adults/children)
    """

    def __init__(
        self,
        *,
        cosine_thresh: float = 0.12,
        max_age: int = 600,
        min_lost: int = 5,
    ) -> None:
        self.cosine_thresh = cosine_thresh
        self.max_age = max_age
        self.min_lost = min_lost
        self._gallery: dict[int, _GalleryEntry] = {}
        self._active_supers: set[int] = set()
        self._canon_to_super: dict[int, int] = {}
        self._merge_count = 0

    @property
    def merge_count(self) -> int:
        return self._merge_count

    def merge(
        self,
        frame_idx: int,
        tracks: list[Track],
        feat_map: dict[int, np.ndarray],
        raw_to_canon: dict[int, int],
    ) -> list[Track]:
        canon_feat: dict[int, np.ndarray] = {}
        for raw_id, feat in feat_map.items():
            canon_id = raw_to_canon.get(raw_id, raw_id)
            if canon_id not in canon_feat:
                canon_feat[canon_id] = feat

        active_canons: set[int] = set()
        track_bboxes: dict[int, Bbox] = {}
        for tid, _c, _s, x1, y1, x2, y2 in tracks:
            active_canons.add(tid)
            track_bboxes[tid] = (x1, y1, x2, y2)

        current_supers: set[int] = set()
        for canon_id in active_canons:
            existing = self._canon_to_super.get(canon_id)
            if existing is not None:
                current_supers.add(existing)

        for canon_id in active_canons:
            if canon_id in self._canon_to_super:
                continue

            feat = canon_feat.get(canon_id)
            if feat is None:
                self._canon_to_super[canon_id] = canon_id
                current_supers.add(canon_id)
                continue

            match = self._find_gallery_match(
                canon_id, feat, frame_idx,
                track_bboxes.get(canon_id),
                current_supers,
            )
            if match is not None:
                self._canon_to_super[canon_id] = match
                current_supers.add(match)
                self._merge_count += 1
            else:
                self._canon_to_super[canon_id] = canon_id
                current_supers.add(canon_id)

        remapped: list[Track] = []
        self._active_supers = set()
        for tid, cls_id, score, x1, y1, x2, y2 in tracks:
            super_id = self._canon_to_super.get(tid, tid)
            if super_id in self._active_supers:
                # Break stale mapping to avoid duplicate IDs in the same frame
                self._canon_to_super[tid] = tid
                super_id = tid
            self._active_supers.add(super_id)
            remapped.append((super_id, cls_id, score, x1, y1, x2, y2))

        self._update_gallery(frame_idx, remapped, canon_feat)
        self._prune_gallery(frame_idx)
        return remapped

    def _find_gallery_match(
        self,
        canon_id: int,
        feat: np.ndarray,
        frame_idx: int,
        bbox: Bbox | None,
        currently_active: set[int],
    ) -> int | None:
        best_id: int | None = None
        best_cos = self.cosine_thresh + 1.0
        for gid, entry in self._gallery.items():
            if gid == canon_id:
                continue
            if gid in currently_active:
                continue
            lost_duration = frame_idx - entry.last_frame
            if lost_duration < self.min_lost or lost_duration > self.max_age:
                continue
            if bbox is not None and _size_ratio(bbox, entry.bbox) < 0.25:
                continue
            cos_dist = 1.0 - float(np.dot(feat, entry.feature))
            if cos_dist < best_cos:
                best_cos = cos_dist
                best_id = entry.canon_id
        return best_id

    def _update_gallery(
        self,
        frame_idx: int,
        tracks: list[Track],
        canon_feat: dict[int, np.ndarray],
    ) -> None:
        for tid, _c, _s, x1, y1, x2, y2 in tracks:
            feat = None
            for canon_id, super_id in self._canon_to_super.items():
                if super_id == tid and canon_id in canon_feat:
                    feat = canon_feat[canon_id]
                    break
            if feat is None:
                feat = canon_feat.get(tid)
            if feat is None:
                continue
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            self._gallery[tid] = _GalleryEntry(
                canon_id=tid,
                feature=feat.copy(),
                last_frame=frame_idx,
                centroid=_centroid(x1, y1, x2, y2),
                bbox=(x1, y1, x2, y2),
            )

    def _prune_gallery(self, frame_idx: int) -> None:
        stale = [gid for gid, e in self._gallery.items()
                 if frame_idx - e.last_frame > self.max_age]
        for gid in stale:
            del self._gallery[gid]

    def reset(self) -> None:
        self._gallery.clear()
        self._active_supers.clear()
        self._canon_to_super.clear()
        self._merge_count = 0


# ── Trajectory consistency guard ──────────────────────────────────────────

class TrajectoryGuard:
    """Reject impossible identity jumps using linear velocity prediction.

    Maintains a short position history per track ID and extrapolates the
    expected next position.  When a track's actual position deviates from
    its prediction by more than *max_jump_px*, the jump is physically
    impossible (e.g. a person walking left cannot teleport 5 m to the right
    in one frame).  The guard then searches for another track whose position
    better fits the prediction and swaps the two IDs back.
    """

    def __init__(self, *, max_jump_px: float = 200.0, history: int = 6) -> None:
        self.max_jump_px = max_jump_px
        self._history: dict[int, deque[Pt]] = {}
        self._history_len = history
        self._fix_count = 0

    @property
    def fix_count(self) -> int:
        return self._fix_count

    def _predict(self, tid: int) -> Pt | None:
        """Linear extrapolation from recent positions."""
        h = self._history.get(tid)
        if h is None or len(h) < 2:
            return None
        n = len(h)
        vx = (h[-1][0] - h[0][0]) / (n - 1)
        vy = (h[-1][1] - h[0][1]) / (n - 1)
        return (h[-1][0] + vx, h[-1][1] + vy)

    def check(self, tracks: list[Track]) -> list[Track]:
        cur: dict[int, Pt] = {}
        for tid, _c, _s, x1, y1, x2, y2 in tracks:
            cur[tid] = _centroid(x1, y1, x2, y2)

        predictions: dict[int, Pt] = {}
        jumpers: set[int] = set()
        for tid, pos in cur.items():
            pred = self._predict(tid)
            if pred is not None:
                predictions[tid] = pred
                if _dist(pos, pred) > self.max_jump_px:
                    jumpers.add(tid)

        swaps: dict[int, int] = {}
        used: set[int] = set()
        for a in jumpers:
            if a in used:
                continue
            pred_a = predictions[a]
            best_b: int | None = None
            best_gain = 0.0
            for b in cur:
                if b == a or b in used:
                    continue
                pred_b = predictions.get(b)
                err_now = _dist(cur[a], pred_a)
                err_swap = _dist(cur[b], pred_a)
                if pred_b is not None:
                    err_now += _dist(cur[b], pred_b)
                    err_swap += _dist(cur[a], pred_b)
                gain = err_now - err_swap
                if gain > self.max_jump_px * 0.5 and gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b is not None:
                swaps[a] = best_b
                swaps[best_b] = a
                used.update((a, best_b))
                self._fix_count += 1

        if swaps:
            fixed: list[Track] = []
            for tid, cls_id, score, x1, y1, x2, y2 in tracks:
                new_id = swaps.get(tid, tid)
                fixed.append((new_id, cls_id, score, x1, y1, x2, y2))
            self._update_history(
                {t[0]: _centroid(t[3], t[4], t[5], t[6]) for t in fixed}
            )
            return fixed

        self._update_history(cur)
        return tracks

    def _update_history(self, positions: dict[int, Pt]) -> None:
        for tid, pos in positions.items():
            if tid not in self._history:
                self._history[tid] = deque(maxlen=self._history_len)
            self._history[tid].append(pos)

    def reset(self) -> None:
        self._history.clear()
        self._fix_count = 0


# ── Anti-swap guard ───────────────────────────────────────────────────────

class AntiSwapGuard:
    """Detect when two IDs swap positions in a single frame and reverse it.

    Maintains a short trajectory per ID.  When the displacement vectors of a
    pair ``(A, B)`` both point toward the *other* ID's previous position
    (within *swap_radius_px*), the pair is deemed "swapped" and the IDs are
    swapped back so each person keeps their original ID.

    This addresses the most common failure mode of appearance-based trackers
    during close crossing events.
    """

    def __init__(self, *, swap_radius_px: float = 80.0, history: int = 4) -> None:
        self.swap_radius_px = swap_radius_px
        self._history: dict[int, deque[Pt]] = {}
        self._history_len = history
        self._swap_count = 0

    @property
    def swap_count(self) -> int:
        return self._swap_count

    def check(self, tracks: list[Track]) -> list[Track]:
        """Return *tracks* with any detected swaps reversed."""
        cur: dict[int, Pt] = {}
        for tid, _c, _s, x1, y1, x2, y2 in tracks:
            cur[tid] = _centroid(x1, y1, x2, y2)

        swaps: dict[int, int] = {}
        ids = list(cur.keys())
        for i in range(len(ids)):
            a = ids[i]
            if a not in self._history or len(self._history[a]) < 2:
                continue
            pa = self._history[a][-1]
            for j in range(i + 1, len(ids)):
                b = ids[j]
                if b not in self._history or len(self._history[b]) < 2:
                    continue
                pb = self._history[b][-1]
                # A jumped to where B was, B jumped to where A was
                if (
                    _dist(cur[a], pb) < self.swap_radius_px
                    and _dist(cur[b], pa) < self.swap_radius_px
                ):
                    swaps[a] = b
                    swaps[b] = a
                    self._swap_count += 1

        if not swaps:
            self._update_history(cur)
            return tracks

        fixed: list[Track] = []
        for tid, cls_id, score, x1, y1, x2, y2 in tracks:
            new_id = swaps.get(tid, tid)
            fixed.append((new_id, cls_id, score, x1, y1, x2, y2))

        self._update_history({t[0]: _centroid(t[3], t[4], t[5], t[6]) for t in fixed})
        return fixed

    def _update_history(self, positions: dict[int, Pt]) -> None:
        for tid, pos in positions.items():
            if tid not in self._history:
                self._history[tid] = deque(maxlen=self._history_len)
            self._history[tid].append(pos)

    def reset(self) -> None:
        self._history.clear()
        self._swap_count = 0
