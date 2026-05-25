"""Tracking quality report — detect ID swaps, fragmentation, and det/track gaps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from person_tracker import Detection, Track


def _centroid(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


@dataclass
class FrameStats:
    frame: int
    num_dets: int
    num_tracks: int
    unmatched_dets: int
    active_ids: list[int]


@dataclass
class SwapEvent:
    frame: int
    id_a: int
    id_b: int
    distance_px: float
    same_cost: float
    swap_cost: float
    confidence: float


@dataclass
class FragmentEvent:
    frame: int
    new_id: int
    lost_id: int
    frames_since_lost: int
    distance_px: float


@dataclass
class IdStats:
    track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int
    span_frames: int
    gaps: int
    longest_streak: int
    avg_score: float
    stability_pct: float


@dataclass
class TrackingReport:
    """Collects per-frame stats and emits a human-readable + JSON report."""

    swap_proximity_px: float = 120.0
    swap_cost_ratio: float = 0.82
    fragment_window: int = 15
    fragment_min_gap: int = 2  # ignore fragmentation when gap < 2 frames (merged by tracker)
    fragment_distance_px: float = 80.0
    short_track_frames: int = 30

    _frames: list[FrameStats] = field(default_factory=list, repr=False)
    _prev_centroids: dict[int, tuple[float, float]] = field(default_factory=dict, repr=False)
    _id_frames: dict[int, set[int]] = field(default_factory=dict, repr=False)
    _id_scores: dict[int, list[float]] = field(default_factory=dict, repr=False)
    _id_streaks: dict[int, int] = field(default_factory=dict, repr=False)
    _id_best_streak: dict[int, int] = field(default_factory=dict, repr=False)
    _id_gaps: dict[int, int] = field(default_factory=dict, repr=False)
    _last_seen: dict[int, int] = field(default_factory=dict, repr=False)
    _born_frame: dict[int, int] = field(default_factory=dict, repr=False)
    _recently_lost: list[tuple[int, int, tuple[float, float]]] = field(default_factory=list, repr=False)
    _swaps: list[SwapEvent] = field(default_factory=list, repr=False)
    _fragments: list[FragmentEvent] = field(default_factory=list, repr=False)

    def observe(self, frame_idx: int, dets: list[Detection], tracks: list[Track]) -> None:
        active_ids = sorted({t[0] for t in tracks})
        unmatched = max(0, len(dets) - len(tracks))
        self._frames.append(
            FrameStats(frame_idx, len(dets), len(tracks), unmatched, active_ids)
        )

        centroids: dict[int, tuple[float, float]] = {}
        for track_id, _cls, score, x1, y1, x2, y2 in tracks:
            centroids[track_id] = _centroid(x1, y1, x2, y2)

        for lost_id in set(self._prev_centroids) - set(centroids):
            self._recently_lost.append((frame_idx, lost_id, self._prev_centroids[lost_id]))
        self._recently_lost = [
            (f, i, c) for f, i, c in self._recently_lost if frame_idx - f <= self.fragment_window
        ]

        for track_id, _cls, score, x1, y1, x2, y2 in tracks:
            self._id_frames.setdefault(track_id, set()).add(frame_idx)
            self._id_scores.setdefault(track_id, []).append(score)

            if track_id in self._last_seen and frame_idx - self._last_seen[track_id] > 1:
                self._id_gaps[track_id] = self._id_gaps.get(track_id, 0) + 1
            if track_id in self._last_seen and frame_idx == self._last_seen[track_id] + 1:
                self._id_streaks[track_id] = self._id_streaks.get(track_id, 0) + 1
            else:
                self._id_streaks[track_id] = 1
            self._id_best_streak[track_id] = max(
                self._id_best_streak.get(track_id, 0), self._id_streaks[track_id]
            )

            if track_id not in self._born_frame:
                self._born_frame[track_id] = frame_idx
                self._check_fragment(frame_idx, track_id, centroids[track_id])

            self._last_seen[track_id] = frame_idx

        self._detect_swaps(frame_idx, centroids)
        self._prev_centroids = centroids

    def _check_fragment(self, frame_idx: int, new_id: int, pos: tuple[float, float]) -> None:
        for lost_f, lost_id, lost_c in self._recently_lost:
            if lost_id == new_id:
                continue
            gap = frame_idx - lost_f
            if gap <= 0 or gap > self.fragment_window:
                continue
            if gap < self.fragment_min_gap:
                continue  # short gap: old ID preserved by merger, not fragmentation
            d = _dist(pos, lost_c)
            if d <= self.fragment_distance_px:
                self._fragments.append(
                    FragmentEvent(frame_idx, new_id, lost_id, gap, round(d, 1))
                )

    def _detect_swaps(self, frame_idx: int, centroids: dict[int, tuple[float, float]]) -> None:
        if frame_idx <= 1 or not self._prev_centroids:
            return
        ids = [i for i in centroids if i in self._prev_centroids]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                c_a_prev = self._prev_centroids[a]
                c_b_prev = self._prev_centroids[b]
                if _dist(c_a_prev, c_b_prev) > self.swap_proximity_px:
                    continue
                c_a, c_b = centroids[a], centroids[b]
                same = _dist(c_a_prev, c_a) + _dist(c_b_prev, c_b)
                swap = _dist(c_a_prev, c_b) + _dist(c_b_prev, c_a)
                if same <= 0 or swap >= same * self.swap_cost_ratio:
                    continue
                conf = min(1.0, max(0.0, 1.0 - swap / same))
                self._swaps.append(
                    SwapEvent(
                        frame_idx, a, b,
                        round(_dist(c_a_prev, c_b_prev), 1),
                        round(same, 1), round(swap, 1), round(conf, 2),
                    )
                )

    def _id_statistics(self) -> list[IdStats]:
        out: list[IdStats] = []
        for tid, frames in sorted(self._id_frames.items()):
            first, last = min(frames), max(frames)
            span = last - first + 1
            seen = len(frames)
            scores = self._id_scores.get(tid, [])
            out.append(
                IdStats(
                    tid, first, last, seen, span,
                    self._id_gaps.get(tid, 0),
                    self._id_best_streak.get(tid, seen),
                    round(sum(scores) / len(scores), 3) if scores else 0.0,
                    round(100.0 * seen / span, 1) if span else 0.0,
                )
            )
        return out

    def summary(self) -> dict:
        if not self._frames:
            return {"frames_processed": 0, "message": "no data"}

        total_frames = len(self._frames)
        id_stats = self._id_statistics()
        stable = [s for s in id_stats if s.frames_seen >= self.short_track_frames and s.stability_pct >= 85]
        short = [s for s in id_stats if s.frames_seen < self.short_track_frames]
        fragmented = [s for s in id_stats if s.gaps > 0]

        det_counts = [f.num_dets for f in self._frames]
        track_counts = [f.num_tracks for f in self._frames]
        unmatched_total = sum(f.unmatched_dets for f in self._frames)
        frames_with_unmatched = sum(1 for f in self._frames if f.unmatched_dets > 0)

        return {
            "frames_processed": total_frames,
            "unique_ids": len(id_stats),
            "stable_ids": len(stable),
            "short_lived_ids": len(short),
            "fragmented_ids": len(fragmented),
            "suspected_swaps": len(self._swaps),
            "suspected_fragments": len(self._fragments),
            "detection": {
                "avg_per_frame": round(sum(det_counts) / total_frames, 2),
                "max_per_frame": max(det_counts),
                "min_per_frame": min(det_counts),
            },
            "tracking": {
                "avg_per_frame": round(sum(track_counts) / total_frames, 2),
                "max_per_frame": max(track_counts),
                "min_per_frame": min(track_counts),
            },
            "unmatched_detections": {
                "total": unmatched_total,
                "frames_with_unmatched": frames_with_unmatched,
                "pct_frames": round(100.0 * frames_with_unmatched / total_frames, 1),
            },
            "id_stability_score": round(100.0 * len(stable) / max(1, len(id_stats)), 1),
        }

    def format_text(self) -> str:
        if not self._frames:
            return "Tracking report: no frames recorded."

        s = self.summary()
        lines = [
            "",
            "=" * 60,
            "TRACKING QUALITY REPORT",
            "=" * 60,
            f"Frames processed     : {s['frames_processed']}",
            f"Unique IDs created   : {s['unique_ids']}",
            f"Stable IDs (≥{self.short_track_frames}f, ≥85% coverage) : {s['stable_ids']}",
            f"Short-lived IDs      : {s['short_lived_ids']}  (likely noise or quick swaps)",
            f"IDs with gaps        : {s['fragmented_ids']}  (disappeared then reappeared)",
            f"ID stability score   : {s['id_stability_score']}%  (higher = better)",
            "",
            "Detection vs tracking",
            f"  Avg detections/frame : {s['detection']['avg_per_frame']}",
            f"  Avg tracks/frame     : {s['tracking']['avg_per_frame']}",
            f"  Unmatched dets       : {s['unmatched_detections']['total']} total "
            f"({s['unmatched_detections']['pct_frames']}% of frames had extra detections)",
            "",
        ]

        if self._swaps:
            lines.append(
                f"SUSPECTED ID SWAPS ({len(self._swaps)}) — pairs may have switched at crossings:"
            )
            for ev in self._swaps[:20]:
                lines.append(
                    f"  frame {ev.frame:5d}  ID {ev.id_a} <-> ID {ev.id_b}  "
                    f"dist={ev.distance_px}px  conf={ev.confidence:.0%}  "
                    f"(keep={ev.same_cost}px vs swap={ev.swap_cost}px)"
                )
            if len(self._swaps) > 20:
                lines.append(f"  ... and {len(self._swaps) - 20} more")
            lines.append("")
        else:
            lines.append("Suspected ID swaps : none detected")
            lines.append("")

        if self._fragments:
            lines.append(
                f"SUSPECTED FRAGMENTATION ({len(self._fragments)}) — new ID where another recently vanished:"
            )
            for ev in self._fragments[:15]:
                lines.append(
                    f"  frame {ev.frame:5d}  new ID {ev.new_id} near lost ID {ev.lost_id}  "
                    f"({ev.frames_since_lost}f ago, {ev.distance_px}px)"
                )
            if len(self._fragments) > 15:
                lines.append(f"  ... and {len(self._fragments) - 15} more")
            lines.append("")

        id_stats = self._id_statistics()
        if id_stats:
            lines.append("Per-ID journey (longest / most stable first):")
            ranked = sorted(id_stats, key=lambda x: (-x.frames_seen, -x.stability_pct))
            for st in ranked[:15]:
                flag = ""
                if st.frames_seen < self.short_track_frames:
                    flag = " [short]"
                elif st.gaps > 0:
                    flag = " [gaps]"
                elif st.stability_pct >= 90:
                    flag = " [stable]"
                lines.append(
                    f"  ID {st.track_id:4d}  frames {st.first_frame}-{st.last_frame}  "
                    f"seen {st.frames_seen}/{st.span_frames} ({st.stability_pct:.0f}%)  "
                    f"best streak {st.longest_streak}  avg score {st.avg_score:.2f}{flag}"
                )
            if len(ranked) > 15:
                lines.append(f"  ... and {len(ranked) - 15} more IDs")

        lines.extend([
            "",
            "How to read this:",
            "  - Stable IDs with high % coverage = good persistent tracking",
            "  - Suspected swaps at crossings = tune APPEARANCE_THRESH in config.py (lower=stricter)",
            "  - Fragmentation = same person got a new ID after occlusion",
            "  - Unmatched detections = detector found person not yet confirmed by tracker",
            "=" * 60,
        ])
        return "\n".join(lines)

    def to_payload(self) -> dict:
        return {
            "summary": self.summary(),
            "suspected_swaps": [asdict(e) for e in self._swaps],
            "suspected_fragments": [asdict(e) for e in self._fragments],
            "per_id": [asdict(s) for s in self._id_statistics()],
            "per_frame": [asdict(f) for f in self._frames],
        }

    def save_json(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")

    def save_html(self, path: Path | str, *, title: str = "Tracking Quality Report") -> None:
        from report.generate_html import save_report_html

        save_report_html(self.to_payload(), path, title=title)

    def save_csv(self, path: Path | str) -> None:
        path = Path(path)
        lines = ["frame,num_dets,num_tracks,unmatched_dets,active_ids"]
        for f in self._frames:
            ids = ";".join(str(i) for i in f.active_ids)
            lines.append(f"{f.frame},{f.num_dets},{f.num_tracks},{f.unmatched_dets},{ids}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
