"""Video overlays: frame HUD and high-visibility track labels."""

from __future__ import annotations

import cv2
import numpy as np

from person_tracker import Track, _color_for_id


def _scale_for_frame(h: int, w: int) -> tuple[float, int, int]:
    """Font scale, line thickness, box thickness scaled to resolution."""
    base = max(h, w)
    font_scale = max(0.9, min(2.2, base / 900.0))
    line_th = max(2, int(base / 500))
    box_th = max(3, int(base / 350))
    return font_scale, line_th, box_th


def draw_frame_hud(
    frame_bgr: np.ndarray,
    frame_idx: int,
    *,
    num_tracked: int = 0,
    active_ids: list[int] | None = None,
) -> np.ndarray:
    """Top-left frame number + track count (burned into saved video)."""
    vis = frame_bgr
    h, w = vis.shape[:2]
    font_scale, line_th, _ = _scale_for_frame(h, w)
    ids = sorted(active_ids or [])
    id_str = ", ".join(str(i) for i in ids[:12])
    if len(ids) > 12:
        id_str += f", +{len(ids) - 12}"

    lines = [f"Frame {frame_idx}", f"Tracked: {num_tracked}"]
    if ids:
        lines.append(f"IDs: {id_str}")

    pad = max(8, int(min(h, w) / 120))
    y = pad + int(28 * font_scale)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, line_th + 1)
        cv2.rectangle(vis, (pad - 4, y - th - 6), (pad + tw + 8, y + 6), (0, 0, 0), -1)
        cv2.putText(
            vis, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85,
            (255, 255, 255), line_th + 1, cv2.LINE_AA,
        )
        y += th + int(12 * font_scale)
    return vis


def draw_tracks_bold(
    frame_bgr: np.ndarray,
    tracks: list[Track],
    *,
    trails: dict[int, list[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """Draw tracks with thick boxes and large high-contrast ID labels."""
    if not tracks and not trails:
        return frame_bgr

    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    font_scale, line_th, box_th = _scale_for_frame(h, w)
    id_font = font_scale * 1.15

    if trails:
        trail_th = max(2, box_th - 1)
        for track_id, points in trails.items():
            if len(points) < 2:
                continue
            color = _color_for_id(track_id)
            for i in range(1, len(points)):
                cv2.line(vis, points[i - 1], points[i], color, trail_th, cv2.LINE_AA)

    for track_id, _cls, score, x1, y1, x2, y2 in tracks:
        color = _color_for_id(track_id)
        # Outer dark border + inner colored box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0), box_th + 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, box_th)

        label = f"ID {track_id}"
        sub = f"{score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, id_font, line_th + 1)
        (_, th2), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.55, line_th)
        label_h = th + th2 + baseline + 10
        label_w = max(tw, int(tw * 0.6)) + 16

        # Label above box, or inside top if no room
        lx = x1
        ly = y1 - 8 if y1 > label_h + 8 else y1 + 4
        if ly < 0:
            ly = y1 + 4

        cv2.rectangle(vis, (lx, ly), (lx + label_w, ly + label_h), (0, 0, 0), -1)
        cv2.rectangle(vis, (lx, ly), (lx + label_w, ly + label_h), color, max(2, box_th - 1))

        ty = ly + th + 4
        # Black outline + white text for readability
        cv2.putText(vis, label, (lx + 8, ty), cv2.FONT_HERSHEY_DUPLEX, id_font, (0, 0, 0), line_th + 3, cv2.LINE_AA)
        cv2.putText(vis, label, (lx + 8, ty), cv2.FONT_HERSHEY_DUPLEX, id_font, (255, 255, 255), line_th + 1, cv2.LINE_AA)
        cv2.putText(
            vis, sub, (lx + 8, ty + th2 + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.55,
            (220, 220, 220), line_th, cv2.LINE_AA,
        )

        # Large ID watermark inside box (center) for crowded scenes
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        id_only = str(track_id)
        (iw, ih), _ = cv2.getTextSize(id_only, cv2.FONT_HERSHEY_DUPLEX, id_font * 1.4, line_th + 2)
        ix, iy = cx - iw // 2, cy + ih // 2
        if (x2 - x1) > iw + 20 and (y2 - y1) > ih + 20:
            cv2.putText(vis, id_only, (ix, iy), cv2.FONT_HERSHEY_DUPLEX, id_font * 1.4, (0, 0, 0), line_th + 4, cv2.LINE_AA)
            cv2.putText(vis, id_only, (ix, iy), cv2.FONT_HERSHEY_DUPLEX, id_font * 1.4, color, line_th + 1, cv2.LINE_AA)

    return vis
