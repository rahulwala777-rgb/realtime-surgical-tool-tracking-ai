"""
utils/visualization.py
───────────────────────
All OpenCV drawing logic lives here.  The Visualizer class is stateless
between frames — it only needs the config once at construction time.

Drawing layers (back → front)
──────────────────────────────
 1. Danger zones         (semi-transparent filled rectangles)
 2. Tool bounding boxes  (green normally, red when in danger)
 3. Track labels         (ID + class name + confidence)
 4. Info panel           (top-left: tool count, alert count, hotkeys)
 5. FPS counter          (top-right)
 6. Danger banner        (bottom: flashes when any tool is in a zone)
"""

from __future__ import annotations

import threading
import time
from typing import List, Tuple

import cv2
import numpy as np

from tracking.tracker import Track
from utils.danger_zone import DangerZone, Alert


# ---------------------------------------------------------------------------

class Visualizer:
    """
    Renders detections, tracks, danger zones, and alerts onto a frame.

    Parameters
    ----------
    config : dict
        Parsed contents of config/config.yaml.
    """

    # BGR colour palette — one colour per track-id slot (cycles)
    _PALETTE: List[Tuple[int, int, int]] = [
        (0,   220,  0),    # green
        (255, 160,  0),    # orange
        (220,   0, 220),   # magenta
        (0,   220, 220),   # cyan
        (220, 220,   0),   # yellow
        (140,   0, 255),   # violet
        (0,   140, 255),   # sky-blue
        (255, 100, 100),   # coral
    ]

    _DANGER_BGR  = (0,   0, 255)   # solid red for tools in danger
    _ALPHA_ZONE  = 0.18            # zone fill transparency

    def __init__(self, config: dict) -> None:
        self._alert_text = config.get("alerts", {}).get("danger_text", "DANGER!")
        self._show_fps   = config.get("display", {}).get("show_fps", True)
        self._sound      = config.get("alerts", {}).get("sound", False)
        self._font       = cv2.FONT_HERSHEY_SIMPLEX
        self._last_sound = 0.0      # epoch time of last beep

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def draw(
        self,
        frame:   np.ndarray,
        tracks:  List[Track],
        zones:   List[DangerZone],
        alerts:  List[Alert],
        fps:     float,
    ) -> np.ndarray:
        """
        Compose all visual layers and return the annotated frame.
        The input frame is NOT modified in place.
        """
        out = frame.copy()
        alert_ids = {a.track_id for a in alerts}

        # Layer 1 — zones
        for zone in zones:
            out = self._draw_zone(out, zone)

        # Layer 2+3 — tracked tools
        for track in tracks:
            out = self._draw_track(out, track, in_danger=(track.track_id in alert_ids))

        # Layer 4 — info panel
        out = self._draw_info_panel(out, len(tracks), len(alerts))

        # Layer 5 — FPS
        if self._show_fps:
            out = self._draw_fps(out, fps)

        # Layer 6 — danger banner
        if alerts:
            out = self._draw_banner(out, alerts)
            self._maybe_beep()

        return out

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_zone(self, frame: np.ndarray, zone: DangerZone) -> np.ndarray:
        color = tuple(int(c) for c in zone.color)

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (zone.x1, zone.y1), (zone.x2, zone.y2), color, thickness=-1)
        cv2.addWeighted(overlay, self._ALPHA_ZONE, frame, 1 - self._ALPHA_ZONE, 0, frame)

        # Border
        cv2.rectangle(frame, (zone.x1, zone.y1), (zone.x2, zone.y2), color, thickness=2)

        # Label pill above zone
        self._label_pill(frame, zone.name, (zone.x1, zone.y1 - 2), color, scale=0.52)

        return frame

    def _draw_track(self, frame: np.ndarray, track: Track, in_danger: bool) -> np.ndarray:
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        color     = self._DANGER_BGR if in_danger else self._PALETTE[track.track_id % len(self._PALETTE)]
        thickness = 3 if in_danger else 2

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Corner accents (small L-shaped marks for cleaner look)
        arm = min(16, (x2 - x1) // 4, (y2 - y1) // 4)
        for px, py, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(frame, (px, py), (px + dx * arm, py), color, thickness + 1)
            cv2.line(frame, (px, py), (px, py + dy * arm), color, thickness + 1)

        # Label
        conf_str  = f" {track.confidence:.2f}" if track.confidence > 0 else ""
        danger_tag = "  [!] DANGER" if in_danger else ""
        label = f"ID:{track.track_id}  {track.class_name}{conf_str}{danger_tag}"
        self._label_pill(frame, label, (x1, y1 - 2), color, scale=0.48)

        # Centre dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, color, -1)

        return frame

    def _draw_banner(self, frame: np.ndarray, alerts: List[Alert]) -> np.ndarray:
        h, w = frame.shape[:2]
        flash = int(time.time() * 3) % 2          # 3 Hz flash

        bg_color   = (0, 0, 220) if flash else (0, 0, 160)
        txt_color  = (255, 255, 255) if flash else (255, 210, 210)

        tool_ids_str  = str(sorted({a.track_id for a in alerts}))
        zone_names_str = ", ".join(sorted({a.zone_name for a in alerts}))
        # Use only ASCII chars — OpenCV Hershey fonts don't support Unicode
        alert_text = self._alert_text.encode("ascii", errors="replace").decode("ascii")
        msg = "  !! {}  Tool(s) {} in zone(s) {}  !!".format(
            alert_text, tool_ids_str, zone_names_str
        )

        banner_h = 46
        cv2.rectangle(frame, (0, h - banner_h), (w, h), bg_color, -1)

        (tw, th), _ = cv2.getTextSize(msg, self._font, 0.62, 2)
        tx = max(4, (w - tw) // 2)
        cv2.putText(frame, msg, (tx, h - 12), self._font, 0.62, txt_color, 2, cv2.LINE_AA)

        return frame

    def _draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(text, self._font, 0.58, 2)
        pad = 6
        cv2.rectangle(frame, (w - tw - pad * 2, pad), (w - pad, th + pad * 2), (30, 30, 30), -1)
        cv2.putText(frame, text, (w - tw - pad, th + pad),
                    self._font, 0.58, (0, 240, 100), 2, cv2.LINE_AA)
        return frame

    def _draw_info_panel(self, frame: np.ndarray, n_tracks: int, n_alerts: int) -> np.ndarray:
        lines = [
            f"Tools tracked : {n_tracks}",
            f"Active alerts : {n_alerts}",
            "P = pause   Q = quit",
        ]
        x0, y0 = 8, 10
        line_h = 22
        panel_w, panel_h = 200, len(lines) * line_h + 14

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 4, y0 - 2), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            if i == 1 and n_alerts > 0:
                color = (60, 60, 255)   # red tint for alert count
            else:
                color = (200, 200, 200)
            cv2.putText(frame, line, (x0, y0 + line_h * (i + 1)),
                        self._font, 0.48, color, 1, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _label_pill(
        self,
        frame: np.ndarray,
        text:  str,
        anchor: Tuple[int, int],   # (x, bottom_y) of pill
        color:  Tuple[int, int, int],
        scale:  float = 0.5,
    ) -> None:
        """Draw a filled rectangle with white text — the 'pill' label."""
        x, y = anchor
        (tw, th), _ = cv2.getTextSize(text, self._font, scale, 1)
        pad = 3
        cv2.rectangle(frame, (x, y - th - pad * 2), (x + tw + pad * 2, y), color, -1)
        cv2.putText(frame, text, (x + pad, y - pad),
                    self._font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _maybe_beep(self) -> None:
        """Fire 3 loud beeps in a background thread — at most once per 1.5 s."""
        if not self._sound:
            return
        now = time.time()
        if now - self._last_sound < 1.5:
            return
        self._last_sound = now
        threading.Thread(target=self._beep_worker, daemon=True).start()

    @staticmethod
    def _beep_worker() -> None:
        """Play 3 rapid high-pitch beeps (Windows winsound — no extra packages)."""
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1200, 180)   # 1200 Hz, 180 ms — clearly audible
                time.sleep(0.08)            # 80 ms gap between beeps
        except Exception:
            pass   # Non-Windows: silently skip
