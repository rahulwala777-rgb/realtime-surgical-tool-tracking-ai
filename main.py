"""
main.py
────────
Entry point for the AI-Based Real-Time Surgical Tool Tracking system.

Usage
─────
  # Webcam (default):
  python main.py

  # Video file:
  python main.py --source surgery_clip.mp4

  # Custom config:
  python main.py --config path/to/config.yaml

Keyboard controls (while the window is focused)
────────────────────────────────────────────────
  P  →  Pause / resume
  Q  →  Quit
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque

import cv2
import yaml

# Ensure the project root is importable from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detection.detector import SurgicalToolDetector
from tracking.tracker import ToolTracker
from utils.danger_zone import DangerZoneManager
from utils.visualization import Visualizer


# ─────────────────────────── helpers ────────────────────────────────────────

class FPSCounter:
    """Rolling average FPS over the last *window* frames."""

    def __init__(self, window: int = 30) -> None:
        self._times: Deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def open_video(source) -> cv2.VideoCapture:
    """Open a video capture, trying the requested source then falling back to webcam 0."""
    # Convert string digits to int so cv2 recognises them as device indices
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap

    print(f"[WARN] Cannot open source '{source}'.  Falling back to webcam 0 ...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap

    print("[ERROR] No video source available.  Check your camera or file path.")
    sys.exit(1)


# ─────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Real-Time Surgical Tool Tracking with Danger Zone Alerting"
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config.yaml  (default: config/config.yaml)",
    )
    parser.add_argument(
        "--source",
        help="Override video source: integer webcam index or path to video file",
    )
    args = parser.parse_args()

    # ── Load configuration ────────────────────────────────────────────────
    config = load_config(args.config)
    if args.source is not None:
        config["video"]["source"] = args.source

    window_title = config.get("display", {}).get(
        "window_title", "Surgical Tool Tracking System"
    )

    # ── Initialise components ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Surgical Tool Tracking — Initialising")
    print("=" * 60)

    detector     = SurgicalToolDetector(config)
    tracker      = ToolTracker(config)
    zone_manager = DangerZoneManager(config)
    visualizer   = Visualizer(config)
    fps_counter  = FPSCounter()

    # ── Open video source ─────────────────────────────────────────────────
    source = config["video"]["source"]
    cap    = open_video(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config["video"].get("width",  1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["video"].get("height",  720))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n[INFO] Video source opened — resolution: {actual_w}×{actual_h}")
    print("[INFO] Press  P  to pause/resume   |   Q  to quit\n")

    # ── Create a resizable window fixed to display dimensions ─────────────
    disp_cfg = config.get("display", {})
    disp_w   = int(disp_cfg.get("window_width",  1280))
    disp_h   = int(disp_cfg.get("window_height",  720))

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, disp_w, disp_h)

    # ── Main loop ─────────────────────────────────────────────────────────
    paused = False
    frame  = None     # keep last frame so 'pause' still shows something

    while True:
        if not paused:
            ret, frame = cap.read()

            if not ret:
                # End of video file — loop or exit
                if isinstance(source, str) and not source.isdigit():
                    print("[INFO] End of video file — restarting ...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[ERROR] Failed to read frame from webcam.")
                    break

            # ── Detection ────────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Tracking ─────────────────────────────────────────────────
            tracks = tracker.update(detections, frame)

            # ── Danger-zone check ─────────────────────────────────────────
            alerts = zone_manager.check_alerts(tracks)

            # Console log
            if config.get("alerts", {}).get("console_log", True):
                for alert in alerts:
                    print(
                        f"[ALERT] Tool ID={alert.track_id} entered "
                        f"'{alert.zone_name}'"
                    )

            # ── FPS ───────────────────────────────────────────────────────
            fps = fps_counter.tick()

            # ── Render ────────────────────────────────────────────────────
            frame = visualizer.draw(frame, tracks, zone_manager.zones, alerts, fps)

        # Always show something (paused or live)
        if frame is not None:
            # Scale to display size so the full frame fits without clipping
            display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_title, display_frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit signal received.")
            break
        elif key == ord("p"):
            paused = not paused
            state  = "PAUSED" if paused else "RESUMED"
            print(f"[INFO] {state}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System shutdown complete.")


if __name__ == "__main__":
    main()
