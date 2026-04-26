"""
tracking/tracker.py
────────────────────
Wraps deep_sort_realtime to assign persistent IDs to detected surgical tools
across frames.

DeepSORT input  → list of ([x, y, w, h], confidence, class_label)
DeepSORT output → list of Track objects with .track_id and .to_ltrb()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from detection.detector import Detection

# deep_sort_realtime uses %-style logging internally; some versions have a
# format-string bug (too many args) that propagates as TypeError.
# Raising to ERROR suppresses those debug/info messages entirely.
logging.getLogger("deep_sort_realtime").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """A confirmed, actively tracked surgical tool."""
    track_id:    int
    bbox:        List[float]   # [x1, y1, x2, y2]  absolute pixels
    class_name:  str
    confidence:  float
    is_confirmed: bool = True


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ToolTracker:
    """
    Multi-object tracker backed by DeepSORT.

    Maintains a mapping from track_id → class_name so labels survive
    frames where no fresh detection is associated with the track.

    Parameters
    ----------
    config : dict
        Parsed contents of config/config.yaml.
    """

    def __init__(self, config: dict) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort  # lazy import

        tcfg = config.get("tracking", {})

        self._tracker = DeepSort(
            max_age             = tcfg.get("max_age",             30),
            n_init              = tcfg.get("n_init",               3),
            max_iou_distance    = tcfg.get("max_iou_distance",   0.7),
            max_cosine_distance = tcfg.get("max_cosine_distance", 0.3),
            nn_budget           = tcfg.get("nn_budget",          100),
        )

        # Persist class label across frames for each track id
        self._id_to_class: dict[int, str] = {}
        print("[Tracker] DeepSORT initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Feed new detections into the tracker and return confirmed tracks.

        Parameters
        ----------
        detections : List[Detection]
            Output of SurgicalToolDetector.detect()
        frame : np.ndarray
            Current BGR frame (used by DeepSORT's appearance extractor).

        Returns
        -------
        List[Track]
        """
        # Convert to DeepSORT's expected format: ([x,y,w,h], conf, class)
        ds_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w, h = x2 - x1, y2 - y1
            ds_dets.append(([x1, y1, w, h], det.confidence, det.class_name))

        raw_tracks = self._tracker.update_tracks(ds_dets, frame=frame)

        tracks: List[Track] = []
        for rt in raw_tracks:
            if not rt.is_confirmed():
                continue

            # Keep the latest class label for this id
            det_class = getattr(rt, "det_class", None)
            if det_class is not None:
                self._id_to_class[rt.track_id] = det_class

            class_name = self._id_to_class.get(rt.track_id, "surgical_tool")
            confidence = float(getattr(rt, "det_conf", None) or 0.0)

            ltrb = rt.to_ltrb()
            bbox = ltrb.tolist() if hasattr(ltrb, "tolist") else list(ltrb)

            # Guard against NaN / Inf from the tracker
            if not all(np.isfinite(v) for v in bbox):
                continue

            tracks.append(Track(
                track_id   = rt.track_id,
                bbox       = bbox,
                class_name = class_name,
                confidence = confidence,
            ))

        return tracks
