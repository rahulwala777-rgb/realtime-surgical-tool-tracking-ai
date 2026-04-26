"""
detection/detector.py
─────────────────────
Wraps YOLOv8 (Ultralytics) to detect objects in a video frame.

In *simulate mode* (config: surgical_tools.simulate = true), every detected
object is relabelled as a surgical tool so the system works out-of-the-box
with a generic COCO-trained model — no custom surgical dataset required.

When simulate is false, only COCO class IDs listed in config are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single detected object in one frame."""
    bbox: List[float]       # [x1, y1, x2, y2]  absolute pixel coordinates
    confidence: float
    class_id: int
    class_name: str         # human-readable label (may be relabelled in sim mode)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class SurgicalToolDetector:
    """
    Loads a YOLOv8 model and runs per-frame inference.

    Parameters
    ----------
    config : dict
        Parsed contents of config/config.yaml.
    """

    # Fallback mapping: COCO IDs → surgical-instrument proxy names
    _COCO_PROXY: dict[int, str] = {
        76: "scissors",
        39: "bottle",
        41: "cup",
        67: "cell-phone",
        84: "book",
        73: "laptop",
    }

    def __init__(self, config: dict) -> None:
        self._cfg_model   = config["model"]
        self._cfg_tools   = config["surgical_tools"]

        self.model_path   = self._cfg_model["path"]
        self.confidence   = float(self._cfg_model["confidence"])
        self.device       = str(self._cfg_model.get("device", "cpu"))
        self.simulate     = bool(self._cfg_tools.get("simulate", True))
        self.tool_classes = set(self._cfg_tools.get("classes", list(self._COCO_PROXY.keys())))

        print(f"[Detector] Loading model '{self.model_path}' on device='{self.device}' ...")
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        print(f"[Detector] Ready  (conf={self.confidence}, simulate={self.simulate})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO inference on *frame* and return filtered detections.

        Parameters
        ----------
        frame : np.ndarray  BGR image (H×W×3)

        Returns
        -------
        List[Detection]
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id   = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                # Class filter (only active when simulate=False)
                if not self.simulate and class_id not in self.tool_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = self._build_label(class_id)

                detections.append(Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                ))

        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_label(self, class_id: int) -> str:
        """Return a surgical-looking label for the given COCO class id."""
        if self.simulate:
            # Rename everything to look like a surgical instrument
            raw = self.model.names.get(class_id, f"obj{class_id}")
            return f"tool:{raw}"
        return self._COCO_PROXY.get(class_id, f"tool_{class_id}")
