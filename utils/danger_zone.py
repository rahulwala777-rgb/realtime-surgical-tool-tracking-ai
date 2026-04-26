"""
utils/danger_zone.py
─────────────────────
Defines rectangular danger zones and checks whether tracked surgical tools
have entered them.

Zone coordinates come from config/config.yaml and are expressed in absolute
pixel coordinates matching the video resolution.

Overlap logic
─────────────
When overlap_threshold = 0.0 (default), ANY pixel intersection triggers an
alert.  Set it to e.g. 0.3 to require that at least 30 % of the tool's
bounding-box area is inside the zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from tracking.tracker import Track


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DangerZone:
    """A single rectangular region that surgical tools must not enter."""
    name:  str
    x1:    int
    y1:    int
    x2:    int
    y2:    int
    color: Tuple[int, int, int] = (0, 0, 255)   # BGR

    # ------------------------------------------------------------------

    def intersects(self, bbox: List[float], threshold: float = 0.0) -> bool:
        """
        Return True when the bounding box overlaps this zone.

        Parameters
        ----------
        bbox      : [x1, y1, x2, y2]
        threshold : minimum (intersection / bbox_area) ratio required.
                    0.0 means any overlap triggers.
        """
        bx1, by1, bx2, by2 = bbox

        ix1 = max(self.x1, bx1)
        iy1 = max(self.y1, by1)
        ix2 = min(self.x2, bx2)
        iy2 = min(self.y2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return False                    # No overlap at all

        if threshold == 0.0:
            return True                     # Any overlap is enough

        intersection = (ix2 - ix1) * (iy2 - iy1)
        bbox_area    = max((bx2 - bx1) * (by2 - by1), 1e-6)
        return (intersection / bbox_area) >= threshold


@dataclass
class Alert:
    """One tool–zone collision event."""
    track_id:  int
    zone_name: str
    bbox:      List[float]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class DangerZoneManager:
    """
    Loads zone definitions from config and evaluates collisions each frame.

    Parameters
    ----------
    config : dict
        Parsed contents of config/config.yaml.
    """

    def __init__(self, config: dict) -> None:
        self.zones: List[DangerZone] = []
        self._threshold: float = float(
            config.get("alerts", {}).get("overlap_threshold", 0.0)
        )
        self._load_zones(config.get("danger_zones", []))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_alerts(self, tracks: List[Track]) -> List[Alert]:
        """
        Check every track against every zone.

        Returns a list of Alert objects — one per (track, zone) collision.
        The same track can appear multiple times if it overlaps several zones.
        """
        alerts: List[Alert] = []
        for track in tracks:
            for zone in self.zones:
                if zone.intersects(track.bbox, self._threshold):
                    alerts.append(Alert(
                        track_id  = track.track_id,
                        zone_name = zone.name,
                        bbox      = track.bbox,
                    ))
        return alerts

    def add_zone(
        self,
        name:  str,
        x1:    int,
        y1:    int,
        x2:    int,
        y2:    int,
        color: Tuple[int, int, int] = (0, 0, 255),
    ) -> DangerZone:
        """Dynamically add a zone at runtime (e.g. from a UI callback)."""
        zone = DangerZone(name=name, x1=x1, y1=y1, x2=x2, y2=y2, color=color)
        self.zones.append(zone)
        print(f"[DangerZone] Added zone '{name}' at ({x1},{y1})→({x2},{y2})")
        return zone

    def remove_zone(self, name: str) -> bool:
        """Remove a zone by name.  Returns True if found and removed."""
        before = len(self.zones)
        self.zones = [z for z in self.zones if z.name != name]
        return len(self.zones) < before

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_zones(self, zone_configs: list) -> None:
        for zc in zone_configs:
            coords = zc["coordinates"]
            raw_color = zc.get("color", [0, 0, 255])
            color = tuple(int(c) for c in raw_color)

            zone = DangerZone(
                name  = zc["name"],
                x1    = int(coords[0]),
                y1    = int(coords[1]),
                x2    = int(coords[2]),
                y2    = int(coords[3]),
                color = color,         # type: ignore[arg-type]
            )
            self.zones.append(zone)
            print(f"[DangerZone] Loaded '{zone.name}'  coords={coords}")
