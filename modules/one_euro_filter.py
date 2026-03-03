"""1-Euro Filter for face landmark stabilization.

Adaptive low-pass filter: heavy smoothing when landmarks are still (removes jitter),
light smoothing when moving fast (preserves responsiveness). ~0.5ms per frame.

Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for
Noisy Input in Interactive Systems", CHI 2012.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple

from modules.typing import Face


class LowPassFilter:
    """Exponentially weighted moving average."""

    __slots__ = ('_y', '_s', '_initialized')

    def __init__(self):
        self._y: float = 0.0
        self._s: float = 0.0
        self._initialized = False

    def filter(self, value: float, alpha: float) -> float:
        if not self._initialized:
            self._s = value
            self._initialized = True
        else:
            self._s = alpha * value + (1.0 - alpha) * self._s
        self._y = value
        return self._s

    @property
    def last_raw(self) -> float:
        return self._y

    def reset(self):
        self._initialized = False


class OneEuroFilter:
    """Adaptive low-pass filter — heavy smoothing when still, light when moving.

    Args:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing when still.
        beta: Speed coefficient. Higher = less lag when moving fast.
        d_cutoff: Cutoff for derivative filter (Hz). Usually 1.0.
    """

    __slots__ = ('_min_cutoff', '_beta', '_d_cutoff', '_x_filter', '_dx_filter', '_last_time')

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.007, d_cutoff: float = 1.0):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value: float, timestamp: float) -> float:
        if self._last_time is None:
            self._last_time = timestamp
            # First sample — no filtering possible
            self._x_filter.filter(value, 1.0)
            self._dx_filter.filter(0.0, 1.0)
            return value

        dt = timestamp - self._last_time
        if dt <= 0:
            dt = 1e-6  # Avoid division by zero
        self._last_time = timestamp

        # Estimate speed (derivative)
        dx = (value - self._x_filter.last_raw) / dt
        edx = self._dx_filter.filter(dx, self._alpha(self._d_cutoff, dt))

        # Adaptive cutoff: increase when moving fast
        cutoff = self._min_cutoff + self._beta * abs(edx)

        return self._x_filter.filter(value, self._alpha(cutoff, dt))

    def reset(self):
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None


class FaceStabilizer:
    """Manages per-face 1-Euro filter banks, tracking faces across frames via bbox IoU.

    Each tracked face gets 14 filters:
      - 4 for bbox (x1, y1, x2, y2)
      - 10 for kps (5 landmarks × 2 coords)

    Faces are matched across frames by IoU overlap (threshold 0.3).
    """

    def __init__(self, iou_threshold: float = 0.3,
                 min_cutoff: float = 0.8, beta: float = 0.007,
                 max_stale_frames: int = 10):
        self._iou_threshold = iou_threshold
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._max_stale_frames = max_stale_frames
        self._tracked: List[_TrackedFace] = []

    def update(self, faces: List[Face], timestamp: float) -> List[Face]:
        """Stabilize a list of detected faces. Returns faces with smoothed bbox/kps."""
        if not faces:
            # Age out tracked faces
            self._tracked = [t for t in self._tracked if t.age < self._max_stale_frames]
            for t in self._tracked:
                t.age += 1
            return faces

        # Match incoming faces to tracked faces via IoU
        matched_incoming = set()
        matched_tracked = set()

        # Build IoU matrix
        for i, face in enumerate(faces):
            if not hasattr(face, 'bbox') or face.bbox is None:
                continue
            best_iou = 0.0
            best_j = -1
            for j, tracked in enumerate(self._tracked):
                if j in matched_tracked:
                    continue
                iou = self._compute_iou(face.bbox, tracked.last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= self._iou_threshold and best_j >= 0:
                matched_incoming.add(i)
                matched_tracked.add(best_j)
                # Update tracked face with smoothed values
                self._tracked[best_j].smooth(face, timestamp)
                self._tracked[best_j].age = 0

        # Create new tracked faces for unmatched detections
        for i, face in enumerate(faces):
            if i not in matched_incoming:
                tf = _TrackedFace(self._min_cutoff, self._beta)
                tf.smooth(face, timestamp)
                self._tracked.append(tf)

        # Prune stale tracked faces
        self._tracked = [
            t for j, t in enumerate(self._tracked)
            if j in matched_tracked or t.age == 0
        ]

        return faces

    def reset(self):
        """Clear all tracked faces."""
        self._tracked.clear()

    @staticmethod
    def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
        """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


class _TrackedFace:
    """Internal: filter bank for a single tracked face."""

    __slots__ = ('bbox_filters', 'kps_filters', 'last_bbox', 'age')

    def __init__(self, min_cutoff: float, beta: float):
        # 4 filters for bbox
        self.bbox_filters = [OneEuroFilter(min_cutoff, beta) for _ in range(4)]
        # 10 filters for kps (5 points × 2 coords)
        self.kps_filters = [OneEuroFilter(min_cutoff, beta) for _ in range(10)]
        self.last_bbox = np.zeros(4, dtype=np.float32)
        self.age = 0

    def smooth(self, face: Face, timestamp: float):
        """Apply 1-Euro filtering to face bbox and kps in-place."""
        if hasattr(face, 'bbox') and face.bbox is not None:
            for i in range(4):
                face.bbox[i] = self.bbox_filters[i].filter(float(face.bbox[i]), timestamp)
            self.last_bbox = face.bbox.copy()

        if hasattr(face, 'kps') and face.kps is not None:
            for i in range(5):
                face.kps[i, 0] = self.kps_filters[i * 2].filter(float(face.kps[i, 0]), timestamp)
                face.kps[i, 1] = self.kps_filters[i * 2 + 1].filter(float(face.kps[i, 1]), timestamp)
