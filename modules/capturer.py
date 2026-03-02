"""PHANTOM-FACE video capturer with connection pooling.

Original opened/closed cv2.VideoCapture for EVERY frame seek (slider preview).
We now cache the capture object per video path and reuse it.
"""

from typing import Any, Optional
import cv2
import modules.globals
from modules.gpu_processing import gpu_cvt_color

# Connection pool: one VideoCapture per video path
_capture_pool: dict[str, cv2.VideoCapture] = {}


def _get_capture(video_path: str) -> cv2.VideoCapture:
    """Get or create a cached VideoCapture for the given path."""
    if video_path not in _capture_pool:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if modules.globals.color_correction:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        _capture_pool[video_path] = cap
    return _capture_pool[video_path]


def release_all():
    """Release all cached captures (call on cleanup)."""
    for cap in _capture_pool.values():
        try:
            cap.release()
        except Exception:
            pass
    _capture_pool.clear()


def get_video_frame(video_path: str, frame_number: int = 0) -> Any:
    """Get a single frame — uses cached VideoCapture (no open/close per call)."""
    capture = _get_capture(video_path)

    frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
    has_frame, frame = capture.read()

    if has_frame and modules.globals.color_correction:
        frame = gpu_cvt_color(frame, cv2.COLOR_BGR2RGB)

    return frame if has_frame else None


def get_video_frame_total(video_path: str) -> int:
    capture = _get_capture(video_path)
    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
