"""FACELESS live webcam pipeline.

Optimized architecture for real-time face swapping:

Thread 1 — CAPTURE:  cv2.VideoCapture.read() → capture_queue
Thread 2 — DETECT:   Face detection on latest frame (decoupled, runs at own rate)
Thread 3 — PROCESS:  Face swap + post-processing using cached detection results
Thread 4 — DISPLAY:  Minimal-transform render to window

Key optimizations over original:
  - Source face embedding cached ONCE at selection (not re-detected each frame)
  - Face detection decoupled from processing (stale-but-fast)
  - Single memory copy path (no redundant .copy() calls)
  - Display uses cv2.imshow (zero PIL/Tk conversion overhead) OR optimized CTk path
  - Quality presets control resolution/detection/enhancement tradeoffs
  - VRAM budget: Normal ~6-8GB, High ~10-15GB (leaves room for OBS + games)
"""

import threading
import queue
import time
from typing import Optional, Callable, Any, List

import cv2
import numpy as np

import modules.globals
from modules.gpu_processing import gpu_flip, gpu_resize, gpu_cvt_color, GpuProcessor
from modules.face_analyser import get_one_face, get_many_faces
from modules.typing import Face, Frame


class LivePipeline:
    """High-performance live face-swap pipeline."""

    def __init__(self):
        # Queues (small maxsize = drop stale frames, reduce latency)
        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)

        # Shared state (protected by lock)
        self._detection_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._detected_faces: Optional[List[Face]] = None
        self._detection_fps: float = 0.0

        # Cached source (set once, never re-detected in hot path)
        self._source_face: Optional[Face] = None

        # Control
        self._stop_event = threading.Event()
        self._threads: List[threading.Thread] = []

        # Stats
        self._process_fps: float = 0.0
        self._capture_fps: float = 0.0
        self._frame_count: int = 0

        # GPU processor for batched ops
        self._gpu = GpuProcessor()

        # Frame processors (set externally)
        self._frame_processors: List[Any] = []

    @property
    def process_fps(self) -> float:
        return self._process_fps

    @property
    def detection_fps(self) -> float:
        return self._detection_fps

    @property
    def capture_fps(self) -> float:
        return self._capture_fps

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def get_stats(self) -> dict:
        """Return pipeline stats for WebSocket clients."""
        return {
            "process_fps": round(self._process_fps, 1),
            "detect_fps": round(self._detection_fps, 1),
            "capture_fps": round(self._capture_fps, 1),
            "frame_count": self._frame_count,
            "is_running": self.is_running,
        }

    def set_source_face(self, face: Optional[Face]):
        """Cache the source face embedding. Called once when user selects source image."""
        self._source_face = face
        # Also cache in globals for other modules
        modules.globals.cached_source_face = face

    def set_frame_processors(self, processors: List[Any]):
        self._frame_processors = processors

    def get_processed_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """Get the latest processed frame (for display thread)."""
        try:
            return self._processed_queue.get_nowait()
        except queue.Empty:
            return None

    def start(self, camera_index: int = 0):
        """Start all pipeline threads."""
        self._stop_event.clear()

        cfg = modules.globals.get_preset_config()
        width = cfg["capture_width"]
        height = cfg["capture_height"]
        fps = cfg["capture_fps"]

        # Capture thread
        t_cap = threading.Thread(
            target=self._capture_loop,
            args=(camera_index, width, height, fps),
            daemon=True,
            name="phantom-capture",
        )

        # Detection thread
        t_det = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="phantom-detect",
        )

        # Processing thread
        t_proc = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="phantom-process",
        )

        self._threads = [t_cap, t_det, t_proc]
        for t in self._threads:
            t.start()

    def stop(self):
        """Stop all pipeline threads."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads.clear()

    # ── Capture Thread ───────────────────────────────────────────────────────

    def _capture_loop(self, camera_index: int, width: int, height: int, fps: int):
        """Read frames from webcam as fast as possible."""
        import platform

        cap = None
        if platform.system() == "Windows":
            for dev_id, backend in [(camera_index, cv2.CAP_DSHOW), (camera_index, cv2.CAP_ANY), (0, cv2.CAP_ANY)]:
                try:
                    cap = cv2.VideoCapture(dev_id, backend)
                    if cap.isOpened():
                        break
                    cap.release()
                    cap = None
                except Exception:
                    continue
        else:
            cap = cv2.VideoCapture(camera_index)

        if not cap or not cap.isOpened():
            print("[FACELESS] Failed to open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        # Try to enable MJPEG for higher FPS on USB cameras
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[FACELESS] Camera: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

        frame_count = 0
        t0 = time.time()

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            # Mirror if enabled
            if modules.globals.live_mirror:
                frame = cv2.flip(frame, 1)

            # Drop old frames (keep only latest)
            try:
                self._capture_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._capture_queue.get_nowait()
                except queue.Empty:
                    pass
                self._capture_queue.put_nowait(frame)

            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                self._capture_fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()

        cap.release()

    # ── Detection Thread ─────────────────────────────────────────────────────

    def _detection_loop(self):
        """Run face detection at its own rate, decoupled from processing.

        The processing thread reads cached (possibly 1-2 frames stale) results.
        This is the right tradeoff: detection is the slowest step (15-30ms GPU),
        and stale positions still produce good swaps.
        """
        det_count = 0
        t0 = time.time()

        while not self._stop_event.is_set():
            # Get latest frame
            with self._detection_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.01)
                continue

            # Run detection
            if modules.globals.many_faces:
                faces = get_many_faces(frame)
            else:
                face = get_one_face(frame)
                faces = [face] if face else None

            # Publish results
            with self._detection_lock:
                self._detected_faces = faces

            det_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                self._detection_fps = det_count / elapsed
                det_count = 0
                t0 = time.time()

    # ── Processing Thread ────────────────────────────────────────────────────

    def _processing_loop(self):
        """Main processing loop: read captured frame, apply face swap, push to display.

        Performance optimization: face swap runs at a lower resolution (PROCESS_RES)
        then the result is upscaled back to the original capture resolution. This is
        because InsightFace's CPU-side warp/paste scales with frame size:
          1080p = 98ms, 540p = 27ms, but the model itself is 128x128 either way.
        """
        from modules.processors.frame.face_swapper import swap_face, apply_post_processing
        from modules.processors.frame.face_enhancer_gpen256 import enhance_face as gpen256_enhance
        from insightface.app.common import Face

        PROCESS_W, PROCESS_H = 854, 480  # 480p swap = ~31fps (540p=27fps, 1080p=10fps)

        frame_count = 0
        t0 = time.time()

        while not self._stop_event.is_set():
            # Get latest captured frame (may be 1080p from camera)
            try:
                frame = self._capture_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            orig_h, orig_w = frame.shape[:2]
            needs_upscale = (orig_w > PROCESS_W)

            # Publish for detection thread (use original res for better detection)
            with self._detection_lock:
                self._latest_frame = frame
                cached_faces = self._detected_faces

            # Skip processing if no source face set
            if self._source_face is None:
                self._push_processed(frame)
                continue

            # Skip if no target faces detected yet
            if not cached_faces:
                self._push_processed(frame)
                continue

            # Downscale for face swap processing
            if needs_upscale:
                work_frame = cv2.resize(frame, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_AREA)
                # Scale detection bboxes/kps to match work_frame
                sx = PROCESS_W / orig_w
                sy = PROCESS_H / orig_h
                scaled_faces = []
                for f in cached_faces:
                    if f is None:
                        continue
                    # Face is a dict subclass — dict copy + scale numpy arrays
                    sf = Face(f)
                    if hasattr(f, 'bbox') and f.bbox is not None:
                        sf.bbox = f.bbox.copy() * np.array([sx, sy, sx, sy])
                    if hasattr(f, 'kps') and f.kps is not None:
                        sf.kps = f.kps.copy() * np.array([sx, sy])
                    scaled_faces.append(sf)
            else:
                work_frame = frame
                scaled_faces = [f for f in cached_faces if f is not None]

            # Apply face swap(s) at process resolution
            result = work_frame
            swapped_bboxes = []

            for target_face in scaled_faces:
                result = swap_face(self._source_face, target_face, result)
                if hasattr(target_face, "bbox") and target_face.bbox is not None:
                    swapped_bboxes.append(target_face.bbox.astype(int))

            # Post-processing (sharpening, interpolation)
            result = apply_post_processing(result, swapped_bboxes)

            # GPEN-256 face enhancement (runs at process resolution, 256x256 internally)
            if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                for target_face in scaled_faces:
                    result = gpen256_enhance(result, target_face)

            # Apply any additional frame processors (enhancers etc.)
            for processor in self._frame_processors:
                if hasattr(processor, 'process_frame'):
                    result = processor.process_frame(None, result)

            # Upscale back to original resolution for output
            if needs_upscale:
                result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

            # FPS overlay
            if modules.globals.show_fps:
                fps_text = f"FPS: {self._process_fps:.0f} | Det: {self._detection_fps:.0f}"
                cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self._push_processed(result)

            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                self._process_fps = frame_count / elapsed
                self._frame_count += frame_count
                frame_count = 0
                t0 = time.time()

    def _push_processed(self, frame: np.ndarray):
        """Push to processed queue, dropping stale frames."""
        try:
            self._processed_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._processed_queue.get_nowait()
            except queue.Empty:
                pass
            self._processed_queue.put_nowait(frame)


# Global singleton
_pipeline: Optional[LivePipeline] = None


def get_pipeline() -> LivePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LivePipeline()
    return _pipeline
