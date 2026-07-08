"""FACELESS WebSocket server.

Wraps LivePipeline for the Electron UI. Runs on 127.0.0.1:7865.

Protocol:
  Client → Server (JSON):
    { "type": "start_preview", "camera_index": 0 }
    { "type": "stop_preview" }
    { "type": "set_source", "path": "C:/path/to/face.jpg" }
    { "type": "control", "key": "many_faces", "value": true }
    { "type": "get_cameras" }
    { "type": "get_state" }

  Server → Client (JSON):
    { "type": "state", "controls": { ... } }
    { "type": "cameras", "list": [...] }
    { "type": "stats", ... }
    { "type": "source_face", "detected": true, "thumbnail": "<base64>" }
    { "type": "status", "message": "..." }
    { "type": "error", "message": "..." }

  Server → Client (Binary):
    Raw JPEG bytes (one frame per message)
"""

import asyncio
import json
import base64
import time
from typing import Optional, Set

import cv2
import numpy as np
import websockets
from websockets.asyncio.server import serve, ServerConnection

import modules.globals
from modules.globals import QualityPreset
from modules.live_pipeline import get_pipeline
from modules.face_analyser import get_one_face

# FFHQ 5-point template (normalized to [0,1], scaled at use time)
_FFHQ_TEMPLATE = np.array([
    [0.31556875, 0.4615741],
    [0.68262291, 0.4615741],
    [0.50009375, 0.6405054],
    [0.34947187, 0.8246919],
    [0.65343645, 0.8246919],
], dtype=np.float32)

try:
    import pyvirtualcam
    HAS_VCAM = True
except ImportError:
    HAS_VCAM = False


# ── Camera Enumeration ─────────────────────────────────────────────────────

def enumerate_cameras(max_index: int = 8) -> list:
    """Enumerate available cameras via direct probe. Returns list of {index, name}."""
    cameras = []
    # Probe each index with CAP_ANY (most reliable on Windows)
    # Skip index 0 initially — probe it last to avoid DSHOW C++ exceptions
    # that can poison subsequent probes
    indices = list(range(1, max_index)) + [0]
    for i in indices:
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({"index": i, "name": f"Camera {i} ({w}x{h})"})
                cap.release()
            else:
                cap.release()
        except Exception:
            pass
    # Sort by index for consistent ordering
    cameras.sort(key=lambda c: c["index"])
    return cameras


# ── State Snapshot ──────────────────────────────────────────────────────────

def get_control_state() -> dict:
    """Snapshot of all control values for the UI."""
    return {
        "many_faces": modules.globals.many_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "mouth_mask": modules.globals.mouth_mask,
        "color_correction": modules.globals.color_correction,
        "face_enhancer": modules.globals.fp_ui.get("face_enhancer", False),
        "face_enhancer_gpen256": modules.globals.fp_ui.get("face_enhancer_gpen256", False),
        "face_enhancer_gpen512": modules.globals.fp_ui.get("face_enhancer_gpen512", False),
        "live_mirror": modules.globals.live_mirror,
        "opacity": modules.globals.opacity,
        "sharpness": modules.globals.sharpness,
        "quality_preset": modules.globals.quality_preset.value,
        "source_path": modules.globals.source_path,
        "show_fps": modules.globals.show_fps,
        "virtual_camera": False,
    }


# ── Control Dispatch ────────────────────────────────────────────────────────

BOOL_CONTROLS = {
    "many_faces": "many_faces",
    "poisson_blend": "poisson_blend",
    "mouth_mask": "mouth_mask",
    "color_correction": "color_correction",
    "live_mirror": "live_mirror",
    "show_fps": "show_fps",
}

FP_UI_CONTROLS = {
    "face_enhancer",
    "face_enhancer_gpen256",
    "face_enhancer_gpen512",
}

FLOAT_CONTROLS = {
    "opacity": "opacity",
    "sharpness": "sharpness",
}


def apply_control(key: str, value) -> None:
    """Apply a single control change to globals."""
    if key in BOOL_CONTROLS:
        setattr(modules.globals, BOOL_CONTROLS[key], bool(value))
    elif key in FP_UI_CONTROLS:
        modules.globals.fp_ui[key] = bool(value)
        # Preload model on toggle-on so first frame isn't laggy
        if bool(value):
            import threading
            if key == "face_enhancer_gpen256":
                try:
                    from modules.processors.frame.face_enhancer_gpen256 import get_enhancer
                    threading.Thread(target=get_enhancer, daemon=True).start()
                except Exception as e:
                    print(f"[FACELESS] GPEN-256 preload error: {e}")
            elif key == "face_enhancer_gpen512":
                try:
                    from modules.processors.frame.face_enhancer_gpen512 import get_enhancer
                    threading.Thread(target=get_enhancer, daemon=True).start()
                except Exception as e:
                    print(f"[FACELESS] GPEN-512 preload error: {e}")
            elif key == "face_enhancer":
                try:
                    from modules.processors.frame.face_enhancer import get_face_enhancer
                    threading.Thread(target=get_face_enhancer, daemon=True).start()
                except Exception as e:
                    print(f"[FACELESS] GFPGAN preload error: {e}")
    elif key in FLOAT_CONTROLS:
        setattr(modules.globals, FLOAT_CONTROLS[key], float(value))
    elif key == "quality_preset":
        preset = QualityPreset.HIGH if value == "high" else QualityPreset.NORMAL
        modules.globals.apply_preset(preset)


# ── Frame Encoding ──────────────────────────────────────────────────────────

def encode_frame_jpeg(frame: np.ndarray, quality: int = 65, max_width: int = 960) -> bytes:
    """Encode BGR frame to JPEG bytes, downscaling if needed."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        return buf.tobytes()
    return b''


def make_thumbnail(frame: np.ndarray, max_size: int = 128) -> str:
    """Create a base64-encoded JPEG thumbnail."""
    h, w = frame.shape[:2]
    scale = min(max_size / w, max_size / h)
    thumb = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode('ascii')


# ── WebSocket Server ────────────────────────────────────────────────────────

class PhantomServer:
    """Async WebSocket server for FACELESS."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7865):
        self.host = host
        self.port = port
        self.clients: Set[ServerConnection] = set()
        self._streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._pipeline = get_pipeline()
        self._vcam: Optional[object] = None
        self._vcam_enabled = False
        self._vcam_thread = None
        self._vcam_latest_frame: Optional[np.ndarray] = None
        self._vcam_err_logged = False

    async def handler(self, ws: ServerConnection) -> None:
        """Handle a single WebSocket connection."""
        self.clients.add(ws)
        print(f"[FACELESS] Client connected ({len(self.clients)} total)")

        try:
            # Send initial state
            await ws.send(json.dumps({
                "type": "state",
                "controls": get_control_state(),
            }))

            async for message in ws:
                if isinstance(message, str):
                    await self._handle_json(ws, json.loads(message))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            print(f"[FACELESS] Client disconnected ({len(self.clients)} total)")

            # Stop streaming if no clients
            if not self.clients:
                await self._stop_streaming()

    async def _handle_json(self, ws: ServerConnection, msg: dict) -> None:
        """Dispatch a JSON message from the client."""
        msg_type = msg.get("type", "")
        print(f"[FACELESS] recv {msg_type}")

        if msg_type == "get_state":
            await ws.send(json.dumps({
                "type": "state",
                "controls": get_control_state(),
            }))

        elif msg_type == "get_cameras":
            cameras = await asyncio.to_thread(enumerate_cameras)
            await ws.send(json.dumps({
                "type": "cameras",
                "list": cameras,
            }))

        elif msg_type == "set_source":
            path = msg.get("path", "")
            await self._set_source(ws, path)

        elif msg_type == "add_source":
            path = msg.get("path", "")
            await self._add_source(ws, path)

        elif msg_type == "control":
            key = msg.get("key", "")
            value = msg.get("value")
            apply_control(key, value)
            # Broadcast updated state to all clients
            state_msg = json.dumps({
                "type": "state",
                "controls": get_control_state(),
            })
            await self._broadcast(state_msg)

        elif msg_type == "start_preview":
            camera_index = msg.get("camera_index", 0)
            await self._start_streaming(camera_index)

        elif msg_type == "stop_preview":
            await self._stop_streaming()

        elif msg_type == "toggle_vcam":
            enabled = msg.get("enabled", False)
            await self._toggle_vcam(enabled)

    async def _set_source(self, ws: ServerConnection, path: str) -> None:
        """Load source face image and cache embedding."""
        print(f"[FACELESS] _set_source: path={path}")
        if not path:
            modules.globals.source_path = None
            self._pipeline.set_source_face(None)
            await self._broadcast(json.dumps({
                "type": "source_face",
                "detected": False,
                "thumbnail": None,
            }))
            return

        try:
            img = await asyncio.to_thread(cv2.imread, path)
            if img is None:
                await self._broadcast(json.dumps({
                    "type": "error",
                    "message": f"Failed to read image: {path}",
                }))
                return

            print(f"[FACELESS] Image loaded ({img.shape}), detecting face...")
            face = await asyncio.to_thread(get_one_face, img)
            print(f"[FACELESS] Face detection result: {'found' if face is not None else 'none'}")

            if face is None:
                modules.globals.source_path = path
                self._pipeline.set_source_face(None)
                thumb = make_thumbnail(img)
                print(f"[FACELESS] Sending source_face (no face), thumbnail len={len(thumb)}")
                await self._broadcast(json.dumps({
                    "type": "source_face",
                    "detected": False,
                    "thumbnail": thumb,
                }))
                await self._broadcast(json.dumps({
                    "type": "status",
                    "message": "No face detected in source image",
                }))
                return

            # Preprocess: align-crop to 512×512 and re-detect for cleaner embedding
            preprocessed_face = await asyncio.to_thread(
                self._preprocess_source_face, img, face
            )
            if preprocessed_face is not None:
                face = preprocessed_face

            modules.globals.source_path = path
            self._pipeline.set_source_face(face)
            modules.globals.cached_source_face = face

            # Initialize multi-source embedding list
            modules.globals.source_embeddings = []
            if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                modules.globals.source_embeddings.append(face.normed_embedding.copy())
            modules.globals.averaged_embedding = None

            thumb = make_thumbnail(img)
            print(f"[FACELESS] Sending source_face (detected), thumbnail len={len(thumb)}, clients={len(self.clients)}")
            await self._broadcast(json.dumps({
                "type": "source_face",
                "detected": True,
                "thumbnail": thumb,
            }))
            await self._broadcast(json.dumps({
                "type": "status",
                "message": "Source face loaded",
            }))

        except Exception as e:
            print(f"[FACELESS] _set_source error: {e}")
            await self._broadcast(json.dumps({
                "type": "error",
                "message": str(e),
            }))

    @staticmethod
    def _preprocess_source_face(img: np.ndarray, face) -> object:
        """Align-crop source to 512×512 using FFHQ template, re-detect for better embedding.

        Returns improved Face object, or None if preprocessing fails.
        """
        try:
            if not hasattr(face, 'kps') or face.kps is None:
                return None

            landmarks = face.kps.astype(np.float32)
            template = _FFHQ_TEMPLATE * 512.0

            M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
            if M is None:
                return None

            aligned = cv2.warpAffine(
                img, M, (512, 512),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )

            # Re-detect on the clean aligned crop for better embedding
            clean_face = get_one_face(aligned)
            if clean_face is None:
                return None

            print("[FACELESS] Source preprocessed: aligned 512x512")
            return clean_face

        except Exception as e:
            print(f"[FACELESS] Source preprocessing failed: {e}")
            return None

    async def _add_source(self, ws: ServerConnection, path: str) -> None:
        """Add an additional source face image, average embeddings for multi-angle quality."""
        if not path:
            return

        try:
            img = await asyncio.to_thread(cv2.imread, path)
            if img is None:
                await self._broadcast(json.dumps({
                    "type": "error",
                    "message": f"Failed to read image: {path}",
                }))
                return

            face = await asyncio.to_thread(get_one_face, img)
            if face is None:
                await self._broadcast(json.dumps({
                    "type": "status",
                    "message": "No face detected in additional source image",
                }))
                return

            # Preprocess for cleaner embedding
            preprocessed = await asyncio.to_thread(
                self._preprocess_source_face, img, face
            )
            if preprocessed is not None:
                face = preprocessed

            if not hasattr(face, 'normed_embedding') or face.normed_embedding is None:
                await self._broadcast(json.dumps({
                    "type": "error",
                    "message": "Could not extract embedding from additional source",
                }))
                return

            # Accumulate embedding
            modules.globals.source_embeddings.append(face.normed_embedding.copy())

            # Compute L2-normalized average across all source embeddings
            stacked = np.stack(modules.globals.source_embeddings, axis=0)
            avg = stacked.mean(axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            modules.globals.averaged_embedding = avg

            # Update the cached source face with the averaged embedding
            cached = modules.globals.cached_source_face
            if cached is not None:
                cached.normed_embedding = avg

            n = len(modules.globals.source_embeddings)
            print(f"[FACELESS] Source embedding averaged from {n} images")
            await self._broadcast(json.dumps({
                "type": "status",
                "message": f"Source averaged from {n} images",
            }))

        except Exception as e:
            print(f"[FACELESS] _add_source error: {e}")
            await self._broadcast(json.dumps({
                "type": "error",
                "message": str(e),
            }))

    async def _start_streaming(self, camera_index: int) -> None:
        """Start the pipeline and begin streaming frames."""
        if self._streaming:
            return

        # Validate camera before starting pipeline
        def check_camera(idx: int) -> bool:
            for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        cap.release()
                        return True
                    cap.release()
                except Exception:
                    pass
            return False

        camera_ok = await asyncio.to_thread(check_camera, camera_index)
        if not camera_ok:
            print(f"[FACELESS] Camera {camera_index} not available")
            await self._broadcast(json.dumps({
                "type": "error",
                "message": f"Camera {camera_index} not available. Is it plugged in?",
            }))
            return

        self._streaming = True

        # Start pipeline in thread (blocking camera init)
        await asyncio.to_thread(self._pipeline.start, camera_index)

        await self._broadcast(json.dumps({
            "type": "status",
            "message": "Pipeline started",
        }))

        # Launch frame streaming coroutine
        loop = asyncio.get_running_loop()
        self._stream_task = loop.create_task(self._frame_loop())
        self._stats_task = loop.create_task(self._stats_loop())

    async def _stop_streaming(self) -> None:
        """Stop the pipeline and frame streaming."""
        if not self._streaming:
            return

        self._streaming = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass
            self._stats_task = None

        await asyncio.to_thread(self._pipeline.stop)

        # Close virtual camera if active
        if self._vcam_enabled and self._vcam:
            try:
                self._vcam.close()
            except Exception:
                pass
            self._vcam = None
            self._vcam_enabled = False

        await self._broadcast(json.dumps({
            "type": "status",
            "message": "Pipeline stopped",
        }))

    async def _toggle_vcam(self, enabled: bool) -> None:
        """Toggle virtual camera output."""
        import threading

        if enabled and not self._vcam_enabled:
            if not HAS_VCAM:
                await self._broadcast(json.dumps({
                    "type": "error",
                    "message": "pyvirtualcam not installed. Run: pip install pyvirtualcam",
                }))
                return
            try:
                self._vcam = pyvirtualcam.Camera(
                    width=1920, height=1080, fps=30, print_fps=False,
                    fmt=pyvirtualcam.PixelFormat.BGR,  # match pipeline output; no per-frame convert
                )
                self._vcam_enabled = True
                self._vcam_err_logged = False
                # Start dedicated vcam output thread for consistent timing
                self._vcam_thread = threading.Thread(
                    target=self._vcam_thread_loop, daemon=True, name="faceless-vcam"
                )
                self._vcam_thread.start()
                print(f"[FACELESS] Virtual camera started: {self._vcam.device}")
                await self._broadcast(json.dumps({
                    "type": "status",
                    "message": f"Virtual camera active: {self._vcam.device}",
                }))
            except Exception as e:
                print(f"[FACELESS] Virtual camera error: {e}")
                await self._broadcast(json.dumps({
                    "type": "error",
                    "message": f"Virtual camera failed: {e}. Install OBS for the driver.",
                }))
        elif not enabled and self._vcam_enabled:
            self._vcam_enabled = False
            # Thread will exit on its own since _vcam_enabled is False
            if hasattr(self, '_vcam_thread') and self._vcam_thread:
                self._vcam_thread.join(timeout=2.0)
                self._vcam_thread = None
            if self._vcam:
                try:
                    self._vcam.close()
                except Exception:
                    pass
                self._vcam = None
            print("[FACELESS] Virtual camera stopped")
            await self._broadcast(json.dumps({
                "type": "status",
                "message": "Virtual camera stopped",
            }))

    def _vcam_thread_loop(self) -> None:
        """Dedicated thread for virtual camera output — consistent 30fps timing.

        Reads from self._vcam_latest_frame (set by the frame loop) so we don't
        compete with WebSocket delivery for the processed queue.
        """
        while self._vcam_enabled and self._vcam:
            frame = self._vcam_latest_frame
            if frame is not None:
                try:
                    h, w = frame.shape[:2]
                    if w != self._vcam.width or h != self._vcam.height:
                        vcam_frame = cv2.resize(frame, (self._vcam.width, self._vcam.height))
                    else:
                        vcam_frame = frame
                    self._vcam.send(vcam_frame)  # camera is BGR; no conversion needed
                    self._vcam.sleep_until_next_frame()
                except Exception as e:
                    if not getattr(self, '_vcam_err_logged', False):
                        print(f"[FACELESS] Virtual camera send error: {e}")
                        self._vcam_err_logged = True
                    break
            else:
                time.sleep(0.01)

    async def _frame_loop(self) -> None:
        """Continuously grab processed frames and send as binary WebSocket messages."""
        target_interval = 1.0 / 30  # Cap at 30fps send rate
        last_send = 0.0

        while self._streaming:
            # Pace FIRST, then pull the freshest frame. Pulling before the sleep
            # would hold a frame for up to a full interval while the pipeline drops
            # newer ones, adding a frame of latency to every send.
            now = asyncio.get_event_loop().time()
            elapsed = now - last_send
            if elapsed < target_interval:
                await asyncio.sleep(target_interval - elapsed)

            frame = self._pipeline.get_processed_frame()
            if frame is None:
                await asyncio.sleep(0.002)
                continue

            # Share frame with vcam thread
            self._vcam_latest_frame = frame

            # Encode for WebSocket preview (downscaled)
            jpeg = await asyncio.to_thread(encode_frame_jpeg, frame)
            if jpeg:
                websockets.broadcast(self.clients, jpeg)
                last_send = asyncio.get_event_loop().time()

    async def _stats_loop(self) -> None:
        """Send FPS stats every 500ms."""
        while self._streaming:
            await asyncio.sleep(0.5)
            stats = self._pipeline.get_stats()
            await self._broadcast(json.dumps({
                "type": "stats",
                **stats,
            }))

    async def _broadcast(self, message: str) -> None:
        """Send a text message to all connected clients."""
        if not self.clients:
            return
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True,
        )

    async def run(self) -> None:
        """Start the WebSocket server."""
        print(f"[FACELESS] WebSocket server starting on ws://{self.host}:{self.port}")

        async with serve(self.handler, self.host, self.port) as server:
            print(f"[FACELESS] Server ready — ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever


def main() -> None:
    """Entry point for --server mode."""
    server = PhantomServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[FACELESS] Server stopped")


if __name__ == "__main__":
    main()
