"""PHANTOM-FACE global configuration.

Quality presets:
  NORMAL  — ~6-8 GB VRAM, balanced FPS/quality (good for streaming + gaming)
  HIGH    — ~10-15 GB VRAM, maximum quality (dedicated face-swap sessions)
"""

import os
from typing import List, Dict, Any
from enum import Enum

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

# ── Quality Presets ──────────────────────────────────────────────────────────

class QualityPreset(Enum):
    NORMAL = "normal"
    HIGH = "high"

# Active preset (can be changed at runtime via UI or CLI)
quality_preset: QualityPreset = QualityPreset.NORMAL

# Preset-specific defaults (overridden by preset selection)
PRESET_CONFIGS = {
    QualityPreset.NORMAL: {
        "det_size": (320, 320),          # Face detection input resolution
        "capture_width": 1920,           # Full HD capture (face model stays 320x320)
        "capture_height": 1080,          # Full HD capture
        "capture_fps": 30,              # 30fps target (more stable than 60 at 1080p)
        "enhancer_model": None,          # No enhancer in normal mode
        "max_vram_gb": 8,                # VRAM budget
        "process_workers": 4,            # ProcessPoolExecutor workers for video
        "use_fp16": True,                # FP16 inference
        "face_swap_model": "inswapper_128_fp16.onnx",
        "poisson_blend": False,          # Too slow for real-time
        "sharpness": 0.3,               # Mild sharpening
        "interpolation_weight": 0.0,     # No temporal smoothing (lowest latency)
    },
    QualityPreset.HIGH: {
        "det_size": (640, 640),          # Higher-res face detection
        "capture_width": 1920,           # Full HD capture
        "capture_height": 1080,          # Full HD capture
        "capture_fps": 30,              # Match camera actual rate
        "enhancer_model": "gfpgan-1024.onnx",
        "max_vram_gb": 15,              # More VRAM budget
        "process_workers": 8,            # More workers
        "use_fp16": True,                # Still FP16 (plenty of quality)
        "face_swap_model": "inswapper_128_fp16.onnx",
        "poisson_blend": False,          # Too slow for real-time (~30ms on CPU)
        "sharpness": 0.5,               # More sharpening
        "interpolation_weight": 0.05,    # Very mild temporal smoothing (was 0.15 = tracers)
    },
}

def get_preset_config() -> dict:
    """Get the configuration dict for the active quality preset."""
    return PRESET_CONFIGS[quality_preset]

# ── File types ───────────────────────────────────────────────────────────────

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm")),
]

# ── Face Mapping Data ────────────────────────────────────────────────────────

source_target_map: List[Dict[str, Any]] = []
simple_map: Dict[str, Any] = {}

# ── Paths ────────────────────────────────────────────────────────────────────

source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# ── Processing Options ───────────────────────────────────────────────────────

frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False
map_faces: bool = False
poisson_blend: bool = False
color_correction: bool = False
nsfw_filter: bool = False

# ── Video Output ─────────────────────────────────────────────────────────────

video_encoder: str | None = None
video_quality: int | None = None

# ── Live Mode ────────────────────────────────────────────────────────────────

live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None
webcam_preview_running: bool = False
show_fps: bool = False

# ── System ───────────────────────────────────────────────────────────────────

max_memory: int | None = None
execution_providers: List[str] = []
execution_threads: int | None = None
headless: bool | None = None
log_level: str = "error"

# ── Face Processor UI Toggles ────────────────────────────────────────────────

fp_ui: Dict[str, bool] = {
    "face_enhancer": False,
    "face_enhancer_gpen256": False,
    "face_enhancer_gpen512": False,
}

# ── Face Swapper Options ─────────────────────────────────────────────────────

face_swapper_enabled: bool = True
opacity: float = 1.0
sharpness: float = 0.0

# ── Mouth Mask ───────────────────────────────────────────────────────────────

mouth_mask: bool = False
show_mouth_mask_box: bool = False
mask_feather_ratio: int = 12
mask_down_size: float = 0.1
mask_size: float = 1.0

# ── Interpolation ────────────────────────────────────────────────────────────

enable_interpolation: bool = False
interpolation_weight: float = 0

# ── Cached Source Embedding (computed once at source selection) ───────────────

cached_source_face: Any = None
cached_source_embedding: Any = None

def apply_preset(preset: QualityPreset) -> None:
    """Apply a quality preset, updating relevant globals."""
    global quality_preset, poisson_blend, sharpness, interpolation_weight, enable_interpolation
    quality_preset = preset
    cfg = PRESET_CONFIGS[preset]
    poisson_blend = cfg["poisson_blend"]
    sharpness = cfg["sharpness"]
    interpolation_weight = cfg["interpolation_weight"]
    enable_interpolation = cfg["interpolation_weight"] > 0
