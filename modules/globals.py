"""FACELESS global configuration.

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
        "det_size": (640, 640),          # Higher-res detection (RTX 5090 has headroom)
        "capture_width": 1920,           # Full HD capture
        "capture_height": 1080,          # Full HD capture
        "capture_fps": 30,              # 30fps target
        "enhancer_model": None,          # No enhancer in normal mode
        "max_vram_gb": 8,                # VRAM budget
        "process_workers": 4,            # ProcessPoolExecutor workers for video
        "use_fp16": True,                # FP16 inference
        "face_swap_model": "reswapper_256.onnx",
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
min_swap_score: float = 0.55
use_tensorrt: bool = True  # TRT with fp16 OFF (see trt_provider_options): correct + faster than CUDA
execution_providers: List[str] = []
execution_threads: int | None = None
headless: bool | None = None
log_level: str = "error"


def cuda_provider_options() -> dict:
    """Tuned CUDAExecutionProvider options.

    HEURISTIC avoids the multi-second EXHAUSTIVE cuDNN algo search that ORT runs
    by default on first inference (and re-runs on every new input shape).
    """
    return {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
        "arena_extend_strategy": "kSameAsRequested",
    }


def trt_provider_options() -> dict:
    """Tuned TensorrtExecutionProvider options.

    fp16 is DISABLED: TRT's aggressive fp16 activations overflow on the inswapper
    swap model for some source faces and produce green/magenta garbage. fp32 is
    numerically correct for every source AND still faster than plain CUDA
    (~3.6 ms/face vs ~6 ms). Engine + timing caches so the slow build happens once.
    """
    cache_dir = os.path.join(ROOT_DIR, "..", "models", "trt_cache_fp32")
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    return {
        "device_id": 0,
        "trt_fp16_enable": False,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_timing_cache_enable": True,
    }


def _provider_name(p) -> str:
    return p[0] if isinstance(p, tuple) else p


def provider_options_for(name: str):
    """Options dict for a provider name, or None for bare providers."""
    if name == "CUDAExecutionProvider":
        return cuda_provider_options()
    if name == "TensorrtExecutionProvider":
        return trt_provider_options()
    return None


def providers_with_options() -> list:
    """execution_providers with CUDA/TensorRT entries expanded to (name, options)."""
    out = []
    for p in execution_providers:
        name = _provider_name(p)
        opts = provider_options_for(name)
        out.append((name, opts) if opts is not None else p)
    return out


def wants_cuda() -> bool:
    return any(_provider_name(p) == "CUDAExecutionProvider" for p in execution_providers)

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

# ── Multi-Source Embedding Averaging ─────────────────────────────────────────

source_embeddings: list = []
averaged_embedding: Any = None

def apply_preset(preset: QualityPreset) -> None:
    """Apply a quality preset, updating relevant globals."""
    global quality_preset, poisson_blend, sharpness, interpolation_weight, enable_interpolation
    quality_preset = preset
    cfg = PRESET_CONFIGS[preset]
    poisson_blend = cfg["poisson_blend"]
    sharpness = cfg["sharpness"]
    interpolation_weight = cfg["interpolation_weight"]
    enable_interpolation = cfg["interpolation_weight"] > 0
