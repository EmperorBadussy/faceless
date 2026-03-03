"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation, pre/post processing, and the core
enhance-face-via-ONNX pipeline.
"""

import os
import platform
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime

import modules.globals

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
THREAD_SEMAPHORE = threading.Semaphore(min(max(1, (os.cpu_count() or 1)), 8))

# Cached feathered masks by input_size (identical every call, compute once)
_MASK_CACHE: dict[int, np.ndarray] = {}


def _get_feathered_mask(input_size: int) -> np.ndarray:
    """Get cached feathered edge mask for blending."""
    if input_size not in _MASK_CACHE:
        mask = np.ones((input_size, input_size), dtype=np.float32)
        border = max(1, input_size // 16)
        mask[:border, :] = np.linspace(0, 1, border, dtype=np.float32)[:, np.newaxis]
        mask[-border:, :] = np.linspace(1, 0, border, dtype=np.float32)[:, np.newaxis]
        mask[:, :border] = np.minimum(mask[:, :border], np.linspace(0, 1, border, dtype=np.float32)[np.newaxis, :])
        mask[:, -border:] = np.minimum(mask[:, -border:], np.linspace(1, 0, border, dtype=np.float32)[np.newaxis, :])
        _MASK_CACHE[input_size] = mask
    return _MASK_CACHE[input_size]


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session with full graph optimization."""
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    providers = modules.globals.execution_providers
    session = onnxruntime.InferenceSession(model_path, sess_options=opts, providers=providers)
    return session


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / compile caching."""
    try:
        input_feed = {
            inp.name: np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
                dtype=np.float32,
            )
            for inp in session.get_inputs()
        }
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    """
    resized = cv2.resize(face_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) - 127.5) * (1.0 / 127.5)
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[np.newaxis, ...])
    return blob


def postprocess_face(output: np.ndarray) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8 image."""
    img = output[0].transpose(1, 2, 0)
    img = ((img + 1.0) / 2.0 * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _get_face_affine(face: Any, input_size: int):
    """Compute affine transform to align a face to GPEN input space.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    template = np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model."""
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    blob = preprocess_face(face_crop, input_size)
    with THREAD_SEMAPHORE:
        output = session.run(None, {session.get_inputs()[0].name: blob})[0]
    enhanced = postprocess_face(output)

    # Use cached feathered mask
    mask = _get_feathered_mask(input_size)

    h, w = frame.shape[:2]
    warped_enhanced = cv2.warpAffine(
        enhanced, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    # ROI-only blending: only convert to float32 in the face region, not the whole frame
    nz_rows = np.any(warped_mask > 0, axis=1)
    nz_cols = np.any(warped_mask > 0, axis=0)
    if not np.any(nz_rows) or not np.any(nz_cols):
        return frame
    ry1, ry2 = np.argmax(nz_rows), h - np.argmax(nz_rows[::-1])
    rx1, rx2 = np.argmax(nz_cols), w - np.argmax(nz_cols[::-1])

    roi_mask = warped_mask[ry1:ry2, rx1:rx2, np.newaxis]
    roi_enhanced = warped_enhanced[ry1:ry2, rx1:rx2].astype(np.float32)
    roi_frame = frame[ry1:ry2, rx1:rx2].astype(np.float32)
    blended = (roi_enhanced * roi_mask + roi_frame * (1.0 - roi_mask))
    result = frame.copy()
    result[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result
