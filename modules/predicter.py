"""FACELESS NSFW predicter.

FACELESS: This module is ONLY loaded when --nsfw-filter is used.
TensorFlow and opennsfw2 are imported lazily, not at startup.
If you don't use --nsfw-filter, neither TF nor opennsfw2 are loaded.
"""

import numpy
from PIL import Image
import cv2
import modules.globals
from modules.gpu_processing import gpu_cvt_color
from modules.typing import Frame

MAX_PROBABILITY = 0.85

model = None
_opennsfw2 = None


def _get_opennsfw2():
    global _opennsfw2
    if _opennsfw2 is None:
        import opennsfw2
        _opennsfw2 = opennsfw2
    return _opennsfw2


def predict_frame(target_frame: Frame) -> bool:
    nsfw = _get_opennsfw2()

    if modules.globals.color_correction:
        target_frame = gpu_cvt_color(target_frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(target_frame)
    image = nsfw.preprocess_image(image, nsfw.Preprocessing.YAHOO)
    global model
    if model is None:
        model = nsfw.make_open_nsfw_model()

    views = numpy.expand_dims(image, axis=0)
    _, probability = model.predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    return _get_opennsfw2().predict_image(target_path) > MAX_PROBABILITY


def predict_video(target_path: str) -> bool:
    _, probabilities = _get_opennsfw2().predict_video_frames(video_path=target_path, frame_interval=100)
    return any(probability > MAX_PROBABILITY for probability in probabilities)
