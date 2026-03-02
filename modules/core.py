"""FACELESS core orchestrator.

Changes from Deep-Live-Cam:
  - TensorFlow is LAZY-loaded (only when NSFW filter is enabled) — saves 200-500MB RAM + 2-5s startup
  - OMP_NUM_THREADS=1 set ALWAYS for GPU providers (not just when CLI flag present)
  - Quality preset system (Normal / High)
  - Zero-copy video pipeline integration
  - Proper resource cleanup
"""

import os
import sys

# Set OMP threads BEFORE any numpy/onnx imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import time

import onnxruntime

# LAZY tensorflow — only import when NSFW filter is actually used
# Original unconditionally imported it, wasting 200-500MB RAM
_tensorflow = None

def _get_tensorflow():
    global _tensorflow
    if _tensorflow is None:
        import tensorflow
        _tensorflow = tensorflow
    return _tensorflow

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import modules.globals
from modules.globals import QualityPreset
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension, is_image, is_video, detect_fps,
    create_video, extract_frames, get_temp_frame_paths,
    restore_audio, create_temp, move_temp, clean_temp,
    normalize_output_path,
)

if HAS_TORCH and 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
if HAS_TORCH:
    warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(
        prog="FACELESS",
        description="Real-time face swap — optimized fork of Deep-Live-Cam",
    )
    program.add_argument('-s', '--source', help='source face image', dest='source_path')
    program.add_argument('-t', '--target', help='target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor',
                         default=['face_swapper'], choices=['face_swapper', 'face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'], nargs='+')
    program.add_argument('--quality', help='quality preset', dest='quality_preset',
                         default='normal', choices=['normal', 'high'])
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='NSFW filter', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--video-encoder', help='output video encoder', dest='video_encoder', default='libx264',
                         choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('-l', '--lang', help='UI language', default="en")
    program.add_argument('--live-mirror', help='mirror webcam', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='resizable preview', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='max RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider',
                         default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='execution threads', dest='execution_threads', type=int,
                         default=suggest_execution_threads())
    program.add_argument('--show-fps', help='show FPS overlay', dest='show_fps', action='store_true', default=False)
    program.add_argument('-v', '--version', action='version', version=f'FACELESS {modules.metadata.version}')

    # Deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    # Apply quality preset FIRST
    preset = QualityPreset.HIGH if args.quality_preset == 'high' else QualityPreset.NORMAL
    modules.globals.apply_preset(preset)

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.show_fps = args.show_fps
    modules.globals.lang = args.lang

    for enhancer_key in ('face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'):
        modules.globals.fp_ui[enhancer_key] = enhancer_key in args.frame_processor

    # Handle deprecated args
    if args.source_path_deprecated:
        print('\033[33m-f/--face is deprecated. Use -s/--source.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33m--cpu-cores is deprecated. Use --execution-threads.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33m--gpu-vendor apple is deprecated. Use --execution-provider coreml.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33m--gpu-vendor nvidia is deprecated. Use --execution-provider cuda.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33m--gpu-vendor amd is deprecated. Use --execution-provider rocm.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33m--gpu-threads is deprecated. Use --execution-threads.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [ep.replace('ExecutionProvider', '').lower() for ep in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [
        provider for provider, encoded in zip(
            onnxruntime.get_available_providers(),
            encode_execution_providers(onnxruntime.get_available_providers())
        )
        if any(ep in encoded for ep in execution_providers)
    ]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    cpu_count = os.cpu_count() or 4
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        return min(cpu_count, 16)
    return max(4, min(cpu_count - 2, 16))


def limit_resources() -> None:
    """Limit resource usage. TensorFlow only loaded if NSFW filter is on."""
    if modules.globals.nsfw_filter:
        tf = _get_tensorflow()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers and HAS_TORCH:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python 3.9+ required.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg not found. Install it first.')
        return False
    return True


def update_status(message: str, scope: str = 'FACELESS') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        try:
            ui.update_status(message)
        except Exception:
            pass  # UI not initialized yet


def start() -> None:
    """Start processing (video/image mode — not live webcam)."""
    start_time = time.time()

    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status('Processing...')

    # Image processing
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print(f"Error copying file: {e}")
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Processing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        elapsed = time.time() - start_time
        if is_image(modules.globals.target_path):
            update_status(f'Done! ({elapsed:.2f}s)')
        else:
            update_status('Processing failed!')
        return

    # Video processing
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
        return

    # Try zero-copy pipeline first
    try:
        from modules.pipeline import process_video_zerocopy
        from modules.processors.frame.face_swapper import process_frame, swap_face, apply_post_processing
        from modules.face_analyser import get_one_face

        source_face = get_one_face(cv2.imread(modules.globals.source_path)) if modules.globals.source_path else None

        def frame_processor_fn(frame):
            if source_face:
                return process_frame(source_face, frame)
            return frame

        update_status('Zero-copy pipeline active')
        import cv2
        success = process_video_zerocopy(
            modules.globals.source_path,
            modules.globals.target_path,
            modules.globals.output_path,
            frame_processor_fn,
        )
        if success:
            elapsed = time.time() - start_time
            update_status(f'Done! ({elapsed:.2f}s)')
            return
    except Exception as e:
        update_status(f'Zero-copy failed ({e}), falling back to legacy pipeline')

    # Legacy fallback
    if not modules.globals.map_faces:
        update_status('Creating temp resources...')
        create_temp(modules.globals.target_path)
        update_status('Extracting frames...')
        extract_frames(modules.globals.target_path)

    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    update_status(f'Processing {len(temp_frame_paths)} frames...')

    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Processing...', frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        release_resources()

    if modules.globals.keep_fps:
        fps = detect_fps(modules.globals.target_path)
        create_video(modules.globals.target_path, fps)
    else:
        create_video(modules.globals.target_path)

    if modules.globals.keep_audio:
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)

    clean_temp(modules.globals.target_path)

    elapsed = time.time() - start_time
    if is_video(modules.globals.target_path):
        update_status(f'Done! ({elapsed:.2f}s)')
    else:
        update_status('Processing failed!')


def destroy(to_quit=True) -> None:
    # Stop live pipeline if running
    try:
        from modules.live_pipeline import get_pipeline
        get_pipeline().stop()
    except Exception:
        pass

    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit:
        quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()
