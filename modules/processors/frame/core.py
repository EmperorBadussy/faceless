"""PHANTOM-FACE frame processor core.

Changes from original:
  - Keeps ThreadPoolExecutor for ONNX inference (ONNX releases GIL in C++)
  - Removes batch barrier (old code waited for each batch before starting next)
  - Uses streaming futures with as_completed() instead of blocking batch waits
  - Proper OMP_NUM_THREADS management
"""

import os
import sys
import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import modules
import modules.globals

# Force OMP threads to 1 when using GPU — prevents thread oversubscription
# (Original only did this when --execution-provider was in argv)
os.environ.setdefault('OMP_NUM_THREADS', '1')

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                print(f"[PHANTOM] Frame processor {frame_processor} missing method {method_name}")
                sys.exit()
    except ImportError:
        print(f"[PHANTOM] Frame processor {frame_processor} not found")
        sys.exit()
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES


def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    global FRAME_PROCESSORS_MODULES
    current_processor_names = [proc.__name__.split('.')[-1] for proc in FRAME_PROCESSORS_MODULES]

    for frame_processor, state in modules.globals.fp_ui.items():
        if state and frame_processor not in current_processor_names:
            try:
                frame_processor_module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.append(frame_processor_module)
                if frame_processor not in modules.globals.frame_processors:
                    modules.globals.frame_processors.append(frame_processor)
            except (SystemExit, Exception) as e:
                print(f"[PHANTOM] Warning: Failed to load {frame_processor}: {e}")
        elif not state and frame_processor in current_processor_names:
            try:
                module_to_remove = next(
                    (mod for mod in FRAME_PROCESSORS_MODULES if mod.__name__.endswith(f'.{frame_processor}')),
                    None
                )
                if module_to_remove:
                    FRAME_PROCESSORS_MODULES.remove(module_to_remove)
                if frame_processor in modules.globals.frame_processors:
                    modules.globals.frame_processors.remove(frame_processor)
            except Exception as e:
                print(f"[PHANTOM] Warning: Error removing {frame_processor}: {e}")


def multi_process_frame(
    source_path: str,
    temp_frame_paths: List[str],
    process_frames: Callable[[str, List[str], Any], None],
    progress: Any = None,
) -> None:
    """Process frames with streaming ThreadPoolExecutor (no batch barriers).

    ONNX Runtime releases the GIL during inference (C++ execution), so threads
    work well for the inference-heavy workload. The key fix is removing the
    batch barrier — we now use as_completed() for streaming execution.
    """
    max_workers = modules.globals.execution_threads or 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit ALL frames at once — let the executor manage scheduling
        # (bounded by max_workers, no artificial batching)
        futures = {}
        for path in temp_frame_paths:
            future = executor.submit(process_frames, source_path, [path], progress)
            futures[future] = path

        # Stream results as they complete (no batch barrier!)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[PHANTOM] Error processing {futures[future]}: {e}")


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=bar_format) as progress:
        progress.set_postfix({
            'providers': modules.globals.execution_providers,
            'threads': modules.globals.execution_threads,
        })
        multi_process_frame(source_path, frame_paths, process_frames, progress)
