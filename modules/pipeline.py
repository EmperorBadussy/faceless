"""FACELESS zero-copy video processing pipeline.

Replaces the original disk-based pipeline (extract PNGs → read → process → write → re-encode)
with in-memory ffmpeg pipes. Frames flow through RAM only — zero disk I/O for temp frames.

Architecture:
    ffmpeg decode (subprocess) → raw bytes → numpy → process → numpy → raw bytes → ffmpeg encode (subprocess)

For live webcam mode, see the separate live_pipeline module.
"""

import os
import subprocess
import struct
import threading
import queue
import time
from typing import List, Callable, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

import modules.globals


def get_video_info(video_path: str) -> dict:
    """Extract video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        parts = output.split(",")
        # codec,width,height,pix_fmt,r_frame_rate,nb_frames
        width = int(parts[1]) if len(parts) > 1 else 1920
        height = int(parts[2]) if len(parts) > 2 else 1080

        # Parse frame rate
        fps_str = parts[4] if len(parts) > 4 else "30/1"
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = int(num) / max(1, int(den))
        else:
            fps = float(fps_str)

        # Parse frame count
        nb_frames = int(parts[5]) if len(parts) > 5 and parts[5].strip().isdigit() else 0

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "nb_frames": nb_frames,
            "codec": parts[0] if parts else "unknown",
        }
    except Exception as e:
        print(f"[pipeline] ffprobe failed: {e}, using defaults")
        return {"width": 1920, "height": 1080, "fps": 30.0, "nb_frames": 0, "codec": "unknown"}


def _build_decode_cmd(video_path: str, info: dict) -> List[str]:
    """Build ffmpeg command for raw frame decoding to stdout pipe."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", modules.globals.log_level,
        "-hwaccel", "auto",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "error",
        "-"
    ]


def _build_encode_cmd(output_path: str, info: dict, fps: float) -> List[str]:
    """Build ffmpeg command for raw frame encoding from stdin pipe."""
    width, height = info["width"], info["height"]

    # Determine encoder
    encoder = "libx264"
    encoder_opts = []
    quality = modules.globals.video_quality or 18

    if "CUDAExecutionProvider" in modules.globals.execution_providers:
        encoder = "h264_nvenc"
        encoder_opts = [
            "-preset", "p4",        # Fast preset (p7 is slow, p4 is good balance)
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(quality),
            "-b:v", "0",
            "-gpu", "0",
        ]
    elif "DmlExecutionProvider" in modules.globals.execution_providers:
        encoder = "h264_amf"
        encoder_opts = [
            "-quality", "speed",
            "-rc", "vbr_latency",
            "-qp_i", str(quality),
            "-qp_p", str(quality),
        ]
    else:
        encoder_opts = [
            "-preset", "medium",
            "-crf", str(quality),
        ]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", modules.globals.log_level,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", encoder,
    ]
    cmd.extend(encoder_opts)
    cmd.extend([
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",
        output_path,
    ])
    return cmd


def _reader_thread(proc: subprocess.Popen, frame_size: int, frame_queue: queue.Queue, total_frames: int):
    """Read raw frames from ffmpeg stdout and push to queue."""
    count = 0
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                break
            frame_queue.put(raw)
            count += 1
    except Exception as e:
        print(f"[pipeline] Reader error: {e}")
    finally:
        frame_queue.put(None)  # Sentinel


def _writer_thread(proc: subprocess.Popen, output_queue: queue.Queue):
    """Write processed frames to ffmpeg stdin."""
    try:
        while True:
            data = output_queue.get()
            if data is None:
                break
            proc.stdin.write(data)
    except Exception as e:
        print(f"[pipeline] Writer error: {e}")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass


def process_video_zerocopy(
    source_path: str,
    target_path: str,
    output_path: str,
    process_frame_fn: Callable[[np.ndarray], np.ndarray],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """Process a video file with zero disk I/O for intermediate frames.

    Args:
        source_path: Path to source face image
        target_path: Path to input video
        output_path: Path for output video
        process_frame_fn: Function that takes a BGR numpy frame and returns processed frame
        progress_callback: Optional callback(current_frame, total_frames)

    Returns:
        True on success
    """
    info = get_video_info(target_path)
    width, height = info["width"], info["height"]
    fps = info["fps"] if modules.globals.keep_fps else 30.0
    total_frames = info["nb_frames"]
    frame_size = width * height * 3  # BGR24

    print(f"[pipeline] Video: {width}x{height} @ {fps:.2f}fps, ~{total_frames} frames")
    print(f"[pipeline] Zero-copy pipeline: decode → process → encode (no temp files)")

    # Temp output (we'll mux audio later)
    temp_output = output_path + ".tmp.mp4"

    # Start decoder
    decode_cmd = _build_decode_cmd(target_path, info)
    decoder = subprocess.Popen(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_size * 4,
    )

    # Start encoder
    encode_cmd = _build_encode_cmd(temp_output, info, fps)
    encoder = subprocess.Popen(
        encode_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_size * 4,
    )

    # Frame queues (bounded to control memory)
    raw_queue: queue.Queue = queue.Queue(maxsize=32)
    out_queue: queue.Queue = queue.Queue(maxsize=32)

    # Start reader/writer threads
    reader = threading.Thread(target=_reader_thread, args=(decoder, frame_size, raw_queue, total_frames), daemon=True)
    writer = threading.Thread(target=_writer_thread, args=(encoder, out_queue), daemon=True)
    reader.start()
    writer.start()

    # Process frames
    processed = 0
    start_time = time.time()

    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    with tqdm(total=total_frames or None, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=bar_format) as pbar:
        while True:
            raw = raw_queue.get()
            if raw is None:
                break

            # Convert raw bytes to numpy (zero-copy via frombuffer + reshape)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()

            # Process
            result = process_frame_fn(frame)

            # Ensure correct format
            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)

            # Write to encoder
            out_queue.put(result.tobytes())

            processed += 1
            pbar.update(1)

            if progress_callback:
                progress_callback(processed, total_frames)

    # Signal writer to finish
    out_queue.put(None)
    writer.join(timeout=30)

    # Clean up
    decoder.wait(timeout=30)
    encoder.stdin.close() if encoder.stdin and not encoder.stdin.closed else None
    encoder.wait(timeout=60)

    elapsed = time.time() - start_time
    avg_fps = processed / elapsed if elapsed > 0 else 0
    print(f"[pipeline] Done: {processed} frames in {elapsed:.1f}s ({avg_fps:.1f} fps)")

    # Mux audio if requested
    if modules.globals.keep_audio:
        _mux_audio(target_path, temp_output, output_path)
        try:
            os.remove(temp_output)
        except OSError:
            pass
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_output, output_path)

    return True


def _mux_audio(source_video: str, processed_video: str, output_path: str):
    """Mux audio from source video into processed video."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", modules.globals.log_level,
        "-i", processed_video,
        "-i", source_video,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-y",
        output_path,
    ]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        # No audio stream or mux failed — just copy video
        import shutil
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.copy2(processed_video, output_path)


def process_image_zerocopy(
    source_path: str,
    target_path: str,
    output_path: str,
    process_frame_fn: Callable[[np.ndarray], np.ndarray],
) -> bool:
    """Process a single image."""
    frame = cv2.imread(target_path)
    if frame is None:
        print(f"[pipeline] Failed to read: {target_path}")
        return False

    result = process_frame_fn(frame)
    cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return True
