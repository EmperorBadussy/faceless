<div align="center">

# PHANTOM FACE

**Real-Time GPU-Accelerated Face Swap Engine**

*High-performance face swapping at 30fps+ with CUDA-optimized pipeline, Electron UI, and virtual camera output*

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)](https://python.org)
[![Electron](https://img.shields.io/badge/Electron-vite-purple?logo=electron&logoColor=white)](https://electron-vite.org)
[![CUDA](https://img.shields.io/badge/CUDA-12-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-GPU-orange)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/License-AGPL--3.0-red)](LICENSE)

---

</div>

## Overview

PHANTOM FACE is a real-time face swap engine built for speed and quality. It combines ONNX Runtime GPU inference with a resolution-decoupled processing pipeline to achieve 30fps+ face swapping on consumer hardware.

The system captures at full 1080p, processes face swaps at an optimized lower resolution (854x480), and composites the result back to 1080p -- because the swap model works at 128x128 internally regardless of input size. This eliminates the biggest bottleneck (CPU-side affine warping) while maintaining output quality.

### Key Features

- **Real-time performance** -- 25-30fps on NVIDIA RTX GPUs with CUDA acceleration
- **Resolution-decoupled pipeline** -- Captures at 1080p, swaps at 480p, outputs at 1080p
- **Modern Electron UI** -- React 19 + TypeScript + Tailwind CSS with glassmorphism design
- **WebSocket architecture** -- Decoupled Python backend + Electron frontend
- **Virtual camera output** -- Feed swapped video to Discord, Zoom, OBS via pyvirtualcam
- **GPU-accelerated processing** -- ONNX Runtime CUDA EP, OpenCV CUDA where available
- **Face enhancement** -- Optional GPEN-BFR-256 neural face restoration
- **Quality presets** -- NORMAL (balanced) and HIGH (maximum quality) modes
- **Multi-face support** -- Swap multiple faces simultaneously
- **Post-processing** -- Color correction, mouth masking, sharpening, opacity control

## Architecture

```
+-------------------------------------------------------------+
|                      Electron UI (React)                     |
|   Source Face | Camera Select | Controls | Live Preview      |
+----------------------------+--------------------------------+
                             | WebSocket (ws://127.0.0.1:7865)
                             | JSON control | Binary JPEG frames
+----------------------------+--------------------------------+
|                    Python WebSocket Server                    |
|                                                              |
|  +----------+  +----------+  +------------+  +----------+   |
|  | Capture  |->| Detect   |->| Processing |->| Display  |   |
|  | Thread   |  | Thread   |  | Thread     |  | Thread   |   |
|  | (30fps)  |  | (async)  |  | (swap+enh) |  | (encode) |   |
|  +----------+  +----------+  +------------+  +----------+   |
|       |                            |                         |
|   Camera/OBS                ONNX Runtime                     |
|   VideoCapture              CUDAExecutionProvider             |
+-------------------------------------------------------------+
```

**Pipeline flow:**
1. **Capture thread** grabs frames at 1080p/30fps from the selected camera
2. **Detection thread** runs face detection (RetinaFace) asynchronously on full-res frames
3. **Processing thread** downscales to 480p, applies face swap, optional enhancement (GPEN-BFR-256), then upscales back to 1080p
4. **Display thread** encodes to JPEG and broadcasts to WebSocket clients + virtual camera

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (RTX series recommended)
- Webcam or capture device
- 8GB+ RAM

### Software
- Python 3.12+
- Node.js 20+
- NVIDIA GPU drivers (latest)
- OBS Studio (optional, for virtual camera driver)

## Installation

```bash
# Clone the repository
git clone https://github.com/EmperorBadussy/phantom-face.git
cd phantom-face

# Install Python dependencies
pip install -r requirements.txt

# Install CUDA support for ONNX Runtime
pip install onnxruntime-gpu

# Install NVIDIA CUDA pip packages (for GPU acceleration)
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12

# Install Electron UI dependencies
cd ui
npm install
cd ..
```

### Model Downloads

Download the required models to the `models/` directory:

| Model | Purpose | Size |
|-------|---------|------|
| `inswapper_128_fp16.onnx` | Face swap | 277MB |
| `GPEN-BFR-256.onnx` | Face enhancement (optional) | 76MB |

The face detection model (`buffalo_l`) is downloaded automatically by InsightFace on first run.

## Usage

### Electron UI (Recommended)

```bash
cd ui
npm run dev
```

This launches the Electron app which automatically starts the Python backend server.

### Server Only

```bash
python run.py --server --execution-provider cuda
```

Then connect any WebSocket client to `ws://127.0.0.1:7865`.

### Controls

| Control | Description |
|---------|-------------|
| **Source Face** | Select an image containing the face to swap onto targets |
| **Camera** | Select input camera device |
| **Quality Preset** | NORMAL (30fps balanced) or HIGH (max quality) |
| **Many Faces** | Swap all detected faces vs. just the primary one |
| **GPEN 256** | Enable neural face enhancement (adds ~5ms) |
| **Color Correction** | Match swapped face color to target |
| **Mouth Mask** | Preserve original mouth for better lip sync |
| **Sharpness** | Post-swap sharpening amount |
| **Opacity** | Blend between original and swapped face |
| **Virtual Camera** | Output to OBS Virtual Camera for use in other apps |

## Performance

Benchmarked with RTX GPU and 1080p webcam input:

| Stage | Time | Notes |
|-------|------|-------|
| Camera capture | ~2ms | 1920x1080 @ 30fps |
| Face detection | ~3-5ms | RetinaFace via ONNX CUDA |
| Downscale to 480p | <1ms | cv2.resize INTER_AREA |
| Face swap (inswapper) | ~5ms | 128x128 model, CUDA EP |
| Face enhance (GPEN-256) | ~5ms | 256x256 model, CUDA EP |
| Upscale to 1080p | <1ms | cv2.resize INTER_LINEAR |
| JPEG encode + send | ~3ms | Quality 65, max 960px wide |
| **Total pipeline** | **~20-30ms** | **30-50fps effective** |

### Resolution vs. Performance

The processing resolution can be tuned for your hardware:

| Process Resolution | Swap Time | Effective FPS |
|-------------------|-----------|---------------|
| 1920x1080 | ~98ms | ~10fps |
| 1280x720 | ~44ms | ~23fps |
| 960x540 | ~37ms | ~27fps |
| **854x480 (default)** | **~32ms** | **~31fps** |
| 640x360 | ~19ms | ~52fps |

## Tech Stack

**Backend:**
- Python 3.12
- ONNX Runtime (CUDA Execution Provider)
- InsightFace (face detection + analysis)
- OpenCV (image processing, camera capture)
- websockets (async WebSocket server)
- pyvirtualcam (virtual camera output)

**Frontend:**
- Electron + electron-vite
- React 19
- TypeScript
- Tailwind CSS 4
- Zustand (state management)

## Project Structure

```
phantom-face/
├── run.py                  # Entry point (CUDA DLL setup + launch)
├── modules/
│   ├── server.py           # WebSocket server + camera enumeration
│   ├── live_pipeline.py    # 4-thread capture/detect/process/display pipeline
│   ├── face_analyser.py    # InsightFace face detection wrapper
│   ├── globals.py          # Global settings + quality presets
│   ├── gpu_processing.py   # OpenCV CUDA wrappers
│   └── processors/frame/
│       ├── face_swapper.py          # Face swap + post-processing
│       ├── face_enhancer_gpen256.py # GPEN-BFR-256 face enhancement
│       └── _onnx_enhancer.py        # Shared ONNX enhancer utilities
├── ui/
│   ├── src/main/           # Electron main process
│   ├── src/preload/        # Context bridge
│   └── src/renderer/       # React UI
│       ├── components/     # Controls, preview, settings panels
│       ├── lib/hooks/      # WebSocket hook
│       └── lib/stores/     # Zustand state stores
└── models/                 # ONNX models (not tracked in git)
```

## License

This project is licensed under the GNU Affero General Public License v3.0 -- see the [LICENSE](LICENSE) file for details.

Based on [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam).

## Disclaimer

This software is intended for educational and research purposes. Users are responsible for ensuring their use complies with all applicable laws and regulations. The developers do not condone the use of this software for deceptive, harmful, or illegal purposes.
