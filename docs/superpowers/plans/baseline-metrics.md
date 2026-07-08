# FACELESS Optimization Metrics

Measured with `bench.py` (detection + GPU swap + post-processing on fixed images,
5 warmup iters discarded, per-stage `torch.cuda.synchronize()` timing).

**Environment:** RTX 5090 (Blackwell, 32 GB), driver 610.62, CUDA 12.8, Python 3.12.10,
torch 2.12.0.dev+cu128, onnxruntime-gpu 1.22.0 (CUDA 12 build, reusing torch cuDNN 9).
Swap model `inswapper_128_fp16.onnx`, detector buffalo_l `det_10g` at det_size 640.
All timings HIGH preset, single-frame (no pipeline thread overlap), 100 iters.

Test inputs: source `ludwig.jpg` (720p). Targets extracted from repo demo gifs:
`movie.jpg` (1 face), `streamers.jpg` (3 faces), `live_show.jpg` (9 faces).

Note: `[FACELESS-GPU] CUDA partial ... using CPU` — OpenCV has no CUDA on this box,
so `gpu_processing.py` runs on CPU. `cudnn_conv_algo_search: EXHAUSTIVE` confirmed
active (provider options not yet tuned — Task 1.1).

---

## Baseline (Phase 0) — before any optimization

| Target | Faces | detect (ms) | swap (ms) | swap/face | post (ms) | total (ms) | FPS |
|--------|------:|------------:|----------:|----------:|----------:|-----------:|----:|
| movie      | 1 | 8.40 | 10.27 | 10.27 | 0.02 | 18.68 | 53.5 |
| streamers  | 3 | 8.42 | 35.85 | 11.95 | 0.02 | 44.29 | 22.6 |
| live_show  | 9 | 8.50 | 95.17 | 10.57 | 0.03 | 103.70 | 9.6 |

Observations:
- Swap cost is linear in face count at ~10.6 to 12 ms/face — this is the full-frame
  warp/blend + full-frame upload/download per face (audit finding #1).
- Detection is constant ~8.4 ms (fixed 640x640 det_size).
- Post-processing is negligible in this config (no mask/interpolation enabled).
- Single-face 10.3 ms swap is dominated by full-frame GPU ops + PCIe transfer, not the
  128x128 model inference itself. ROI clipping (Phase 2) should cut this sharply.

---

## Phase 1 — CUDA provider options + no silent CPU fallback

Provider options now `cudnn_conv_algo_search: HEURISTIC` (was EXHAUSTIVE),
`arena_extend_strategy: kSameAsRequested`. All sessions confirmed on
`CUDAExecutionProvider`; no `[WARN] running on CPU`.

| Target | Faces | detect (ms) | swap (ms) | total (ms) | FPS |
|--------|------:|------------:|----------:|-----------:|----:|
| streamers | 3 | 8.39 | 35.84 | 44.25 | 22.6 |

Steady-state unchanged vs baseline (expected): HEURISTIC removes the first-inference
EXHAUSTIVE algo-search stall and re-search on shape changes (a startup/latency win),
which the warmup-discarding benchmark does not capture. Value here is correctness
(no silent CPU fallback) + faster cold start, not steady-state throughput.

## Phase 2 — ROI-clipped resident-GPU swap

`swap_face_gpu` now warps/blends only the face bbox (+64px) and uploads/downloads
only that ROI instead of the full frame; crop mask cached. Output verified correct
(no seams) at 1, 3, and 9 faces.

| Target | Faces | swap baseline | swap now | total baseline | total now | FPS base | FPS now |
|--------|------:|--------------:|---------:|---------------:|----------:|---------:|--------:|
| movie      | 1 | 10.27 |  9.30 |  18.68 | 17.93 | 53.5 | 55.8 |
| streamers  | 3 | 35.85 | 26.56 |  44.29 | 35.32 | 22.6 | 28.3 |
| live_show  | 9 | 95.17 | 78.36 | 103.70 | 87.16 |  9.6 | 11.5 |

Gain scales with face count (multi-face -26%, single -9%). Measurement reveals the
full-frame warp was NOT the dominant single-face cost: there is a large per-face
fixed cost (~8 ms/face) from the ONNX inswapper inference plus the per-face
GPU->CPU sync in swap_face_gpu. That per-face fixed cost is the next target.

## Per-face cost decomposition (measured)

Profiling `swap_face_gpu` internals (single face, 810p, 100 iters):
- ONNX `inswapper.get()` (128 crop + inference): **7.52 ms** ← dominant per-face cost
- torch warp/blend/transfer (ROI): ~1.8 ms
- frame.copy(): 0.49 ms

The remaining per-face bottleneck is the ONNX swap inference itself, not the warp.
7.5 ms for a 128x128 fp16 model on a 5090 is high; `TensorRT EP skipped (libs not
installed)` in the logs. TensorRT (or IO binding + fp16 I/O) is the next lever, but
it is a heavy dependency and out of the original plan's scope.

## TensorRT execution provider (biggest lever)

Installed tensorrt-cu12 10.x (v11 is ABI-incompatible with onnxruntime 1.22's TRT
EP). gpu_dll_setup.py registers the tensorrt_libs dir before onnxruntime imports;
core.py detects TRT via that flag (pip ships versioned nvinfer_10.dll, so
find_library('nvinfer') failed). Swap + detection sessions use the TRT EP with
fp16 and an engine cache (models/trt_cache). First run builds engines (slow, ~1-2
min), cached thereafter.

Full result — ROI + TensorRT vs original baseline (HIGH preset, 100 iters):

| Target | Faces | detect | swap | total | FPS | baseline FPS | speedup |
|--------|------:|-------:|-----:|------:|----:|-------------:|--------:|
| movie      | 1 | 5.69 |  5.18 | 10.89 | 91.8 | 53.5 | 1.72x |
| streamers  | 3 | 5.63 | 13.64 | 19.29 | 51.8 | 22.6 | 2.29x |
| live_show  | 9 | 5.56 | 39.05 | 44.64 | 22.4 |  9.6 | 2.33x |

TensorRT fp16 cut per-face swap ~10.6 ms -> ~4.5 ms and detection 8.4 -> 5.6 ms.
Output verified visually correct (no quality loss from fp16). The 9-face case moved
from below real-time (9.6 FPS) to above (22.4 FPS).

## Status summary

Done + verified: Phase 0 (GPU stack fixed — was running inference on CPU), Phase 1
(provider options + CPU-fallback warnings; GpuProcessor channel-count fix), Phase 2
(ROI-clipped swap: multi-face -26%), Phase 5 (dead code removed), Phase 4 partial
(stale-frame pacing, buffersize, vcam BGR — verified by import, not live camera),
Phase 3 partial (enhancers detect-only).

Deferred (need decision or unavailable models / live camera to verify):
- TensorRT EP for the swap inference (biggest remaining lever; heavy dep).
- Task 3.2/3.3: GFPGAN mask caching port + enhancer warmup at real size (need
  gfpgan-1024.onnx / GPEN models to benchmark).
- Task 4.3: single camera open + pygrabber enumeration (startup-only; restructures
  camera handling, needs a live device to verify).
- reswapper_256.onnx (NORMAL preset default) not sourced; benchmarks used inswapper.
