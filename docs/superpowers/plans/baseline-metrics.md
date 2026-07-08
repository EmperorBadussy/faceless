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

_(to be filled after Task 2.1/2.2 — expect large multi-face gains)_

## Final

_(to be filled at end)_
