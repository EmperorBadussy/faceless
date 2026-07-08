# FACELESS Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FACELESS actually run on the GPU end-to-end, then eliminate the per-frame full-frame work that dominates the real-time swap loop, so effective FPS on the target hardware goes up and end-to-end latency comes down.

**Architecture:** The engine is a 3-thread pipeline (capture → detect → process) feeding a WebSocket/virtual-camera output loop. The per-frame hot path is `LivePipeline._processing_loop` → `swap_face_gpu` (PyTorch `grid_sample` warp) → `apply_post_processing`, plus optional ONNX enhancers. Optimization proceeds in strict order: (0) make the GPU real and establish a measured baseline, (1) fix silent CPU fallbacks, (2) restrict the GPU warp/blend to the face ROI, (3) tighten inference/enhancers, (4) cut capture/output latency, (5) delete dead code. Every change is verified against a headless benchmark harness — no change is accepted without before/after numbers.

**Tech Stack:** Python 3.12, PyTorch 2.12+cu128 (CUDA on RTX 5090 Blackwell), ONNX Runtime, InsightFace (buffalo_l), OpenCV (CPU-only build on this machine), websockets, pyvirtualcam. Electron/React UI is out of scope for this plan (headless backend only).

## Global Constraints

- Target hardware: NVIDIA RTX 5090 (Blackwell, sm_120), 32 GB VRAM, driver 610.62, CUDA 12.8. Any GPU package MUST be CUDA 12.8 compatible.
- Python 3.12.10. Do not change the Python version.
- `opencv-python` on this machine has NO CUDA support (`cv2.cuda.getCudaEnabledDeviceCount() == 0`). Do NOT assume `cv2.cuda` works. GPU compute goes through PyTorch, which does work.
- PyTorch is `2.12.0.dev+cu128` and `torch.cuda.is_available()` is True — this is the only working GPU compute path today.
- No hyphens/em-dashes in any user-facing strings or status messages (project copy rule).
- Every optimization task MUST show a before/after measurement from the benchmark harness (`bench.py`) and confirm the output frame is still visually correct (face present, no artifacts, no exceptions). Numbers before assertions.
- Do not restructure files that a task does not touch. Follow existing patterns in each file.
- Commit after every task with a descriptive message.

## Verification model (read before starting)

This repo has no unit-test suite, and unit tests do not meaningfully capture "is the swap faster and still correct." The verification instrument for this plan is **`bench.py`** (built in Task 0.3): it runs the real detection + swap + post-processing stages on fixed input images, reports per-stage mean milliseconds and effective FPS, and writes an output frame to disk for a visual correctness check. "The test" for each optimization task means: run `bench.py`, compare against the recorded baseline, and confirm the written output frame still shows a correct swap.

## File Structure

Files created or modified by this plan:

- Create: `bench.py` (repo root) — headless benchmark harness and correctness check. The measuring instrument for the whole plan.
- Create: `docs/superpowers/plans/baseline-metrics.md` — recorded baseline + post-phase numbers.
- Modify: `modules/core.py` — provider decoding + a loud CPU-fallback warning.
- Modify: `modules/face_analyser.py` — pass tuned CUDA provider options; assert CUDA actually bound.
- Modify: `modules/processors/frame/face_swapper.py` — provider options for the swap session; ROI-clipped resident-GPU warp in `swap_face_gpu`; cache the constant crop mask.
- Modify: `modules/processors/frame/_onnx_enhancer.py` — provider options; warmup at real input size.
- Modify: `modules/processors/frame/face_enhancer_gpen256.py`, `face_enhancer_gpen512.py` — use detect-only face lookup.
- Modify: `modules/processors/frame/face_enhancer.py` — detect-only lookup; cached feather mask + ROI blend (port from GPEN path).
- Modify: `modules/gpu_processing.py` — remove the download-to-read-channel-count round trip; make CPU fallback explicit.
- Modify: `modules/live_pipeline.py` — copy the work frame once before the per-face loop; set `CAP_PROP_BUFFERSIZE`; FOURCC ordering.
- Modify: `modules/server.py` — freshest-frame rate limiter; single camera open; fast enumeration; virtual camera in BGR; skip no-op resize before JPEG encode.
- Delete: `modules/processors/frame/face_masking.py` — dead duplicate (nothing imports it).
- Modify: `modules/processors/frame/face_swapper.py` — remove the broken `get_faces_optimized` (references undefined globals).

---

## Phase 0: Make the GPU real and establish a baseline (BLOCKING)

No optimization below is meaningful until ONNX inference runs on CUDA and we can measure. Do this phase first, in order.

### Task 0.1: Install a CUDA-enabled ONNX Runtime and confirm the CUDA provider is available

**Files:**
- Modify: `requirements.txt` (pin the GPU runtime)

**Interfaces:**
- Produces: a Python environment where `onnxruntime.get_available_providers()` includes `"CUDAExecutionProvider"`.

- [ ] **Step 1: Record the current (broken) state**

Run:
```bash
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```
Expected now: `1.25.0 ['AzureExecutionProvider', 'CPUExecutionProvider']` (no CUDA — this is the bug).

- [ ] **Step 2: Replace the CPU onnxruntime with the GPU build**

The plain `onnxruntime` wheel has no CUDA provider. Install the GPU build matching CUDA 12.x (Blackwell needs a recent build):
```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu>=1.22"
```
If the default index lacks a CUDA-12.8 wheel for this version, use the CUDA 12 index:
```bash
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

- [ ] **Step 3: Verify the CUDA provider is present**

Run:
```bash
python -c "import onnxruntime as ort; ps=ort.get_available_providers(); print(ps); assert 'CUDAExecutionProvider' in ps, 'CUDA EP STILL MISSING'"
```
Expected: a list containing `'CUDAExecutionProvider'` and no assertion error.

- [ ] **Step 4: Smoke-test a real session on CUDA**

Run:
```bash
python -c "import onnxruntime as ort, numpy as np; print('OK providers', ort.get_available_providers())"
```
Expected: prints `OK providers [... 'CUDAExecutionProvider' ...]`. If ORT prints a `LoadLibrary`/cuDNN error, the CUDA/cuDNN runtime DLLs are not on PATH — install `nvidia-cudnn-cu12` and `nvidia-cublas-cu12` via pip (they ship the DLLs) and retry.

- [ ] **Step 5: Pin it**

Edit `requirements.txt`: replace any `onnxruntime` line with `onnxruntime-gpu>=1.22`. If there is no such line, add it.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "build: use onnxruntime-gpu so inference runs on CUDA"
```

### Task 0.2: Download and reconcile the model files

**Files:**
- Modify: `models/instructions.txt` (correct the model list)

**Interfaces:**
- Produces: `models/` contains the ONNX files the presets actually reference.

- [ ] **Step 1: Determine which model files the code requires**

The presets in `modules/globals.py` reference: `reswapper_256.onnx` (NORMAL swap), `inswapper_128_fp16.onnx` (HIGH swap), `gfpgan-1024.onnx` (HIGH enhancer), and the GPEN enhancers use `GPEN-BFR-256.onnx` / `GPEN-BFR-512.onnx`. `instructions.txt` currently lists only `inswapper_128_fp16.onnx` and a `GFPGANv1.4.pth` (a .pth, not the .onnx the code loads). This mismatch is why NORMAL preset cannot start.

- [ ] **Step 2: Place the required models in `models/`**

At minimum for the benchmark, obtain `inswapper_128_fp16.onnx` (already documented) and set the active preset to one whose swap model you have. `inswapper_128_fp16.onnx`:
```
https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true
```
For `reswapper_256.onnx`, `GPEN-BFR-256.onnx`, `GPEN-BFR-512.onnx`, `gfpgan-1024.onnx`: source the ONNX exports the code expects. If a model is not available, do not reference that preset/enhancer in benchmarks.

- [ ] **Step 3: Confirm the files exist**

Run:
```bash
ls -la models/*.onnx
```
Expected: at least `inswapper_128_fp16.onnx` (~277MB) listed.

- [ ] **Step 4: Fix the instructions to match the code**

Rewrite `models/instructions.txt` to list every `.onnx` the presets reference (names exactly as in `globals.py`) with a working source URL each, and drop the `.pth` line (the code loads `.onnx`).

- [ ] **Step 5: Commit**

```bash
git add models/instructions.txt
git commit -m "docs: list the actual onnx models the presets require"
```
(Model binaries are gitignored — do not commit them.)

### Task 0.3: Build the headless benchmark harness

**Files:**
- Create: `bench.py`

**Interfaces:**
- Produces: `python bench.py --source <img> --target <img> [--iters N] [--preset normal|high]` prints per-stage mean ms + effective FPS and writes `bench_out.jpg`. Later tasks call this exact command to measure.

- [ ] **Step 1: Write the harness**

Create `bench.py` at the repo root:

```python
"""Headless benchmark + correctness check for the FACELESS hot path.

Runs detection + GPU swap + post-processing on fixed input images and reports
per-stage timing. This is the measuring instrument for the optimization plan.

Usage:
    python bench.py --source face.jpg --target scene.jpg --iters 100 --preset normal
"""
import argparse
import time
import cv2
import numpy as np


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Source face image")
    ap.add_argument("--target", required=True, help="Target frame image (a face to swap onto)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--preset", choices=["normal", "high"], default="normal")
    ap.add_argument("--provider", default="cuda")
    args = ap.parse_args()

    import modules.globals
    from modules.globals import QualityPreset
    modules.globals.quality_preset = (
        QualityPreset.HIGH if args.preset == "high" else QualityPreset.NORMAL
    )
    from modules.core import decode_execution_providers
    modules.globals.execution_providers = decode_execution_providers([args.provider])
    print(f"[bench] execution_providers = {modules.globals.execution_providers}")

    from modules.face_analyser import get_one_face, detect_many_faces
    from modules.processors.frame.face_swapper import swap_face_gpu, apply_post_processing

    source_img = cv2.imread(args.source)
    target_img = cv2.imread(args.target)
    if source_img is None or target_img is None:
        raise SystemExit("Could not read source/target image")

    source_face = get_one_face(source_img)
    if source_face is None:
        raise SystemExit("No face found in source image")

    # Confirm the swap session actually reports CUDA.
    from modules.processors.frame.face_swapper import get_face_swapper
    swp = get_face_swapper()
    sess = getattr(swp, "session", None)
    if sess is not None:
        print(f"[bench] swap session providers = {sess.get_providers()}")

    stages = {"detect": [], "swap": [], "post": [], "total": []}
    out = None
    for i in range(args.iters + 5):  # first 5 = warmup, discarded
        frame = target_img.copy()

        _sync(); t0 = time.perf_counter()
        faces = detect_many_faces(frame) or []
        _sync(); t1 = time.perf_counter()

        result = frame
        for f in faces:
            result = swap_face_gpu(source_face, f, result)
        _sync(); t2 = time.perf_counter()

        bboxes = [f.bbox.astype(int) for f in faces if getattr(f, "bbox", None) is not None]
        result = apply_post_processing(result, bboxes)
        _sync(); t3 = time.perf_counter()

        if i >= 5:
            stages["detect"].append((t1 - t0) * 1000)
            stages["swap"].append((t2 - t1) * 1000)
            stages["post"].append((t3 - t2) * 1000)
            stages["total"].append((t3 - t0) * 1000)
        out = result

    cv2.imwrite("bench_out.jpg", out)
    print(f"[bench] preset={args.preset} faces/frame={len(faces)} iters={args.iters}")
    for k in ("detect", "swap", "post", "total"):
        arr = np.array(stages[k])
        print(f"  {k:6s}  mean {arr.mean():6.2f} ms   p50 {np.percentile(arr,50):6.2f}   p95 {np.percentile(arr,95):6.2f}")
    fps = 1000.0 / np.array(stages["total"]).mean()
    print(f"  effective FPS (single frame, no pipeline overlap): {fps:5.1f}")
    print("  wrote bench_out.jpg for visual check")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and confirm it works end-to-end on CUDA**

Run (use any two face photos you have; the target should contain a face):
```bash
python bench.py --source path/to/face.jpg --target path/to/scene_with_face.jpg --iters 50
```
Expected: prints `execution_providers = [('CUDAExecutionProvider', ...), 'CPUExecutionProvider']` (or the string form), prints `swap session providers = [...'CUDAExecutionProvider'...]`, prints per-stage timings, and writes `bench_out.jpg`.

- [ ] **Step 3: Open `bench_out.jpg` and confirm the swap looks correct**

The written frame must show the source face swapped onto the target with no obvious artifacts. If the swap is missing, resolve model/detection issues before proceeding — the baseline must be a working swap.

- [ ] **Step 4: Add `bench_out.jpg` to .gitignore, commit the harness**

Add `bench_out.jpg` to `.gitignore`. Then:
```bash
git add bench.py .gitignore
git commit -m "test: add headless benchmark harness for the swap hot path"
```

### Task 0.4: Record the baseline

**Files:**
- Create: `docs/superpowers/plans/baseline-metrics.md`

- [ ] **Step 1: Run the harness for both presets and save the output**

```bash
python bench.py --source path/to/face.jpg --target path/to/scene.jpg --iters 100 --preset normal
python bench.py --source path/to/face.jpg --target path/to/scene.jpg --iters 100 --preset high
```

- [ ] **Step 2: Paste the numbers into a metrics file**

Create `docs/superpowers/plans/baseline-metrics.md` with a "Baseline (Phase 0)" section containing the full harness output for both presets (single-face and, if you have a multi-face target image, multi-face too — the multi-face numbers matter for Phase 2). Note the GPU, driver, torch, and onnxruntime-gpu versions.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/baseline-metrics.md
git commit -m "docs: record baseline performance metrics"
```

---

## Phase 1: Eliminate silent CPU fallbacks

### Task 1.1: Pass tuned CUDA provider options and assert CUDA actually bound

**Files:**
- Modify: `modules/face_analyser.py:31-36` and `:51-56` (both `FaceAnalysis` constructions)
- Modify: `modules/processors/frame/face_swapper.py` (the `INSwapper`/`get_model` and any `InferenceSession` creation around `:147-173`)
- Modify: `modules/core.py` (the provider decoder around `:162-185`)

**Interfaces:**
- Consumes: `modules.globals.execution_providers` (a list of provider entries).
- Produces: a helper `cuda_provider_options()` returning the tuned options dict, and a guarantee that when `cuda` was requested, sessions log loudly if they land on CPU.

- [ ] **Step 1: Add a provider-options helper in `modules/globals.py`**

Add near the execution-provider settings:
```python
def cuda_provider_options() -> dict:
    """Tuned CUDAExecutionProvider options. HEURISTIC avoids the multi-second
    EXHAUSTIVE cuDNN algo search on first inference."""
    return {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
        "arena_extend_strategy": "kSameAsRequested",
    }


def providers_with_options() -> list:
    """execution_providers with the CUDA entry expanded to (name, options)."""
    out = []
    for p in execution_providers:
        name = p[0] if isinstance(p, tuple) else p
        if name == "CUDAExecutionProvider":
            out.append(("CUDAExecutionProvider", cuda_provider_options()))
        else:
            out.append(p)
    return out
```

- [ ] **Step 2: Use the options in both analyser constructions**

In `modules/face_analyser.py`, change both `providers=modules.globals.execution_providers` (lines ~33 and ~53) to `providers=modules.globals.providers_with_options()`. After each `.prepare(...)`, add a fallback check:
```python
if any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
       for p in modules.globals.execution_providers):
    # buffalo_l models are internal to insightface; log the models' providers
    for m in FACE_ANALYSER.models.values():
        provs = m.session.get_providers()
        if "CUDAExecutionProvider" not in provs:
            print(f"[FACELESS][WARN] {m.__class__.__name__} running on CPU: {provs}")
```
(Adapt attribute access to the installed insightface version; the goal is a loud warning, not a crash.)

- [ ] **Step 3: Use the options in the swap session and assert**

In `modules/processors/frame/face_swapper.py` where the swap model/session is created (around `:147-173`), pass `providers=modules.globals.providers_with_options()`. After creation, add:
```python
_sess = getattr(FACE_SWAPPER, "session", None)
if _sess is not None:
    _provs = _sess.get_providers()
    _want_cuda = any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
                     for p in modules.globals.execution_providers)
    if _want_cuda and "CUDAExecutionProvider" not in _provs:
        print(f"[FACELESS][WARN] Face swapper running on CPU, not CUDA: {_provs}")
    else:
        print(f"[FACELESS] Face swapper providers: {_provs}")
```

- [ ] **Step 4: Run the harness, confirm no regression and CUDA bound**

```bash
python bench.py --source path/to/face.jpg --target path/to/scene.jpg --iters 100 --preset normal
```
Expected: `swap session providers` includes `CUDAExecutionProvider`; no `[WARN] running on CPU`; per-stage timings equal or better than baseline (first-run stall should shrink because HEURISTIC replaces EXHAUSTIVE). Record the numbers under a "Phase 1" section in `baseline-metrics.md`.

- [ ] **Step 5: Confirm `bench_out.jpg` still shows a correct swap, then commit**

```bash
git add modules/globals.py modules/face_analyser.py modules/processors/frame/face_swapper.py docs/superpowers/plans/baseline-metrics.md
git commit -m "perf: tuned CUDA provider options + loud CPU-fallback warnings"
```

### Task 1.2: Make the OpenCV-CUDA fallback explicit (no false GPU claims)

**Files:**
- Modify: `modules/gpu_processing.py:141,152` (remove the download-to-read-type round trip)
- Modify: `modules/gpu_processing.py:1-46` (docstring + startup log)

**Interfaces:**
- Produces: `GpuProcessor` blur/sharpen never call `.download()` for metadata; the module logs honestly that it is CPU-only on this machine.

- [ ] **Step 1: Track channel count at upload instead of downloading**

In `gpu_processing.py`, in `gaussian_blur` and `sharpen` (lines ~141 and ~152), replace `cv_type = _cv_type_for(self._gpu_mat.download())` with a cached value set during `upload()`. Add `self._channels` in `upload()` from the source array's channel count, and compute `cv_type` from `self._channels` without any `.download()`.

- [ ] **Step 2: Fix the misleading docstring/log**

Update the module docstring (lines 3-8) to state that GPU paths require an OpenCV built with CUDA, and that on a stock `opencv-python` wheel this module runs on CPU. The startup print already covers the CPU case; ensure the `CUDA_AVAILABLE=False` branch is the expected path here.

- [ ] **Step 3: Confirm no behavior change and no crash**

Run:
```bash
python -c "import modules.gpu_processing as g; print('CUDA_AVAILABLE', g.CUDA_AVAILABLE)"
python bench.py --source path/to/face.jpg --target path/to/scene.jpg --iters 50 --preset normal
```
Expected: prints `CUDA_AVAILABLE False`, harness runs, timings unchanged (this module is not in the default hot path but the round-trip trap is removed). Confirm `bench_out.jpg` correct.

- [ ] **Step 4: Commit**

```bash
git add modules/gpu_processing.py
git commit -m "perf: stop GpuProcessor downloading full frame just to read channel count"
```

---

## Phase 2: ROI-clipped, resident-GPU swap (the dominant win)

### Task 2.1: Restrict the GPU warp/blend/blur to the face ROI

**Files:**
- Modify: `modules/processors/frame/face_swapper.py:221-271` (the GPU section of `swap_face_gpu`)
- Modify: `modules/live_pipeline.py:347` (copy the work frame once before the per-face loop)

**Interfaces:**
- Consumes: `swap_face_gpu(source_face, target_face, temp_frame) -> Frame` (unchanged signature).
- Produces: same signature and return; internally warps/blends only within the target face bounding box plus padding, uploading/downloading only that ROI. Writes the blended ROI back into `temp_frame` and returns `temp_frame` (in place). Caller must pass a frame it owns (see the call-site change).

- [ ] **Step 1: Make the caller own the frame (avoid mutating the shared capture frame)**

In `modules/live_pipeline.py`, the per-face loop currently does `result = work_frame` then repeatedly reassigns `result = swap_face_gpu(...)`. Because `work_frame` is the same object published to the detection thread, in-place ROI writes would corrupt detection input. Change line ~347 from:
```python
            result = work_frame
```
to:
```python
            result = work_frame.copy() if scaled_faces else work_frame
```
So a single copy is made once per frame (only when there is something to swap), and `swap_face_gpu` may write into it in place.

- [ ] **Step 2: Replace the full-frame GPU block with an ROI block**

In `face_swapper.py`, replace the block from the two full-frame uploads through the download (lines ~221-271) with an ROI-scoped version. The affine `M` maps full-frame pixel coords to 128-crop coords, so the ROI meshgrid must use absolute frame coordinates:

```python
        h, w = temp_frame.shape[:2]
        crop_size = float(bgr_fake.shape[0])  # 128

        # Compute an ROI around the target face, padded to cover the feather blur.
        bbox = target_face.bbox.astype(int)
        pad = 64  # >= the k=41 blur radius so feathering stays inside the ROI
        x0 = max(0, int(bbox[0]) - pad)
        y0 = max(0, int(bbox[1]) - pad)
        x1 = min(w, int(bbox[2]) + pad)
        y1 = min(h, int(bbox[3]) + pad)
        if x1 <= x0 or y1 <= y0:
            return temp_frame
        rh, rw = y1 - y0, x1 - x0

        # Upload only the face crop (128x128) and the target ROI.
        fake_t = torch.from_numpy(np.ascontiguousarray(bgr_fake)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        target_roi = np.ascontiguousarray(temp_frame[y0:y1, x0:x1])
        target_t = torch.from_numpy(target_roi).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Meshgrid in ABSOLUTE frame coords over the ROI (M expects frame coords).
        ys = torch.arange(y0, y1, device=device, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        M_t = torch.from_numpy(M.astype(np.float32)).to(device)

        x_src = M_t[0, 0] * grid_x + M_t[0, 1] * grid_y + M_t[0, 2]
        y_src = M_t[1, 0] * grid_x + M_t[1, 1] * grid_y + M_t[1, 2]
        x_norm = x_src * (2.0 / crop_size) - 1.0
        y_norm = y_src * (2.0 / crop_size) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

        warped_face = F.grid_sample(fake_t, grid, mode="bilinear", padding_mode="border", align_corners=True)

        cs = int(crop_size)
        mask_src = _get_crop_mask(cs, device)  # cached constant mask (Task 2.2)
        warped_mask = F.grid_sample(mask_src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        k = 41
        warped_mask = F.avg_pool2d(warped_mask, kernel_size=k, stride=1, padding=k // 2)
        warped_mask = F.avg_pool2d(warped_mask, kernel_size=k, stride=1, padding=k // 2)

        opacity = getattr(modules.globals, "opacity", 1.0)
        if opacity < 1.0:
            warped_mask = warped_mask * opacity

        result_roi = warped_face * warped_mask + target_t * (1.0 - warped_mask)
        temp_frame[y0:y1, x0:x1] = result_roi.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
        return temp_frame
```

Delete the now-unused `_get_meshgrid` full-frame path only if nothing else references it (grep first); otherwise leave it.

- [ ] **Step 3: Run the harness and compare against baseline**

```bash
python bench.py --source path/to/face.jpg --target path/to/scene.jpg --iters 100 --preset normal
```
Expected: the `swap` stage mean ms drops substantially versus the Phase 1 number (ROI is ~300x300 vs 1920x1080, roughly an order of magnitude fewer pixels warped/blended, plus far less PCIe traffic). Record under "Phase 2" in `baseline-metrics.md`. If you have a multi-face target, also rerun it — the multi-face gain should be even larger.

- [ ] **Step 4: Visual correctness is critical here**

Open `bench_out.jpg`. The swap must look identical to the Phase 1 output (same blend, same feathering) with no visible box seam around the face. If a seam appears, increase `pad` (the feather blur is reaching the ROI edge). Confirm before committing.

- [ ] **Step 5: Commit**

```bash
git add modules/processors/frame/face_swapper.py modules/live_pipeline.py docs/superpowers/plans/baseline-metrics.md
git commit -m "perf: ROI-clip GPU warp/blend and transfer only the face region"
```

### Task 2.2: Cache the constant crop mask

**Files:**
- Modify: `modules/processors/frame/face_swapper.py` (add `_get_crop_mask`, referenced by Task 2.1)

**Interfaces:**
- Produces: `_get_crop_mask(cs: int, device) -> torch.Tensor` returning a cached `(1,1,cs,cs)` mask (ones with an 8px zeroed border), built once per `(cs, device)`.

- [ ] **Step 1: Add the cached-mask helper**

Near the existing grid cache in `face_swapper.py`, add:
```python
_CROP_MASK_CACHE = {}


def _get_crop_mask(cs: int, device):
    """Constant (1,1,cs,cs) mask: ones with an 8px zeroed border. Cached."""
    import torch
    key = (cs, str(device))
    m = _CROP_MASK_CACHE.get(key)
    if m is None:
        m = torch.ones(1, 1, cs, cs, device=device)
        b = 8
        m[:, :, :b, :] = 0
        m[:, :, -b:, :] = 0
        m[:, :, :, :b] = 0
        m[:, :, :, -b:] = 0
        _CROP_MASK_CACHE[key] = m
    return m
```
This is already referenced by the Task 2.1 code (`mask_src = _get_crop_mask(cs, device)`), replacing the per-call `torch.ones` + border writes.

- [ ] **Step 2: Run the harness, confirm no regression**

```bash
python bench.py --source path/to/face.jpg --target path/to/scene.jpg --iters 100 --preset normal
```
Expected: swap stage equal or marginally faster than Task 2.1; `bench_out.jpg` unchanged.

- [ ] **Step 3: Commit**

```bash
git add modules/processors/frame/face_swapper.py
git commit -m "perf: cache the constant crop mask instead of rebuilding per call"
```

---

## Phase 3: Inference and enhancer efficiency

### Task 3.1: Enhancers use detection-only face lookup

**Files:**
- Modify: `modules/processors/frame/face_enhancer_gpen256.py:86` (`get_one_face` -> `detect_one_face`)
- Modify: `modules/processors/frame/face_enhancer_gpen512.py:86` (same)
- Modify: `modules/processors/frame/face_enhancer.py:265` (`get_many_faces` -> `detect_many_faces`)

**Interfaces:**
- Consumes: `detect_one_face`, `detect_many_faces` from `modules.face_analyser` (already exist, documented "~2x faster", detection-only).
- Produces: enhancers align using landmarks from the detection-only analyser; no ArcFace recognition computed per frame.

- [ ] **Step 1: Swap the import/calls**

In each file, import the detect-only variant and replace the call. GPEN 256/512: `from modules.face_analyser import detect_one_face` and call `detect_one_face(...)` where `get_one_face(...)` was used. GFPGAN: `detect_many_faces(...)` in place of `get_many_faces(...)`. Note: the live pipeline already passes detected faces into the enhancers, so this only affects any path that re-detects — keep the behavior identical, just detection-only.

- [ ] **Step 2: Enable an enhancer in the harness and measure**

Temporarily benchmark with an enhancer on (only if you have the model). Run with `--preset high` (GFPGAN) or set the GPEN flag. Expected: enhancer face-lookup cost drops; `bench_out.jpg` still enhanced correctly.

- [ ] **Step 3: Commit**

```bash
git add modules/processors/frame/face_enhancer_gpen256.py modules/processors/frame/face_enhancer_gpen512.py modules/processors/frame/face_enhancer.py
git commit -m "perf: enhancers use detection-only analyser (skip unused ArcFace)"
```

### Task 3.2: GFPGAN paste-back caches the feather mask and blends ROI-only

**Files:**
- Modify: `modules/processors/frame/face_enhancer.py:184-216` (`_paste_back`)

**Interfaces:**
- Consumes: the GPEN pattern in `modules/processors/frame/_onnx_enhancer.py:27-37` (`_get_feathered_mask`) and `:157-170` (ROI-only float blend) as the reference implementation.
- Produces: `_paste_back` reuses a size-keyed cached feather mask and converts only the ROI to float, mirroring the GPEN path.

- [ ] **Step 1: Port the cached-mask + ROI-blend approach**

Replace the per-face mask rebuild and full-frame float conversion in `face_enhancer.py._paste_back` with the same size-keyed mask cache and ROI-scoped float blend used by `_onnx_enhancer.py`. Do not invent a new approach — copy the GPEN structure so both enhancer families match.

- [ ] **Step 2: Measure with GFPGAN enabled (`--preset high`)**

```bash
python bench.py --source path/to/face.jpg --target path/to/scene.jpg --iters 100 --preset high
```
Expected: `post`/enhancer stage faster; `bench_out.jpg` enhancement visually identical.

- [ ] **Step 3: Commit**

```bash
git add modules/processors/frame/face_enhancer.py
git commit -m "perf: GFPGAN paste-back caches feather mask, blends ROI only"
```

### Task 3.3 (stretch): Warmup enhancer sessions at real input size

**Files:**
- Modify: `modules/processors/frame/_onnx_enhancer.py:55-58`

- [ ] **Step 1:** In `warmup_session`, when a dim is dynamic use the known static `INPUT_SIZE` (256/512/1024) instead of `1`, so the first real frame does not trigger a fresh cuDNN algo search.
- [ ] **Step 2:** Measure first-frame latency with an enhancer on; the first post-warmup frame should no longer stall.
- [ ] **Step 3:** Commit: `perf: warm up enhancers at real input resolution`.

---

## Phase 4: Capture and output latency

### Task 4.1: Freshest-frame rate limiter (pace, then pull newest)

**Files:**
- Modify: `modules/server.py:611-633` (`_frame_loop`)

**Interfaces:**
- Consumes: `self._pipeline.get_processed_frame()` (non-blocking, returns newest available or None).
- Produces: the loop sleeps to pace first, then pulls the freshest frame right before encoding, so no held frame goes stale while newer ones are dropped.

- [ ] **Step 1: Reorder to pace-then-pull**

Rewrite `_frame_loop` so the pacing sleep happens before the frame fetch:
```python
    async def _frame_loop(self) -> None:
        """Grab the freshest processed frame and send as binary WebSocket messages."""
        target_interval = 1.0 / 30
        last_send = 0.0
        while self._streaming:
            now = asyncio.get_event_loop().time()
            elapsed = now - last_send
            if elapsed < target_interval:
                await asyncio.sleep(target_interval - elapsed)

            frame = self._pipeline.get_processed_frame()  # newest available now
            if frame is None:
                await asyncio.sleep(0.002)
                continue

            self._vcam_latest_frame = frame
            jpeg = await asyncio.to_thread(encode_frame_jpeg, frame)
            if jpeg:
                websockets.broadcast(self.clients, jpeg)
                last_send = asyncio.get_event_loop().time()
```

- [ ] **Step 2: Manual latency check**

Start the server, connect the UI (or a WS client), wave a hand in front of the camera, and confirm preview responsiveness improved (the held-frame delay is gone). There is no harness path for this; verify by observation.

- [ ] **Step 3: Commit**

```bash
git add modules/server.py
git commit -m "perf: pace before pulling so the freshest frame is sent, not a stale one"
```

### Task 4.2: Set capture buffer size to 1 and order FOURCC before resolution

**Files:**
- Modify: `modules/live_pipeline.py:186-190`
- Modify: `modules/video_capture.py:59-61`

- [ ] **Step 1:** In the capture setup, set `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` right after opening, and set `CAP_PROP_FOURCC` to MJPG BEFORE width/height/fps (several backends ignore a format change applied after resolution). Apply the same in `video_capture.py`.
- [ ] **Step 2:** Start the server, confirm capture still works and latency under GPU load is lower (older frames no longer accumulate in the driver buffer).
- [ ] **Step 3:** Commit: `perf: CAP_PROP_BUFFERSIZE=1 and FOURCC before resolution`.

### Task 4.3: Open the camera once; enumerate without full-opening devices

**Files:**
- Modify: `modules/server.py:60-81` (`enumerate_cameras`)
- Modify: `modules/server.py:462-486` (drop the extra `check_camera` probe)

- [ ] **Step 1:** Replace `enumerate_cameras` device probing with `pygrabber`'s `FilterGraph().get_input_devices()` (already imported in `video_capture.py:9`) to list device names instantly without opening each index. Keep a fallback to the current probe if pygrabber is unavailable.
- [ ] **Step 2:** In the start path, remove the separate `check_camera` open/release probe and let `_capture_loop` open the device exactly once, reporting failure via the existing status/error message channel.
- [ ] **Step 3:** Measure start time: from "start" click to first preview frame should drop by seconds (no triple-open). Verify by observation.
- [ ] **Step 4:** Commit: `perf: open camera once and enumerate via pygrabber`.

### Task 4.4: Output virtual camera in BGR (drop per-frame color convert)

**Files:**
- Modify: `modules/server.py:549` and `:601`

- [ ] **Step 1:** Create the vcam as `pyvirtualcam.Camera(width=1920, height=1080, fps=30, fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False)` and change the send at line 601 to `self._vcam.send(vcam_frame)` (no `cv2.cvtColor`). Ensure `pyvirtualcam` is imported such that `PixelFormat` is accessible.
- [ ] **Step 2:** Enable the virtual camera, confirm colors are correct in a consumer app (OBS/Zoom) and the per-frame BGR2RGB copy is gone.
- [ ] **Step 3:** Commit: `perf: send virtual camera frames as BGR, no per-frame convert`.

### Task 4.5: Skip the pre-encode resize when the frame is already small enough

**Files:**
- Modify: `modules/server.py:165-174` (`encode_frame_jpeg`)

- [ ] **Step 1:** Guard the `cv2.resize` so it only runs when `frame.shape[1] > max_width`; otherwise encode directly.
- [ ] **Step 2:** Confirm preview still renders and the encode path is unchanged for large frames.
- [ ] **Step 3:** Commit: `perf: skip no-op resize before JPEG encode`.

---

## Phase 5: Delete dead code

### Task 5.1: Remove the dead duplicate masking module

**Files:**
- Delete: `modules/processors/frame/face_masking.py`

- [ ] **Step 1: Confirm nothing imports it**

Run:
```bash
grep -rn "face_masking" modules/ --include=*.py
```
Expected: matches only inside `face_masking.py` itself (no importers). If any importer exists, STOP and reassess.

- [ ] **Step 2: Delete and verify the app still imports**

```bash
git rm modules/processors/frame/face_masking.py
python -c "import modules.processors.frame.face_swapper; import modules.live_pipeline; print('imports ok')"
```
Expected: `imports ok`.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove dead duplicate face_masking module"
```

### Task 5.2: Remove the broken adaptive-detection function

**Files:**
- Modify: `modules/processors/frame/face_swapper.py:401-433` (`get_faces_optimized`)

- [ ] **Step 1: Confirm it is never called**

Run:
```bash
grep -rn "get_faces_optimized\|LAST_DETECTION_TIME\|DETECTION_INTERVAL\|FACE_DETECTION_CACHE" modules/ --include=*.py
```
Expected: matches only the definition of `get_faces_optimized` and its own body (it references globals defined nowhere, so it would `NameError` if called).

- [ ] **Step 2: Delete the function**

Remove `get_faces_optimized` entirely (dead + broken). Do not attempt to "fix" it — the live pipeline's thread decoupling + coast already handle detection rate.

- [ ] **Step 3: Verify imports and harness still work**

```bash
python -c "import modules.processors.frame.face_swapper; print('ok')"
python bench.py --source path/to/face.jpg --target path/to/scene.jpg --iters 20 --preset normal
```
Expected: `ok`, harness runs, `bench_out.jpg` correct.

- [ ] **Step 4: Commit**

```bash
git add modules/processors/frame/face_swapper.py
git commit -m "chore: remove broken dead get_faces_optimized"
```

---

## Final verification

- [ ] Run the harness for both presets one final time and paste results into a "Final" section of `baseline-metrics.md`.
- [ ] Confirm the total pipeline time and effective FPS improved versus the Phase 0 baseline, and that `bench_out.jpg` is correct for each preset.
- [ ] Start the full server + UI once and confirm live swap works end to end (camera in, preview out, virtual camera out), with lower latency than baseline.
- [ ] Commit `baseline-metrics.md`.

## Self-review notes (author checklist, already applied)

- **Spec coverage:** Every audit finding maps to a task — #1 (Task 2.1/2.2), #2 provider options (Task 1.1), #3 GpuProcessor download (Task 1.2), #4 stale frame (Task 4.1), #5 buffersize (Task 4.2), #6 camera open/enum (Task 4.3), #7 detect-only enhancers (Task 3.1), #8 IO/fallback (Task 1.1 warnings; IO binding intentionally deferred as it is high-effort/low-certainty until baseline shows inference-bound), #9 GFPGAN mask (Task 3.2), #11 vcam BGR (Task 4.4), #12 encode resize (Task 4.5), dead code (Phase 5). Enhancer warmup is Task 3.3 (stretch).
- **Deferred deliberately:** ONNX IO binding and OpenCV-CUDA resurrection are NOT in this plan — IO binding is only worth it if Phase 0/1 numbers show inference dominating after CUDA is on, and OpenCV-CUDA requires a custom Windows build that conflicts with the "consolidate on torch" direction. Revisit only if the final numbers justify it.
- **Type consistency:** `_get_crop_mask(cs, device)` is defined in Task 2.2 and consumed in Task 2.1; `cuda_provider_options()`/`providers_with_options()` defined in Task 1.1 and used in the same task; `detect_one_face`/`detect_many_faces` already exist in `face_analyser.py`.
- **Ordering rationale:** Phase 0 must precede everything (no valid measurement otherwise). Phase 1 before Phase 2 so the ROI rewrite is measured on a real GPU session, not a CPU fallback.
