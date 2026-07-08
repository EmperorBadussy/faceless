"""Headless benchmark + correctness check for the FACELESS hot path.

Runs detection + GPU swap + post-processing on fixed input images and reports
per-stage timing. This is the measuring instrument for the optimization plan.

Usage:
    python bench.py --source face.jpg --target scene.jpg --iters 100 --preset high
"""
import argparse
import time
import torch  # import first: onnxruntime then reuses torch's CUDA 12 / cuDNN 9 DLLs
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

    from modules.processors.frame.face_swapper import get_face_swapper
    swp = get_face_swapper()
    sess = getattr(swp, "session", None)
    if sess is not None:
        print(f"[bench] swap session providers = {sess.get_providers()}")

    stages = {"detect": [], "swap": [], "post": [], "total": []}
    out = None
    faces = []
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
