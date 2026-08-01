from __future__ import annotations

import cv2
import numpy as np

from app.services.frames import grab_frames, scene_sample_times


def analyze_motion(path: str, start: float, end: float) -> dict:
    """Estimasi motion (0..10-ish), camera_movement, dan transition-in scene.
    Heuristik murni via frame-diff + optical flow Farneback (ringan, CPU)."""
    times = scene_sample_times(start, end, n=5)
    frames = grab_frames(path, times)
    result = {"motion": 0.0, "camera_movement": "static"}
    if len(frames) < 2:
        return result

    grays = [g for _, g, _ in frames]
    # samakan ukuran
    h = min(g.shape[0] for g in grays)
    w = min(g.shape[1] for g in grays)
    grays = [g[:h, :w] for g in grays]

    # motion = rata-rata beda absolut antar frame berurutan, dinormalisasi ke ~0..10
    diffs = [float(np.mean(cv2.absdiff(grays[i], grays[i + 1]))) for i in range(len(grays) - 1)]
    motion = round(min((sum(diffs) / len(diffs)) / 2.55, 10.0), 3)  # 0..100 -> 0..~39, cap 10

    # camera_movement dari optical flow satu pasang representatif (tengah)
    mid = len(grays) // 2
    a, b = grays[max(0, mid - 1)], grays[mid]
    cam = "static"
    try:
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        mean_dx, mean_dy = float(np.mean(fx)), float(np.mean(fy))
        mag = float(np.mean(np.sqrt(fx * fx + fy * fy)))
        # divergence (zoom): korelasi flow dgn posisi radial
        yy, xx = np.mgrid[0:fy.shape[0], 0:fx.shape[1]].astype(np.float32)
        cx, cy = fx.shape[1] / 2.0, fy.shape[0] / 2.0
        radial = ((xx - cx) * fx + (yy - cy) * fy)
        divergence = float(np.mean(radial))
        translation = (mean_dx ** 2 + mean_dy ** 2) ** 0.5

        if mag < 0.6:
            cam = "static"
        elif mag > 3.0:
            cam = "fast_move"
        elif abs(divergence) > 40.0:
            cam = "zoom"
        elif translation > 0.6:
            cam = "pan"
        else:
            cam = "shake"
    except Exception:
        cam = "static"

    result["motion"] = motion
    result["camera_movement"] = cam
    return result


def transition_before(path: str, prev_end: float, cur_start: float) -> str:
    """Klasifikasi transisi masuk scene: cut (abrupt) vs fade/dissolve (gradual)."""
    if cur_start <= 0.05:
        return "cut"
    eps = 0.12
    frames = grab_frames(path, [cur_start - eps, cur_start, cur_start + eps])
    if len(frames) < 3:
        return "cut"
    grays = [g for _, g, _ in frames]
    h = min(g.shape[0] for g in grays)
    w = min(g.shape[1] for g in grays)
    grays = [g[:h, :w] for g in grays]
    b1 = float(np.mean(grays[0]))
    b2 = float(np.mean(grays[2]))
    d = float(np.mean(cv2.absdiff(grays[0], grays[2])))
    # fade: luminance bergeser besar tapi beda per-piksel "halus" merata
    if (b1 < 18 or b2 < 18) and d > 8:
        return "fade"
    return "cut"
