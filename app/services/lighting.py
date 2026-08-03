from __future__ import annotations

import cv2
import numpy as np

from app.services.frames import grab_frames, scene_sample_times


def analyze_lighting(path: str, start: float, end: float) -> dict:
    """Statistik PENCAHAYAAN per-scene dari beberapa frame BGR sampel.

    Deterministik (tanpa ML). Untuk dipetakan ke VN Adjust:
      brightness -> Kecerahan/Eksposur, contrast -> Kontras,
      warmth -> Suhu (warna), saturation -> Saturasi.
    Nilai numerik ternormalisasi + label heuristik.
    """
    result = {
        "brightness": 0.0, "contrast": 0.0, "warmth": 0.0, "saturation": 0.0,
        "brightness_label": "normal", "contrast_label": "normal",
        "temperature_label": "neutral",
    }
    frames = grab_frames(path, scene_sample_times(start, end, n=3), max_w=320)
    bgrs = [f[2] for f in frames]
    if not bgrs:
        return result

    lums, stds, warms, sats = [], [], [], []
    for bgr in bgrs:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lums.append(float(np.mean(gray)))          # 0..255
        stds.append(float(np.std(gray)))           # sebar luminance = kontras
        b, g, r = cv2.split(bgr)                    # OpenCV = BGR
        warms.append(float(np.mean(r)) - float(np.mean(b)))  # R-B: + hangat, - dingin
        sats.append(float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1])))

    lum = sum(lums) / len(lums)
    std = sum(stds) / len(stds)
    warm = sum(warms) / len(warms)
    sat = sum(sats) / len(sats)

    result["brightness"] = round(lum / 255.0, 3)
    result["contrast"] = round(min(std / 128.0, 1.0), 3)
    result["warmth"] = round(max(-1.0, min(warm / 255.0, 1.0)), 3)
    result["saturation"] = round(sat / 255.0, 3)
    result["brightness_label"] = "dark" if lum < 70 else "bright" if lum > 165 else "normal"
    result["contrast_label"] = "low" if std < 35 else "high" if std > 70 else "normal"
    result["temperature_label"] = "warm" if warm > 12 else "cool" if warm < -12 else "neutral"
    return result
