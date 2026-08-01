from __future__ import annotations

import cv2
import numpy as np


def grab_frames(path: str, times: list[float], max_w: int = 320):
    """Ambil frame (grayscale kecil + BGR kecil) pada daftar detik. Kembalikan list (t, gray, bgr)."""
    cap = cv2.VideoCapture(path)
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0.0) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w > max_w:
                scale = max_w / float(w)
                frame = cv2.resize(frame, (max_w, max(1, int(h * scale))))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.append((t, gray, frame))
    finally:
        cap.release()
    return out


def scene_sample_times(start: float, end: float, n: int = 5) -> list[float]:
    if end <= start:
        return [start]
    step = (end - start) / (n + 1)
    return [round(start + step * (i + 1), 3) for i in range(n)]
