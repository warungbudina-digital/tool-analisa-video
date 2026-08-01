from __future__ import annotations

import cv2

from app.services.frames import grab_frames

_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _cascade


def analyze_framing(path: str, start: float, end: float) -> dict:
    """Deteksi wajah (Haar cascade, tanpa unduhan) di frame tengah scene.
    faces = list kotak ternormalisasi; speaker_position dari wajah terbesar."""
    mid = (start + end) / 2.0
    frames = grab_frames(path, [mid], max_w=480)
    faces_out: list[dict] = []
    position = "unknown"
    if not frames:
        return {"faces": faces_out, "speaker_position": position}

    _, gray, _ = frames[0]
    h, w = gray.shape[:2]
    cascade = _get_cascade()
    dets = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(24, 24))

    biggest = None
    for (x, y, fw, fh) in dets:
        box = {
            "x": round(x / w, 4), "y": round(y / h, 4),
            "w": round(fw / w, 4), "h": round(fh / h, 4),
        }
        faces_out.append(box)
        if biggest is None or (fw * fh) > (biggest[2] * biggest[3]):
            biggest = (x, y, fw, fh)

    if biggest is not None:
        cx = (biggest[0] + biggest[2] / 2.0) / w
        position = "left" if cx < 0.38 else "right" if cx > 0.62 else "center"

    return {"faces": faces_out, "speaker_position": position}
