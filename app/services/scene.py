from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scene:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


def detect_scenes(path: str, duration: float, threshold: float = 27.0) -> list[Scene]:
    """Deteksi batas scene (content-aware) via PySceneDetect.
    Fallback: satu scene penuh kalau deteksi gagal / video sangat pendek."""
    try:
        from scenedetect import detect, ContentDetector

        raw = detect(path, ContentDetector(threshold=threshold))
        scenes = [
            Scene(round(s.get_seconds(), 3), round(e.get_seconds(), 3))
            for s, e in raw
            if e.get_seconds() > s.get_seconds()
        ]
    except Exception:
        scenes = []

    if not scenes:
        scenes = [Scene(0.0, round(duration, 3))]
    return scenes
