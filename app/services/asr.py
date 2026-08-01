from __future__ import annotations

import os

from app.models import SubtitleSegment, SubtitleStyle

_model = None
_MODEL_SIZE = os.getenv("ASR_MODEL", "base")


def _load():
    """Muat faster-whisper sekali. None kalau lib/model tak tersedia (V1 core)."""
    global _model
    if _model is not None:
        return _model
    try:
        from faster_whisper import WhisperModel

        _model = WhisperModel(_MODEL_SIZE, device="cpu", compute_type="int8")
        return _model
    except Exception:
        return None


def available() -> bool:
    return _load() is not None


def transcribe(path: str) -> tuple[list[SubtitleSegment], SubtitleStyle]:
    model = _load()
    if model is None:
        return [], SubtitleStyle()

    segments, _info = model.transcribe(path, vad_filter=True)
    segs: list[SubtitleSegment] = []
    total_words = 0
    total_dur = 0.0
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        segs.append(SubtitleSegment(text=text, start=round(s.start, 3), end=round(s.end, 3)))
        total_words += len(text.split())
        total_dur += max(s.end - s.start, 0.0)

    wca = round(total_words / len(segs), 2) if segs else None
    wps = (total_words / total_dur) if total_dur > 0 else 0.0
    speed = "fast" if wps >= 3.0 else "slow" if wps and wps < 1.5 else "medium"
    return segs, SubtitleStyle(speed=speed, word_count_avg=wca)
