from __future__ import annotations

import os

from app.models import SubtitleSegment, SubtitleStyle

_model = None
_MODEL_SIZE = os.getenv("ASR_MODEL", "base")
_MODEL_DIR = os.getenv("WHISPER_DIR", "/app/models_cache/whisper")


def _load():
    """Muat openai-whisper sekali. None kalau lib/model tak tersedia (V1 core)."""
    global _model
    if _model is not None:
        return _model
    try:
        import whisper

        _model = whisper.load_model(_MODEL_SIZE, download_root=_MODEL_DIR)
        return _model
    except Exception:
        return None


def available() -> bool:
    return _load() is not None


def transcribe(path: str) -> tuple[list[SubtitleSegment], SubtitleStyle]:
    model = _load()
    if model is None:
        return [], SubtitleStyle()

    result = model.transcribe(path, fp16=False)  # fp16=False -> CPU
    segs: list[SubtitleSegment] = []
    total_words = 0
    total_dur = 0.0
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        segs.append(SubtitleSegment(text=text, start=round(start, 3), end=round(end, 3)))
        total_words += len(text.split())
        total_dur += max(end - start, 0.0)

    wca = round(total_words / len(segs), 2) if segs else None
    wps = (total_words / total_dur) if total_dur > 0 else 0.0
    speed = "fast" if wps >= 3.0 else "slow" if wps and wps < 1.5 else "medium"
    return segs, SubtitleStyle(speed=speed, word_count_avg=wca)
