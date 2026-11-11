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


def _is_hallucination(text: str, seg: dict) -> bool:
    """True kalau segmen kemungkinan besar halusinasi whisper (audio non-speech /
    musik / diam). Whisper tetap memuntahkan segmen berulang meski tak ada ucapan;
    kita saring pakai metrik per-segmen + penjaga repetisi eksplisit."""
    try:
        if float(seg.get("no_speech_prob", 0.0)) >= 0.6:
            return True
        if float(seg.get("avg_logprob", 0.0)) < -1.0:
            return True
        # compression_ratio tinggi = teks sangat mudah dikompres = berulang (mis. "2.1.1.1...")
        if float(seg.get("compression_ratio", 0.0)) > 2.4:
            return True
    except (TypeError, ValueError):
        pass
    # penjaga repetisi eksplisit (jaring pengaman kalau metrik tak ada)
    toks = text.split()
    if len(toks) >= 6 and len(set(toks)) / len(toks) < 0.3:
        return True
    compact = text.replace(" ", "")
    if len(compact) >= 20 and len(set(compact)) <= 4:
        return True
    return False


def transcribe(path: str) -> tuple[list[SubtitleSegment], SubtitleStyle]:
    model = _load()
    if model is None:
        return [], SubtitleStyle()

    try:
        # fp16=False -> CPU; condition_on_previous_text=False menekan loop halusinasi;
        # threshold no_speech/logprob/compression memicu fallback whisper pada non-speech.
        result = model.transcribe(
            path,
            language="id",
            fp16=False,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    except Exception:
        # mis. video tanpa stream audio -> whisper.load_audio (ffmpeg) gagal.
        # Degrade anggun, jangan jatuhkan seluruh /analyze.
        return [], SubtitleStyle()

    segs: list[SubtitleSegment] = []
    total_words = 0
    total_dur = 0.0
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if not text or _is_hallucination(text, s):
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
