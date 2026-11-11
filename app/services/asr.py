from __future__ import annotations

import os
import sys

from app.models import SubtitleSegment, SubtitleStyle

_model = None
_MODEL_SIZE = os.getenv("ASR_MODEL", "base")
_MODEL_DIR = os.getenv("WHISPER_DIR", "/app/models_cache/whisper")
_VALID_LANGS = {"id", "en", "auto"}


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


def _resolve_language(explicit: str | None) -> tuple[str | None, str]:
    """Urutan: override per-job eksplisit (sudah divalidasi Literal di API) ->
    fallback env ASR_LANGUAGE -> default "id". "auto" (dari job ATAU env) ->
    None (whisper auto-detect). Nilai tak dikenal (mis. ASR_LANGUAGE di-typo)
    TAK PERNAH lolos diam-diam ke whisper -- whisper raise kalau dikasih kode
    bahasa yg tak dikenalnya, lalu except umum di transcribe() menelan jadi
    subtitle kosong tanpa jejak (kelas kegagalan sama spt
    feedback_kegagalan_menyamar_sukses). Balik (param_whisper, label_dilaporkan)."""
    raw = explicit if explicit else os.getenv("ASR_LANGUAGE", "id")
    raw = (raw or "auto").strip().lower()
    if raw not in _VALID_LANGS:
        print(f"[asr] language='{raw}' tak dikenal, fallback 'id'", file=sys.stderr, flush=True)
        raw = "id"
    return (None if raw == "auto" else raw), raw


_RUN_SHORT_DUR = 1.5  # detik; ambang "pendek" per segmen
# 10 (BUKAN 5): divalidasi thd job-24 nyata -- rentetan halusinasi asli
# (episode 35:12, musik) panjang 33 segmen; rentetan TERPANJANG speech asli
# yg legitim (seruan dramatis "Oh! Oh! No! Ah! Ah!" @10:12) panjang 6. 10 =
# margin aman di 2 sisi (+4 dari 6, -23 dari 33) -- JANGAN turunkan ke <=6
# tanpa validasi ulang, berisiko buang reaksi/dialog cepat yg asli.
_RUN_MIN_LEN = 10


def _mark_hallucination_runs(raw_segments: list[dict]) -> set[int]:
    """Deteksi RENTETAN segmen super-pendek berurutan -- ciri nyata loop
    halusinasi whisper (job-24, episode 35:12-35:38: >20 segmen 0.3-1.3dtk
    berturut-turut, 1-2 token/segmen) yang LOLOS dari _is_hallucination()
    krn metrik itu per-SEGMEN -- tiap segmen sendiri terlalu singkat utk kena
    ambang token/karakternya. Episode speech asli yg cuma rusak transkripsi
    (bkn loop) punya segmen BERDURASI NORMAL (cth job-24 24:32-24:56, 4
    segmen @4dtk) -- TAK kena filter ini, sengaja (lihat memori
    project_viral_analyzer.md: VAD diuji, bukan ucapan asli yg mau dibuang).
    Balik set INDEX (bukan teks, krn teks boleh berisi placeholder kosong)."""
    bad: set[int] = set()
    run_start = None
    for i, s in enumerate(raw_segments):
        dur = float(s.get("end", 0.0)) - float(s.get("start", 0.0))
        if dur < _RUN_SHORT_DUR:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= _RUN_MIN_LEN:
                bad.update(range(run_start, i))
            run_start = None
    if run_start is not None and len(raw_segments) - run_start >= _RUN_MIN_LEN:
        bad.update(range(run_start, len(raw_segments)))
    return bad


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


def transcribe(path: str, language: str | None = None) -> tuple[list[SubtitleSegment], SubtitleStyle, str | None]:
    model = _load()
    if model is None:
        return [], SubtitleStyle(), None

    lang_param, lang_label = _resolve_language(language)
    try:
        # fp16=False -> CPU; condition_on_previous_text=False menekan loop halusinasi;
        # threshold no_speech/logprob/compression memicu fallback whisper pada non-speech.
        result = model.transcribe(
            path,
            language=lang_param,
            fp16=False,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
    except Exception:
        # mis. video tanpa stream audio -> whisper.load_audio (ffmpeg) gagal.
        # Degrade anggun, jangan jatuhkan seluruh /analyze. lang_label = upaya
        # yg DICOBA (bukan konfirmasi whisper berhasil pakai itu).
        return [], SubtitleStyle(), lang_label

    raw_segments = result.get("segments", [])
    run_bad = _mark_hallucination_runs(raw_segments)

    segs: list[SubtitleSegment] = []
    total_words = 0
    total_dur = 0.0
    dropped_run = 0
    for i, s in enumerate(raw_segments):
        text = (s.get("text") or "").strip()
        if not text or _is_hallucination(text, s):
            continue
        if i in run_bad:
            dropped_run += 1
            continue
        start = float(s.get("start", 0.0))
        end = float(s.get("end", 0.0))
        segs.append(SubtitleSegment(text=text, start=round(start, 3), end=round(end, 3)))
        total_words += len(text.split())
        total_dur += max(end - start, 0.0)
    if dropped_run:
        # jejak WAJIB -- jangan pernah buang diam2 (feedback_kegagalan_menyamar_sukses)
        print(f"[asr] buang {dropped_run} segmen (rentetan halusinasi lintas-segmen)",
              file=sys.stderr, flush=True)

    wca = round(total_words / len(segs), 2) if segs else None
    wps = (total_words / total_dur) if total_dur > 0 else 0.0
    speed = "fast" if wps >= 3.0 else "slow" if wps and wps < 1.5 else "medium"
    # result["language"] = bahasa YG BENAR-BENAR whisper pakai (kalau auto-detect,
    # ini hasil deteksinya; kalau dipin, whisper echo balik kode yg sama) -- lebih
    # akurat drpd lang_label (itu cuma niat kita sblm dicoba).
    return segs, SubtitleStyle(speed=speed, word_count_avg=wca), result.get("language", lang_label)
