from __future__ import annotations

import os
import time

from app.models import (AnalyzeResponse, FramingData, LightingData, SceneAnalysis,
                         SubtitleStyle, ViralAnalysis)
from app.services import (asr, audio, faces, hook as hook_mod, lighting, motion, probe,
                          scene, semantic)
from app.services.frames import grab_frames

ANALYZER_VERSION = "0.1.0"


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _aspect_ratio(w: int, h: int) -> str:
    """Label rasio dari dimensi. Kosong kalau tak diketahui."""
    if not w or not h:
        return ""
    r = w / h
    if r < 0.85:
        return "9:16"   # vertikal (reel/shorts)
    if r < 1.15:
        return "1:1"    # persegi
    return "16:9"       # lanskap


def analyze(video_path: str, run_asr: bool | None = None,
            run_semantic: bool | None = None, language: str | None = None) -> AnalyzeResponse:
    warnings: list[str] = []
    timings: dict[str, int] = {}

    t = time.perf_counter()
    meta = probe.probe(video_path)
    timings["probe"] = _ms(t)

    t = time.perf_counter()
    aud = audio.analyze_audio(video_path, meta.has_audio)
    timings["audio"] = _ms(t)
    if meta.has_audio and aud.bpm is None:
        warnings.append("audio_analysis_incomplete")
    rms_max = max(aud.rms_values) if aud.rms_values else 0.0

    t = time.perf_counter()
    scenes = scene.detect_scenes(video_path, meta.duration)
    timings["scene_detect"] = _ms(t)

    do_asr = asr.available() if run_asr is None else (run_asr and asr.available())
    do_sem = semantic.available() if run_semantic is None else (run_semantic and semantic.available())
    if run_asr and not asr.available():
        warnings.append("asr_requested_but_unavailable")
    if run_semantic and not semantic.available():
        warnings.append("semantic_requested_but_unavailable")

    t = time.perf_counter()
    scene_out: list[SceneAnalysis] = []
    prev_end = 0.0
    for i, sc in enumerate(scenes):
        mot = motion.analyze_motion(video_path, sc.start, sc.end)
        trans = "cut" if i == 0 else motion.transition_before(video_path, prev_end, sc.start)
        fr = faces.analyze_framing(video_path, sc.start, sc.end)
        lig = lighting.analyze_lighting(video_path, sc.start, sc.end)
        rms = audio.mean_rms(sc.start, sc.end, aud.rms_times, aud.rms_values)

        sem = {"semantic": "unknown", "emotion": "unknown", "meme": "unknown"}
        if do_sem:
            mid = grab_frames(video_path, [(sc.start + sc.end) / 2.0], max_w=480)
            if mid:
                sem = semantic.classify_scene(mid[0][2])

        hook = hook_mod.hook_strength(
            index=i, duration=sc.duration, motion=mot["motion"], rms=rms,
            has_face=bool(fr["faces"]), rms_max=rms_max,
        )

        scene_out.append(SceneAnalysis(
            start=sc.start, end=sc.end, duration=sc.duration,
            semantic=sem["semantic"], emotion=sem["emotion"], meme=sem["meme"],
            motion=mot["motion"], camera_movement=mot["camera_movement"], transition=trans,
            beat_sync=audio.beat_sync_for(sc.start, sc.end, aud.beat_times),
            framing=FramingData(faces=fr["faces"], speaker_position=fr["speaker_position"]),
            lighting=LightingData(**lig),
            hook_strength=hook,
        ))
        prev_end = sc.end
    timings["scenes_loop"] = _ms(t)

    subtitle_segments = []
    subtitle_style = SubtitleStyle()
    language_used = None
    if do_asr and meta.has_audio:
        t = time.perf_counter()
        subtitle_segments, subtitle_style, language_used = asr.transcribe(video_path, language=language)
        timings["asr"] = _ms(t)
    elif do_asr and not meta.has_audio:
        warnings.append("asr_skipped_no_audio")

    analysis = ViralAnalysis(
        video=os.path.basename(video_path),
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        duration_sec=meta.duration,
        aspect_ratio=_aspect_ratio(meta.width, meta.height),
        bpm=aud.bpm,
        subtitle_style=subtitle_style,
        subtitle_segments=subtitle_segments,
        scene_analysis=scene_out,
    )

    status = "ok"
    if not do_asr and not do_sem:
        status = "deterministic_only"  # V1: field semantic/emotion/subtitle di-skip
    return AnalyzeResponse(status=status, analyzer_version=ANALYZER_VERSION,
                           analysis=analysis, timings_ms=timings, warnings=warnings,
                           language_used=language_used)
