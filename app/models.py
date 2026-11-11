from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

# ── Output IR — HARUS identik dengan input viral_editor (app/models.py ViralAnalysis) ──


class SubtitleStyle(BaseModel):
    speed: str = "medium"
    word_count_avg: float | None = None


class SubtitleSegment(BaseModel):
    text: str
    start: float
    end: float


class FramingData(BaseModel):
    faces: list[dict[str, Any]] = Field(default_factory=list)
    speaker_position: str = "unknown"


class LightingData(BaseModel):
    brightness: float = 0.0   # 0..1 (mean luminance / 255)
    contrast: float = 0.0     # 0..1 (std luminance / 128)
    warmth: float = 0.0       # signed -1..1 ((mean R - mean B)/255); + hangat, - dingin
    saturation: float = 0.0   # 0..1
    brightness_label: str = "normal"    # dark | normal | bright
    contrast_label: str = "normal"      # low | normal | high
    temperature_label: str = "neutral"  # warm | neutral | cool


class SceneAnalysis(BaseModel):
    start: float
    end: float
    duration: float
    semantic: str = "unknown"
    emotion: str = "unknown"
    meme: str = "unknown"
    motion: float = 0.0
    camera_movement: str = "static"
    transition: str = "cut"
    beat_sync: bool = False
    framing: FramingData = Field(default_factory=FramingData)
    lighting: LightingData = Field(default_factory=LightingData)
    hook_strength: str = "weak_hook"


class ViralAnalysis(BaseModel):
    video: str
    # meta video sumber — dipakai penerjemah IR→VN (rasio = dimensi target reproduksi).
    # Semua opsional/berdefault → aman thd konsumen lama (viral_editor abaikan field ekstra).
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_sec: float = 0.0
    aspect_ratio: str = ""  # "9:16" | "1:1" | "16:9" | "" (kosong = tak diketahui)
    bpm: float | None = None
    subtitle_style: SubtitleStyle = Field(default_factory=SubtitleStyle)
    subtitle_segments: list[SubtitleSegment] = Field(default_factory=list)
    scene_analysis: list[SceneAnalysis] = Field(default_factory=list)


# ── API request/response ──


class AnalyzeRequest(BaseModel):
    # path video di dalam container (di bawah INPUT_DIR, divalidasi di main.py)
    video_path: str
    # override opsional
    run_asr: bool | None = None
    run_semantic: bool | None = None
    # bahasa ASR per-job (2026-11-11): "id"/"en" paksa, "auto" = auto-detect
    # whisper eksplisit, None = tak disebut -> pakai default env ASR_LANGUAGE
    # analyzer (perilaku lama, mundur-kompatibel). Literal -> FastAPI balas 422
    # otomatis kalau nilai di luar daftar (jangan lolos diam-diam ke whisper).
    language: Literal["id", "en", "auto"] | None = None


class AnalyzeResponse(BaseModel):
    status: str
    analyzer_version: str
    analysis: ViralAnalysis
    timings_ms: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    # bahasa YANG BENAR-BENAR dipakai whisper (hasil resolve request/env/auto-
    # detect) -- field TERPISAH dari `analysis`/ViralAnalysis (itu wajib identik
    # persis dgn input viral_editor, lihat komentar di atas) supaya terlihat
    # jelas "diterapkan" vs "diam-diam diabaikan" (pydantic abaikan field tak
    # dikenal secara default -- analyzer versi lama tanpa field ini tak akan error,
    # tapi respons JUGA tak akan punya key ini sama sekali, itu sinyalnya).
    language_used: str | None = None
