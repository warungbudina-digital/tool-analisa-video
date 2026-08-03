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


class AnalyzeResponse(BaseModel):
    status: str
    analyzer_version: str
    analysis: ViralAnalysis
    timings_ms: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
