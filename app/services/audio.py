from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioAnalysis:
    bpm: float | None = None
    beat_times: list[float] = field(default_factory=list)
    rms_times: list[float] = field(default_factory=list)
    rms_values: list[float] = field(default_factory=list)


def analyze_audio(path: str, has_audio: bool) -> AudioAnalysis:
    """BPM + waktu beat + envelope RMS via librosa (CPU). Aman kalau tak ada audio."""
    if not has_audio:
        return AudioAnalysis()
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(path, sr=22050, mono=True)
        if y.size == 0:
            return AudioAnalysis()

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else None

        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        t = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop)
        return AudioAnalysis(
            bpm=round(bpm, 2) if bpm else None,
            beat_times=[round(float(b), 3) for b in np.atleast_1d(beats)],
            rms_times=[round(float(x), 3) for x in t],
            rms_values=[float(x) for x in rms],
        )
    except Exception:
        return AudioAnalysis()


def beat_sync_for(scene_start: float, scene_end: float, beat_times: list[float],
                  tol: float = 0.18) -> bool:
    """Scene dianggap beat-sync kalau ada beat dekat batas awalnya."""
    for b in beat_times:
        if abs(b - scene_start) <= tol:
            return True
        if b > scene_start + tol:
            break
    return False


def mean_rms(scene_start: float, scene_end: float, times: list[float], values: list[float]) -> float:
    vals = [v for t, v in zip(times, values) if scene_start <= t < scene_end]
    return round(sum(vals) / len(vals), 5) if vals else 0.0
