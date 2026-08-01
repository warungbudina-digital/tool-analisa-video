from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass
class VideoMeta:
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool


def probe(path: str) -> VideoMeta:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError("no_video_stream")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    # fps dari r_frame_rate "num/den"
    fps = 0.0
    rate = v.get("r_frame_rate", "0/1")
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0

    duration = float(data.get("format", {}).get("duration") or v.get("duration") or 0.0)
    return VideoMeta(
        duration=round(duration, 3),
        fps=round(fps, 3),
        width=int(v.get("width") or 0),
        height=int(v.get("height") or 0),
        has_audio=has_audio,
    )
