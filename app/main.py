from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.models import AnalyzeRequest, AnalyzeResponse
from app.services import asr, semantic
from app.services.pipeline import ANALYZER_VERSION, analyze

# Direktori input yang diizinkan (mount volume). Cegah path traversal (pelajaran viral_editor).
INPUT_DIR = Path(os.getenv("VIRAL_ANALYZER_INPUT_DIR", "/data/input")).resolve()

app = FastAPI(title="viral-analyzer", version=ANALYZER_VERSION)


def _safe_path(p: str) -> Path:
    rp = Path(p).resolve()
    if rp != INPUT_DIR and INPUT_DIR not in rp.parents:
        raise HTTPException(status_code=400, detail={"error": "path_outside_input_dir",
                                                     "input_dir": str(INPUT_DIR)})
    if not rp.exists() or not rp.is_file():
        raise HTTPException(status_code=404, detail={"error": "video_not_found", "path": str(rp)})
    return rp


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "service": "viral-analyzer",
        "version": ANALYZER_VERSION,
        "input_dir": str(INPUT_DIR),
        "asr": asr.available(),
        "semantic": semantic.available(),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def do_analyze(req: AnalyzeRequest):
    path = _safe_path(req.video_path)
    try:
        return analyze(str(path), run_asr=req.run_asr, run_semantic=req.run_semantic)
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail={"error": "analyze_failed", "message": str(exc)})
