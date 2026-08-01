from __future__ import annotations

import os

# Label kandidat utk CLIP zero-shot (selaras dgn nilai yg dipakai viral_editor/planner)
SEMANTIC_LABELS = [
    "youtube talking head", "person talking", "reaction face", "gameplay footage",
    "product showcase", "food closeup", "outdoor scenery", "dancing", "text on screen",
    "tutorial screen recording", "street interview", "pet animal", "car vehicle",
]
EMOTION_LABELS = ["neutral face", "happy face", "surprised face", "angry face",
                  "sad face", "excited face"]
MEME_LABELS = ["reaction meme", "text meme", "not a meme"]

_clip = None
_proc = None


def _load_clip():
    global _clip, _proc
    if _clip is not None:
        return _clip, _proc
    try:
        import torch  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor

        name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
        _clip = CLIPModel.from_pretrained(name)
        _clip.eval()
        _proc = CLIPProcessor.from_pretrained(name)
        return _clip, _proc
    except Exception:
        return None, None


def available() -> bool:
    m, _ = _load_clip()
    return m is not None


def _zero_shot(image_bgr, labels: list[str]) -> str:
    import numpy as np
    import torch
    from PIL import Image

    model, proc = _load_clip()
    if model is None:
        return "unknown"
    rgb = image_bgr[:, :, ::-1]
    img = Image.fromarray(np.ascontiguousarray(rgb))
    inputs = proc(text=labels, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
    probs = out.logits_per_image.softmax(dim=1)[0]
    return labels[int(probs.argmax())]


def classify_scene(image_bgr) -> dict:
    """Kembalikan {semantic, emotion, meme} via CLIP zero-shot. 'unknown' kalau model absen."""
    if image_bgr is None or not available():
        return {"semantic": "unknown", "emotion": "unknown", "meme": "unknown"}
    try:
        return {
            "semantic": _zero_shot(image_bgr, SEMANTIC_LABELS),
            "emotion": _zero_shot(image_bgr, EMOTION_LABELS),
            "meme": _zero_shot(image_bgr, MEME_LABELS),
        }
    except Exception:
        return {"semantic": "unknown", "emotion": "unknown", "meme": "unknown"}
