"""Unduh model HF ke HF_HOME saat build image (WITH_ML=1) agar runtime OFFLINE.
Dijalankan dengan HF_HUB_OFFLINE=0/TRANSFORMERS_OFFLINE=0 (lihat Dockerfile)."""
import os

os.environ.setdefault("CLIP_MODEL", "openai/clip-vit-base-patch32")

from faster_whisper import WhisperModel  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

# whisper base (int8 CPU) -> cache HF_HOME
WhisperModel("base", device="cpu", compute_type="int8")

clip = os.environ["CLIP_MODEL"]
CLIPModel.from_pretrained(clip)
CLIPProcessor.from_pretrained(clip)
print("models baked:", clip, "+ faster-whisper base")
