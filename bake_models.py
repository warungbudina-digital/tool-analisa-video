"""Unduh model saat build image (WITH_ML=1) agar runtime OFFLINE.
- CLIP via HF hub -> cache HF_HOME (dijalankan dgn *_OFFLINE=0, lihat Dockerfile).
- openai-whisper via unduhan sendiri -> download_root khusus."""
import os

os.environ.setdefault("CLIP_MODEL", "openai/clip-vit-base-patch32")
WHISPER_DIR = os.getenv("WHISPER_DIR", "/app/models_cache/whisper")

import whisper  # noqa: E402
from transformers import CLIPModel, CLIPProcessor  # noqa: E402

whisper.load_model("base", download_root=WHISPER_DIR)

clip = os.environ["CLIP_MODEL"]
CLIPModel.from_pretrained(clip)
CLIPProcessor.from_pretrained(clip)
print("models baked:", clip, "+ openai-whisper base")
