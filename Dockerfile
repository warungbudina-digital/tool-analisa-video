FROM python:3.12-slim

WORKDIR /app

# Dependensi sistem: ffmpeg (probe/decode + backend librosa), OpenCV, libsndfile (soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 0 = core deterministik (ringan); 1 = + model HF (whisper/CLIP) di-bake
ARG WITH_ML=0
ENV HF_HOME=/app/models_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Lapisan model (V2): install deps ML + unduh model SEKARANG agar runtime OFFLINE
RUN if [ "$WITH_ML" = "1" ]; then \
        pip install --no-cache-dir -r requirements-ml.txt && \
        HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PY' ; \
from faster_whisper import WhisperModel
from transformers import CLIPModel, CLIPProcessor
# unduh ke HF_HOME (dipakai runtime) - jangan set download_root khusus
WhisperModel("base", device="cpu", compute_type="int8")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("models baked")
PY
    fi

COPY app ./app

EXPOSE 9021
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9021"]
