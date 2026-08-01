# viral-analyzer

Tool container **FastAPI + ffmpeg + heuristik + model lokal HF** untuk menganalisa
video (short-form) menjadi IR `ViralAnalysis` — **output-nya identik dengan input
`viral_editor`**, sehingga hasil analisa bisa diterjemahkan jadi template VN.

Berjalan di **Cloud Shell** (CPU-only, offline). Tidak memakai Gemini / API luar.
akses-vps hanya jadi orchestrator ringan (ambil blob dari DB-VPS → alirkan ke sini →
hapus blob setelah dianalisa).

## Endpoint
- `GET /healthz` — status + apakah lapisan ASR/semantic aktif.
- `POST /analyze` — `{ "video_path": "/data/input/x.mp4" }` → `ViralAnalysis` + timings.

`video_path` WAJIB di bawah `/data/input` (validasi anti path-traversal).

## Dua lapisan
| Lapisan | Isi | Build |
|---|---|---|
| **Core (V1, default)** | scene detect, motion/kamera, transisi, BPM/beat-sync, wajah/framing, hook_strength (heuristik) | `WITH_ML=0` (ringan) |
| **ML (V2)** | subtitle via **faster-whisper**, `semantic`/`emotion`/`meme` via **CLIP zero-shot** | `WITH_ML=1` (model di-bake ke image, runtime OFFLINE) |

Tanpa lapisan ML, field `semantic/emotion/meme` = `unknown` dan `subtitle_segments` kosong
(status respons `deterministic_only`).

## Jalankan
```bash
# V1 core (cepat)
docker compose -f docker-compose.render-worker.yml build
docker compose -f docker-compose.render-worker.yml up -d
# V2 dengan model HF (unduh+bake saat build, butuh disk + waktu)
WITH_ML=1 docker compose -f docker-compose.render-worker.yml build
```

## Aturan stabilitas (Cloud Shell ephemeral, disk ketat)
- Model **di-bake ke image** (bukan unduh runtime) → offline, tahan VM-recycle sesi.
- Satu video pada satu waktu; bersihkan `/data/input` setelah dipakai.
- Blob video hidup di DB-VPS `media.video_ingest`, DIHAPUS setelah `media.video_analysis` terisi.
