# Deployment Guide

DocForensics ships as a **single Docker container**: the FastAPI backend serves
both the REST API (`/api/*`) and the pre-built React frontend (everything else),
so there's one image, one URL, and no CORS to configure.

> **Heads up — this is a heavy ML app.** It pulls in PyTorch, Transformers and
> OpenCV, and downloads a ~350 MB Hugging Face model on first use. Plan for
> **~2 GB RAM** and a few-GB image. Free tiers with 512 MB RAM (e.g. Render Free)
> are **not** enough — use Hugging Face Spaces (free, 16 GB) or a paid small instance.

---

## 0. Prerequisites

The trained model `model/checkpoints/best.pt` must be present (it's committed).
If you ever need to regenerate it:

```bash
python -m synth.generate_balanced --per_class 250
python -c "from model.train import train; train()"
```

---

## 1. Run locally with Docker (recommended first step)

```bash
docker compose up --build
# open http://localhost:8000
```

That builds the frontend, bundles it with the API, and serves the whole app on
port 8000.

---

## 2. Deploy to Hugging Face Spaces  ⭐ best free option for this stack

Spaces gives you 16 GB RAM for free and is built for ML apps.

1. Create a new Space → **SDK: Docker** → Blank.
2. Push this repo to the Space's git remote (include `best.pt`).
3. Add this front-matter to the top of the Space's `README.md`:

   ```yaml
   ---
   title: DocForensics
   emoji: 🔍
   colorFrom: indigo
   colorTo: purple
   sdk: docker
   app_port: 8000
   ---
   ```

4. The Space builds the `Dockerfile` automatically and your app goes live at
   `https://<user>-<space>.hf.space`.

No env vars are required — the frontend is same-origin.

---

## 3. Deploy to Render (Docker web service)

1. New → **Web Service** → connect the repo.
2. Runtime: **Docker** (Render auto-detects the `Dockerfile`).
3. Instance type: **Standard (2 GB)** or larger — *not* Free.
4. Render sets `$PORT` automatically; the container already respects it.
5. Deploy. URL: `https://<service>.onrender.com`.

---

## 4. Deploy to Railway / Fly.io

**Railway**: New Project → Deploy from repo → it builds the `Dockerfile`. Set the
service to ≥1 GB. Railway injects `$PORT`.

**Fly.io**:
```bash
fly launch --dockerfile Dockerfile   # accept detected settings
fly scale memory 2048
fly deploy
```

---

## 5. Split deployment (frontend and backend separately)

If you'd rather host the static frontend on a CDN (Vercel/Netlify) and the API
elsewhere:

**Backend** — deploy the container (steps 2–4) but you only need the API.

**Frontend** — on Vercel/Netlify:
- Build command: `npm run build`  ·  Output dir: `dist`  ·  Root: `frontend`
- Set env var `VITE_API_URL = https://<your-backend>/api`
- On the backend, set `ALLOWED_ORIGINS = https://<your-frontend-domain>`

---

## Production notes

- **Lock down CORS**: set `ALLOWED_ORIGINS` to your real domain(s) instead of `*`.
- **First request is slow**: it downloads the AI-detector model and warms the
  CNN. Subsequent requests are fast (both are cached in-process).
- **Low-RAM hosts**: set `DOCFORENSICS_DISABLE_AI_MODEL=1` to skip the ~350 MB
  AI-classifier download. The AI detector then uses only its frequency/EXIF
  signals, cutting memory use substantially.
- **Max upload size** is 20 MB (`MAX_FILE_MB` in `core/config.py`).
- **Health check**: `GET /api/health` → `{"status": "ok"}` — wire this into your
  host's health probe.
- **OCR**: PaddleOCR isn't bundled (too heavy); the app falls back to Tesseract
  (installed in the image) and degrades gracefully if neither is available.
