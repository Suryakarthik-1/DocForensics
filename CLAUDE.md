# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocForensics is a document tampering and forgery detection system that combines seven classical forensic detectors with a trained two-stream CNN (TamperNet). It analyzes uploaded documents/images and produces a verdict: **AUTHENTIC**, **TAMPERED**, or **AI-GENERATED**, with a heatmap showing suspicious regions.

## Commands

### Development
```bash
# Backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
# Open http://localhost:5173 (proxies /api → :8000)

# Full stack in one container
docker compose up --build
# Open http://localhost:8000
```

### Testing
```bash
# All tests (87 tests)
pytest tests/ -v

# Single test file
pytest tests/unit/test_ela.py -v

# With coverage
pytest tests/ --cov=core --cov=detectors --cov=fusion --cov=model --cov=api
```

### Model Training
```bash
# Generate balanced dataset (genuine + tampered pairs from same sources)
python -m synth.generate_balanced --per_class 250

# Train TamperNet
python -c "from model.train import train; train()"
```

### Docker
```bash
# Build image
docker build -t docforensics .

# Run container
docker run -p 7860:7860 docforensics
```

## Architecture

```
ingestion → preprocessing → OCR → ┌─ 7 forensic detectors ─┐
                                   ├─ TamperNet CNN          ├─ fusion → Verdict + heatmap
                                   └────────────────────────┘
                       FastAPI (REST + serves the React UI)
```

### Key Components

**Core Pipeline** (`core/pipeline.py`)
- `analyze(path)` — entry point, loads document pages, runs detectors per page
- `register_detectors()` — returns list of 7 classical detectors
- `_run_model()` — loads and runs TamperNet CNN if checkpoint exists
- `fuse()` in `fusion/decision.py` — weighted average + strong-signal boost

**Detectors** (`detectors/`) — all implement `Detector.run(img, ctx) -> Detection`
- `ela.py` — Error Level Analysis (re-compression artifacts)
- `noise.py` — Local noise fingerprint mismatches
- `copymove.py` — Cloned/duplicated region detection
- `double_jpeg.py` — Double compression artifacts
- `font_forensics.py` — Text baseline/anti-aliasing inconsistencies
- `metadata.py` — EXIF/PDF metadata anomalies
- `ai_generated.py` — Frequency signature + Hugging Face classifier

**Fusion** (`fusion/decision.py`)
- Weighted average using `FUSION_WEIGHTS` from `core/config.py`
- Strong-signal boost: `max(weighted, 0.55 * strongest + 0.45 * weighted)`
- Reliable detectors (CNN, AI-classifier, ELA) get highest weights
- Heatmaps merged with same weights, resized to common resolution

**Model** (`model/`)
- `architecture.py` — TamperNet: two-stream U-Net (RGB + SRM noise residual)
- `inference.py` — `load_model()`, `predict()` returning confidence + heatmap
- `dataset.py` — Balanced dataset loader (same source docs in both classes)
- `train.py` — Training loop with BCE + Dice loss, early stopping
- Checkpoint at `model/checkpoints/best.pt` (~5 MB fp16)

**API** (`api/`)
- `main.py` — FastAPI app, serves React frontend from `frontend/dist/`
- `routes.py` — `POST /api/analyze` (multipart upload), `GET /api/health`
- `schemas.py` — Pydantic models for request/response

**Frontend** (`frontend/`)
- React + Vite, TypeScript
- Dev proxy: `/api` → `http://localhost:8000`
- Build: `npm run build` → `frontend/dist/`

**Config** (`core/config.py`)
- All thresholds, weights, paths, model hyperparameters
- `FUSION_WEIGHTS` — detector weights (sum ≈ 1.0)
- `TAMPER_THRESHOLD = 0.45` — verdict threshold

## Data Flow

1. **Ingestion** (`ingestion/loader.py`) — PDF → pages via PyMuPDF; images via OpenCV
2. **Preprocessing** (`ingestion/preprocess.py`) — resize (max 2000px), deskew
3. **OCR** (`ingestion/ocr.py`) — pytesseract extracts text regions for font/metadata detectors
4. **Detection** — 7 classical detectors run in parallel, each returns `Detection(score, heatmap, regions, details)`
5. **CNN** — TamperNet runs on preprocessed image, returns confidence + heatmap
6. **Fusion** — weighted average + strong-signal boost → verdict + fused heatmap
7. **API Response** — heatmap encoded as base64 PNG (COLORMAP_JET)

## Important Notes

- **Model download**: First API request downloads ~350 MB Hugging Face model. Set `DOCFORENSICS_DISABLE_AI_MODEL=1` to skip.
- **Memory**: Needs ~2 GB RAM (PyTorch + Transformers + OpenCV).
- **Training data**: Synthetic forgeries — genuine and tampered classes share same source documents so CNN learns tampering, not document identity.
- **Limitations**: Copy-move and double-JPEG are weak on text-heavy docs (legitimate repetition mimics tampering); CNN carries the verdict.
- **No pytest config**: Tests run with `pytest tests/ -v` directly.