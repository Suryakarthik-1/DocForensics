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

**Detectors** (`detectors/`) — all implement `Detector.run(img, ctx: Context) -> Detection`
- `base.py` — abstract `Detector` class, `Context` carries `file_path`, `text_regions`, `original_img`, `extra`
- `ela.py` — Error Level Analysis (re-compression at quality 95, amplify ×15)
- `noise.py` — Local noise sigma via sliding 16px window, flags > 2σ anomalies
- `copymove.py` — DCT block hashing (16px, 8px stride), capped at 0.6 score
- `double_jpeg.py` — DCT histogram periodicity, capped at 0.6 score
- `font_forensics.py` — Laplacian variance + baseline alignment (uses OCR regions)
- `metadata.py` — EXIF tags + PDF structure anomalies
- `ai_generated.py` — Frequency spectrum + EXIF absence + HuggingFace classifier (cached, respects `DOCFORENSICS_DISABLE_AI_MODEL`)

**Fusion** (`fusion/decision.py`)
- Weighted average using `FUSION_WEIGHTS` from `core/config.py`:
  ```
  ela 0.15, noise 0.13, copy_move 0.04, double_jpeg 0.03,
  font 0.10, metadata 0.05, ai_generated 0.20, model 0.30
  ```
- Strong-signal boost: `max(weighted, 0.55 * strongest + 0.45 * weighted)`
- Label logic: `score >= 0.45` → flagged; if flagged and ai ≥ 0.6 and ai is strongest → `AI-GENERATED`; else `TAMPERED`; else `AUTHENTIC`
- Heatmaps merged with same weights, resized to common resolution
- `build_evidence()` lists detectors with score > 0.3

**Model** (`model/`)
- `architecture.py` — TamperNet: two-stream U-Net (RGB stream + SRM noise-residual stream with fixed high-pass filters), fused at bottleneck (256ch), outputs sigmoid mask + classification logit
- `inference.py` — `load_model()` (global singleton), `predict()` resizes to 128×128, returns confidence + heatmap resized to original dims
- `dataset.py` — `TamperDataset`: genuine/tampered from `data/genuine`, `data/tampered`, `data/masks`; 15% val split (seed 42); on-the-fly augmentation
- `train.py` — BCE + Dice loss, Adam, ReduceLROnPlateau, early stopping patience 6, saves `best.pt` (by AUC) + `last.pt` + `history.json`
- `evaluate.py` — returns AUC, F1, pixel IoU
- Checkpoint at `model/checkpoints/best.pt` (~5 MB fp16)

**API** (`api/`)
- `main.py` — FastAPI app, serves React frontend from `frontend/dist/`
- `routes.py` — `POST /api/analyze` (multipart upload), `GET /api/health`
- `schemas.py` — Pydantic models for request/response

**Frontend** (`frontend/`)
- React + Vite, plain JSX (not TypeScript)
- ESLint: `cd frontend && npm run lint`
- Dev proxy: `/api` → `http://localhost:5173` proxies to `:8000`
- Build: `npm run build` → `frontend/dist/`

**Config** (`core/config.py`)
- All thresholds, weights, paths, model hyperparameters
- `FUSION_WEIGHTS` — detector weights (sum ≈ 1.0)
- `TAMPER_THRESHOLD = 0.45` — verdict threshold

## Data Flow

1. **Ingestion** (`ingestion/loader.py`) — PDF → pages via PyMuPDF; images via OpenCV, normalized to float32, grayscale→3ch
2. **Preprocessing** (`ingestion/preprocess.py`) — resize (max 2000px, `resize_keep_ratio`), deskew (Otsu → minAreaRect → warp)
3. **OCR** (`ingestion/ocr.py`) — PaddleOCR (cached) with pytesseract fallback, extracts text regions for font/metadata detectors
4. **Detection** — 7 classical detectors run in parallel, each returns `Detection(score, heatmap, regions, details)`
5. **CNN** — TamperNet runs on preprocessed image (128×128), returns confidence + heatmap
6. **Fusion** — weighted average + strong-signal boost → verdict + fused heatmap
7. **API Response** — heatmap encoded as base64 PNG (COLORMAP_JET)

## Important Notes

- **Model download**: First API request downloads ~350 MB Hugging Face model (`umm-maybe/AI-image-detector`). Set `DOCFORENSICS_DISABLE_AI_MODEL=1` to skip. TamperNet checkpoint (`best.pt`, ~5 MB) is local.
- **Memory**: Needs ~2 GB RAM (PyTorch + Transformers + OpenCV).
- **Training data**: Synthetic forgeries — genuine and tampered classes share same source documents so CNN learns tampering, not document identity.
- **Limitations**: Copy-move and double-JPEG are weak on text-heavy docs (legitimate repetition mimics tampering); CNN carries the verdict.
- **No pytest config**: Tests run with `pytest tests/ -v` directly.
- **CI** (`.github/workflows/ci.yml`): Runs `pytest tests/ -v --maxfail=1` on Python 3.11 with `DOCFORENSICS_DISABLE_AI_MODEL=1`, then Docker build.