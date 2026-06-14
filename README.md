# 🔍 DocForensics — Document Tampering & Forgery Detector

A computer-vision system that inspects documents and images for signs of
manipulation. It combines **seven classical forensic detectors** with a
**trained two-stream CNN** and fuses their signals into a single verdict:
**AUTHENTIC**, **TAMPERED**, or **AI-GENERATED** — with a heatmap showing *where*
the suspicion lies.

![status](https://img.shields.io/badge/tests-87%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.11-blue)
![stack](https://img.shields.io/badge/stack-FastAPI%20%C2%B7%20PyTorch%20%C2%B7%20React-6c63ff)

---

## What it does

| Detector | Looks for |
|---|---|
| **ELA** | Error-level inconsistencies from re-compression |
| **Noise** | Local noise-fingerprint mismatches at splice edges |
| **Copy-Move** | Cloned/duplicated regions (offset-clustered) |
| **Double-JPEG** | Periodic artifacts of re-compression |
| **Font forensics** | Baseline / anti-aliasing inconsistencies in text |
| **Metadata** | Editor software, incremental PDF edits, date mismatches |
| **AI-generated** | Frequency signature + a Hugging Face classifier |
| **TamperNet (CNN)** | Learned tamper localization (RGB + SRM noise streams) |

A weighted fusion layer combines these into a confidence score and a 3-way
verdict, with a per-pixel heatmap merged from each detector.

## Architecture

```
ingestion → preprocessing → OCR → ┌─ 7 forensic detectors ─┐
                                   ├─ TamperNet CNN          ├─ fusion → Verdict + heatmap
                                   └────────────────────────┘
                       FastAPI (REST + serves the React UI)
```

## Tech stack

- **Backend**: FastAPI, PyTorch, OpenCV, scikit-image, PyMuPDF, Transformers
- **Model**: custom two-stream U-Net (`TamperNet`) with SRM noise filters, AUC ≈ 0.94
- **Frontend**: React + Vite (single-origin, served by the API)
- **Testing**: 87 tests across unit / integration / API / model layers

## Quick start (local dev)

```bash
# Backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
# open http://localhost:5173  (proxies /api → :8000)
```

## Run the full app in one container

```bash
docker compose up --build
# open http://localhost:8000
```

## Tests

```bash
pytest tests/ -v        # 87 tests
```

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** — single-container Docker, with step-by-step
guides for Hugging Face Spaces, Render, Railway, Fly.io, and split
frontend/backend hosting.

## Notes & honest limitations

- Trained on **synthetic** forgeries (CPU). It reliably separates clean documents
  from forged ones, but very small/subtle copy-move patches can sit near the
  decision boundary.
- Copy-move and double-JPEG are weak on text-heavy documents (legitimate
  repetition mimics tampering), so they're low-weight hints; the CNN carries the
  verdict.
- First request downloads a ~350 MB AI-detector model, then caches it.
