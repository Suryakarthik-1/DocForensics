import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

app = FastAPI(title='DocForensics API', version='1.0.0')

# CORS — defaults to permissive for local dev; lock down in prod via env var,
# e.g. ALLOWED_ORIGINS="https://myapp.com,https://www.myapp.com"
_origins = os.getenv('ALLOWED_ORIGINS', '*').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router, prefix='/api')

# ── Serve the built React frontend (single-container deployment) ──────────────
# If frontend/dist exists (created by `npm run build`), serve it at the root so
# the whole app runs from one origin with no CORS configuration required.
_DIST = Path(__file__).resolve().parent.parent / 'frontend' / 'dist'
if _DIST.is_dir():
    app.mount('/assets', StaticFiles(directory=_DIST / 'assets'), name='assets')

    @app.get('/')
    def _index():
        return FileResponse(_DIST / 'index.html')

    @app.get('/{full_path:path}')
    def _spa_fallback(full_path: str):
        # Serve real files if present, otherwise fall back to index.html (SPA routing).
        candidate = _DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / 'index.html')
