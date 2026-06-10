import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile

from api.schemas import AnalyzeResponse, DetectorResult
from core.config import SUPPORTED_EXTS
from core.pipeline import analyze

router = APIRouter()


def _heatmap_to_base64(heatmap: np.ndarray) -> str:
    heat_u8  = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    colored  = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    _, buf   = cv2.imencode('.png', colored)
    return base64.b64encode(buf.tobytes()).decode()


@router.post('/analyze', response_model=AnalyzeResponse)
async def analyze_document(file: UploadFile):
    ext = Path(file.filename or '').suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f'Unsupported file type: {ext}')

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        verdict = analyze(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return AnalyzeResponse(
        is_tampered=verdict.is_tampered,
        confidence=round(verdict.confidence, 4),
        evidence=verdict.evidence,
        per_detector=[
            DetectorResult(name=d.detector_name, score=round(d.score, 4), details=d.details)
            for d in verdict.per_detector
        ],
        heatmap_base64=_heatmap_to_base64(verdict.fused_heatmap),
    )


@router.get('/health')
def health():
    return {'status': 'ok'}