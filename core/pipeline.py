import numpy as np

from core.config import CHECKPOINTS_DIR
from core.types import Detection, Verdict
from detectors.ai_generated import AIGeneratedDetector
from detectors.base import Context
from detectors.copymove import CopyMoveDetector
from detectors.double_jpeg import DoubleJpegDetector
from detectors.ela import ELADetector
from detectors.font_forensics import FontForensicsDetector
from detectors.metadata import MetadataDetector
from detectors.noise import NoiseDetector
from fusion.decision import fuse
from ingestion.loader import load_document
from ingestion.ocr import extract_text_regions
from ingestion.preprocess import deskew, resize_keep_ratio


def register_detectors():
    return [
        ELADetector(),
        NoiseDetector(),
        CopyMoveDetector(),
        DoubleJpegDetector(),
        FontForensicsDetector(),
        MetadataDetector(),
        AIGeneratedDetector(),
    ]


def _run_model(img: np.ndarray):
    """Run the trained CNN if a checkpoint exists. Returns (confidence, heatmap)."""
    weights_path = CHECKPOINTS_DIR / 'best.pt'
    if not weights_path.exists():
        return 0.0, None
    try:
        from model.inference import load_model, predict
        model = load_model(str(weights_path))
        out   = predict(model, img)
        return out.confidence, out.heatmap
    except Exception:
        return 0.0, None


def _analyze_page(img: np.ndarray, file_path: str) -> Verdict:
    img = resize_keep_ratio(img)
    img, _ = deskew(img)

    text_regions = extract_text_regions(img)
    ctx = Context(file_path=file_path, text_regions=text_regions, original_img=img)

    detections = [d.run(img, ctx) for d in register_detectors()]

    model_conf, model_heat = _run_model(img)
    # Surface the model as its own row in the breakdown.
    detections.append(Detection(
        detector_name='model',
        score=float(model_conf),
        heatmap=None,   # fusion merges the model heatmap via the model_heat arg below
        regions=[],
        details={'note': 'CNN tamper-localization confidence'},
    ))

    return fuse(detections, model_conf, model_heat)


def analyze(path: str) -> Verdict:
    pages = load_document(path)

    verdicts = [_analyze_page(p.image, path) for p in pages]

    # For multi-page PDFs, report the most suspicious page.
    return max(verdicts, key=lambda v: v.confidence)
