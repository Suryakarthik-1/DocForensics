from core.config import CHECKPOINTS_DIR
from core.types import Verdict
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


def analyze(path: str) -> Verdict:
    pages = load_document(path)
    img   = pages[0].image
    img   = resize_keep_ratio(img)
    img, _ = deskew(img)

    text_regions = extract_text_regions(img)
    ctx          = Context(
        file_path=path,
        text_regions=text_regions,
        original_img=img,
    )

    detections = [d.run(img, ctx) for d in register_detectors()]

    model_confidence = 0.0
    model_heatmap    = None

    weights_path = CHECKPOINTS_DIR / 'best.pt'
    if weights_path.exists():
        from model.inference import load_model, predict
        model     = load_model(str(weights_path))
        model_out = predict(model, img)
        model_confidence = model_out.confidence
        model_heatmap    = model_out.heatmap

    return fuse(detections, model_confidence, model_heatmap)