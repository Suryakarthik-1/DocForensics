from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class TextRegion:
    bbox: BBox
    text: str
    confidence: float


@dataclass
class Detection:
    detector_name: str
    score: float
    heatmap: Optional[np.ndarray] = None
    regions: list[BBox] = field(default_factory=list)
    details: dict = field(default_factory=dict)


@dataclass
class Verdict:
    is_tampered: bool
    confidence: float
    fused_heatmap: np.ndarray
    evidence: list[str]
    per_detector: list[Detection]
    label: str = "AUTHENTIC"   # AUTHENTIC | AI-GENERATED | TAMPERED