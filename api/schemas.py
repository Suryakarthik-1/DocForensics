from pydantic import BaseModel


class DetectorResult(BaseModel):
    name: str
    score: float
    details: dict


class AnalyzeResponse(BaseModel):
    is_tampered: bool
    label: str
    confidence: float
    evidence: list[str]
    per_detector: list[DetectorResult]
    heatmap_base64: str