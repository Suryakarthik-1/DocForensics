from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from core.types import Detection, TextRegion


@dataclass
class Context:
    file_path: str
    text_regions: list[TextRegion]
    original_img: np.ndarray
    extra: dict = field(default_factory=dict)


class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        ...

    def _empty(self) -> Detection:
        return Detection(detector_name=self.name, score=0.0)