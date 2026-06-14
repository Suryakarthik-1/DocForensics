import numpy as np
import pytest

from core.config import CHECKPOINTS_DIR

pytestmark = pytest.mark.skipif(
    not (CHECKPOINTS_DIR / 'best.pt').exists(),
    reason='trained checkpoint not available',
)


def test_predict_returns_confidence_and_heatmap():
    from model.inference import load_model, predict
    model = load_model()
    img = np.random.rand(120, 90, 3).astype(np.float32)
    out = predict(model, img)
    assert 0.0 <= out.confidence <= 1.0
    assert out.heatmap.shape == (120, 90)


def test_heatmap_values_are_probabilities():
    from model.inference import load_model, predict
    model = load_model()
    out = predict(model, np.random.rand(64, 64, 3).astype(np.float32))
    assert float(out.heatmap.min()) >= 0.0
    assert float(out.heatmap.max()) <= 1.0


def test_model_is_cached():
    from model.inference import load_model
    assert load_model() is load_model()
