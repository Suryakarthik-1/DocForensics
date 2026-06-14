import pytest
import torch

from core.config import GENUINE_DIR, TAMPERED_DIR, MODEL_INPUT_SIZE

_has_data = GENUINE_DIR.exists() and TAMPERED_DIR.exists() and \
            any(GENUINE_DIR.glob('*.jpg')) and any(TAMPERED_DIR.glob('*.jpg'))

pytestmark = pytest.mark.skipif(not _has_data, reason='dataset not generated')


def test_train_and_val_are_nonempty():
    from model.dataset import TamperDataset
    train = TamperDataset(split='train')
    val   = TamperDataset(split='val')
    assert len(train) > 0
    assert len(val) > 0


def test_item_shapes_and_label():
    from model.dataset import TamperDataset
    img, mask, label = TamperDataset(split='val')[0]
    assert img.shape == (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    assert mask.shape == (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    assert label in (0, 1)


def test_pixels_normalised_zero_to_one():
    from model.dataset import TamperDataset
    img, _, _ = TamperDataset(split='val')[0]
    assert float(img.min()) >= 0.0
    assert float(img.max()) <= 1.0


def test_both_classes_present_in_split():
    from model.dataset import TamperDataset
    ds = TamperDataset(split='train')
    labels = {lbl for _, lbl in ds.items}
    assert labels == {0, 1}            # balanced split has both classes
