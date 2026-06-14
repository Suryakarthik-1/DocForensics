import numpy as np

from synth.forge import (forge_copy_move, forge_splice, forge_text_edit,
                         apply_random_forgery)
from core.types import BBox, TextRegion


def test_copy_move_shapes_and_mask(bgr_uint8):
    result, mask = forge_copy_move(bgr_uint8)
    assert result.shape == bgr_uint8.shape
    assert mask.shape == bgr_uint8.shape[:2]
    assert mask.max() == 255            # something was marked tampered


def test_splice_shapes(bgr_uint8):
    donor = bgr_uint8[::-1].copy()
    result, mask = forge_splice(bgr_uint8, donor)
    assert result.shape == bgr_uint8.shape
    assert mask.shape == bgr_uint8.shape[:2]
    assert mask.max() == 255


def test_text_edit_without_regions_falls_back(bgr_uint8):
    # no regions → falls back to copy-move rather than crashing
    result, mask = forge_text_edit(bgr_uint8, [])
    assert result.shape == bgr_uint8.shape
    assert mask.max() == 255


def test_text_edit_with_region(bgr_uint8):
    regions = [TextRegion(BBox(10, 10, 30, 12), 'total', 0.9)]
    result, mask = forge_text_edit(bgr_uint8, regions)
    assert mask[10:22, 10:40].max() == 255


def test_apply_random_forgery_returns_op(bgr_uint8):
    donor = bgr_uint8[::-1].copy()
    result, mask, op = apply_random_forgery(bgr_uint8, donor=donor)
    assert result.shape == bgr_uint8.shape
    assert op in ('copy_move', 'splice', 'text_edit')
    assert mask.max() == 255
