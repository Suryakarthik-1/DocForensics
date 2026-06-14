import torch

from model.architecture import TamperNet, srm_filter


def test_srm_filter_preserves_spatial_shape():
    x = torch.rand(2, 3, 64, 64)
    out = srm_filter(x)
    assert out.shape == (2, 3, 64, 64)


def test_forward_output_shapes():
    model = TamperNet()
    model.eval()
    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        mask, logit = model(x)
    assert mask.shape == (2, 1, 128, 128)
    assert logit.shape[0] == 2


def test_mask_is_probability():
    model = TamperNet()
    model.eval()
    with torch.no_grad():
        mask, _ = model(torch.rand(1, 3, 128, 128))
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def test_handles_non_square_input():
    model = TamperNet()
    model.eval()
    with torch.no_grad():
        mask, _ = model(torch.rand(1, 3, 96, 128))
    assert mask.shape[2:] == (96, 128)
