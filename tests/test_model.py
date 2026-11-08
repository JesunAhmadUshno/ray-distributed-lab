import torch

from src.model import EXPECTED_PARAMS, build_model


def test_parameter_count():
    model = build_model(42)
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS


def test_forward_shape():
    model = build_model(42)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 10)
