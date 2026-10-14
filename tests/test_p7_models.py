import torch

from edalint_p7 import models as P


def test_mavi_shapes_and_params():
    m = P.MAVI(1, 4)
    assert sum(p.numel() for p in m.parameters()) == 17063812
    m.eval()
    with torch.no_grad():
        y = m(torch.rand(1, 1, 24, 32, 32))
    assert y.shape == (1, 32, 32)


def test_gpdl_shapes_and_params():
    g = P.GPDL(3, 1)
    assert sum(p.numel() for p in g.parameters()) == 119169
    g.eval()
    with torch.no_grad():
        y = g(torch.rand(2, 3, 64, 64))
    assert y.shape == (2, 1, 64, 64) and float(y.min()) >= 0 and float(y.max()) <= 1
