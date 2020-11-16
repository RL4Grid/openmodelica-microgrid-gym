import numpy as np
from pytest import approx

from openmodelica_microgrid_gym.aux_ctl.base import LimitLoadIntegral


def test_limit_load_integral():
    dt = .5
    freq = 5
    limit_energy = 50
    i2t = LimitLoadIntegral(dt, freq, limit_energy, limit_energy / 5)
    size = int(freq / dt / 2)
    assert i2t._buffer.size == size

    i2t.reset()
    seq = [5, 4]
    for i in seq:
        i2t.step(i)
    integral = i2t.integral
    assert integral == (np.power(seq, 2) * dt).sum()
    assert i2t.risk() == approx(.2625)
    for i in [5, 4, 5, 5, 5, 5, 5, 5] + [0] * size + seq:
        i2t.step(i)
    integral = i2t.integral
    assert integral == (np.power(seq, 2) * dt).sum()
