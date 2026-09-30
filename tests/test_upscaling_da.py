import numpy as np

from rdupscale import Grid, DAParams, upscale_realisation, deterministic_annealing


def test_homogeneous_medium_upscales_to_itself():
    k = np.full((15, 15), 2.5)
    Kxx, Kyy, Kxy = upscale_realisation(k, Grid())
    assert np.allclose(Kxx, 2.5) and np.allclose(Kyy, 2.5) and np.allclose(Kxy, 0.0, atol=1e-12)


def test_layers_parallel_to_flow_give_arithmetic_mean():
    g = Grid()
    layers = np.exp(np.random.default_rng(1).normal(size=g.nf))
    k = np.repeat(layers[:, None], g.nf, axis=1)          # varia só em y
    Kxx, _, _ = upscale_realisation(k, g)
    expected = layers.reshape(g.nc, g.alpha).mean(axis=1)
    assert np.allclose(Kxx, expected[:, None], rtol=1e-10)


def test_da_basic_properties():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(1.0, 0.1, 400), rng.normal(3.0, 0.1, 600)])
    yk, qk, h, Tc = deterministic_annealing(s, DAParams(), seed=1)
    assert np.isclose(Tc, 2.0 * s.var(ddof=1))
    assert np.isclose(qk.sum(), 1.0) and (qk >= 0).all()
    assert s.min() - 1e-9 <= yk.min() and yk.max() <= s.max() + 1e-9
    assert len(yk) <= DAParams().K_max
    # as massas dos dois grupos são recuperadas
    assert abs(qk[yk < 2.0].sum() - 0.4) < 0.01
    # reprodutível com a mesma semente
    yk2, qk2, _, _ = deterministic_annealing(s, DAParams(), seed=1)
    assert np.array_equal(yk, yk2) and np.array_equal(qk, qk2)


def test_merge_tolerance_modes():
    assert DAParams(merge_tol=3.8e-4).merge_atol(0.5) == 3.8e-4 * 0.5
    assert DAParams.legacy(merge_tol=3.8e-4).merge_atol(0.5) == 3.8e-4


def test_relative_merge_is_scale_invariant():
    rng = np.random.default_rng(4)
    s = np.exp(rng.normal(size=600))
    a = deterministic_annealing(s, DAParams(), seed=5)
    for c in (1.0 / 1024.0, 1024.0):            # potências de 2: escala exata em ponto flutuante
        b = deterministic_annealing(c * s, DAParams(), seed=5)
        assert len(a[0]) == len(b[0])
        np.testing.assert_allclose(c * a[0], b[0], rtol=1e-9)
        np.testing.assert_allclose(a[1], b[1])
