"""Regressão contra os resultados publicados.

``artigos``: caches dos quatro artigos (P1.1, P1.3, P1.4; sementes 2026101/3/4,
macrorrealizações com semente 7, hiperparâmetros padrão da tese).
``tese``: caches de 02/08/2026 (P1.1–P1.5), com os hiperparâmetros do DA
recuperados da sequência determinística da busca aleatória."""
import numpy as np
import pytest

from rdupscale import (PROBLEMS, micro_stage, macro_stage, lml_stage, run_problem,
                       deterministic_annealing, da_seed)
from _refutil import load, da_all_from_ref

CASES = [("artigos", p) for p in ("1_1", "1_3", "1_4")] + [("tese", f"1_{i}") for i in range(1, 6)]
ids = [f"{k}-P{p}" for k, p in CASES]


def _name(pid):
    return pid.replace("_", ".")


@pytest.mark.parametrize("kind,pid", CASES, ids=ids)
def test_geostat_field_from_seed(kind, pid):
    ref, K, _, _ = load(kind, pid)
    np.testing.assert_allclose(K[0], ref["K_micro_first_realiz"], rtol=1e-6)
    np.testing.assert_allclose(K.mean(0), ref["K_micro_mean"], rtol=1e-6)
    np.testing.assert_allclose(K.std(0), ref["K_micro_std"], rtol=1e-6)


@pytest.mark.parametrize("kind,pid", CASES, ids=ids)
def test_micro_pressure_and_upscaling(kind, pid):
    ref, K, _, _ = load(kind, pid)
    n = ref["P_micro_sub"].shape[0]
    P, Kxx, Kyy, Kxy = micro_stage(K[:n], PROBLEMS[_name(pid)], n_jobs=1)
    np.testing.assert_allclose(P, ref["P_micro_sub"], atol=1e-7)
    np.testing.assert_allclose(Kxx, ref["Kxx_all"][:n], rtol=1e-5)
    np.testing.assert_allclose(Kyy, ref["Kyy_all"][:n], rtol=1e-5)
    np.testing.assert_allclose(Kxy, ref["Kxy_all"][:n], atol=1e-6)


def _check_da(kind, pid, entries):
    ref, _, prm, _ = load(kind, pid)
    samples = {"xx": ref["Kxx_all"], "yy": ref["Kyy_all"], "xy": ref["Kxy_all"]}
    for d, I, J in entries:
        yk, qk, h, _ = deterministic_annealing(samples[d][:, I, J], prm, seed=da_seed(I, J, d))
        np.testing.assert_allclose(yk, ref[f"da_{d}_{I}{J}_yk"], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(qk, ref[f"da_{d}_{I}{J}_qk"], atol=1e-9)
        np.testing.assert_allclose(h, ref[f"da_{d}_{I}{J}_h"], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kind,pid", CASES, ids=ids)
def test_da_codebooks_sample(kind, pid):
    _check_da(kind, pid, [("xx", 0, 0), ("yy", 2, 2), ("xy", 4, 1)])


@pytest.mark.slow
@pytest.mark.parametrize("kind,pid", CASES, ids=ids)
def test_da_codebooks_all(kind, pid):
    _check_da(kind, pid, [(d, I, J) for I in range(5) for J in range(5) for d in ("xx", "yy", "xy")])


@pytest.mark.parametrize("pid", ["1_1", "1_3", "1_4"])
def test_articles_macro_realizations(pid):
    ref, _, _, meta = load("artigos", pid)
    P = macro_stage(da_all_from_ref(ref), PROBLEMS[_name(pid)], meta["n_R"], meta["seed_macro"])
    np.testing.assert_allclose(P, ref["P_macro_all"], atol=1e-10)


@pytest.mark.parametrize("kind,pid", CASES, ids=ids)
def test_lml_from_reference_macro(kind, pid):
    ref, K, _, _ = load(kind, pid)
    P, *_ = micro_stage(K, PROBLEMS[_name(pid)], n_jobs=1)
    L = lml_stage(P, ref["P_macro_all"])
    np.testing.assert_allclose(L["Pdown_mean"], ref["Pdown_mean"], atol=1e-7)
    np.testing.assert_allclose(L["Pdown_std"], ref["Pdown_std"], atol=1e-7)


@pytest.mark.slow
@pytest.mark.parametrize("pid", ["1_1", "1_3", "1_4"])
def test_articles_end_to_end_parallel(pid):
    """Pipeline completo em processos paralelos reproduz o cache dos artigos."""
    ref, K, prm, meta = load("artigos", pid)
    res = run_problem(_name(pid), PROBLEMS[_name(pid)], K, da_params=prm,
                      n_R=meta["n_R"], seed_macro=meta["seed_macro"], n_jobs=-1)
    np.testing.assert_allclose(res["P_macro_all"], ref["P_macro_all"], atol=1e-8)
    np.testing.assert_allclose(res["Pdown_mean"], ref["Pdown_mean"], atol=1e-8)
    np.testing.assert_allclose(res["Pdown_std"], ref["Pdown_std"], atol=1e-8)
    assert abs(res["rms"] - float(ref["rms"])) < 1e-9
