"""Pipeline completo de um problema elíptico: micro -> upscaling -> DA ->
macromalha -> downscaling LML. Reproduz ``run_problem`` da tese."""
from __future__ import annotations

import numpy as np

from .config import Grid, DAParams, LMLParams
from .da import deterministic_annealing, da_seed
from .fv import tpfa
from .lml import (lml_downscale, lml_downscale_stochastic, build_chol_corr_fine,
                  calibrate_lml_residual)
from .metrics import recommend_n_R
from .upscaling import upscale_realisation

DIRS = ("xx", "yy", "xy")


def resolve_n_jobs(n_jobs) -> int:
    """-1 = todos os núcleos; None = 1."""
    import os
    if n_jobs is None:
        return 1
    n = int(n_jobs)
    cpus = os.cpu_count() or 1
    return max(1, cpus + 1 + n) if n < 0 else max(1, min(n, cpus))


def _map(fn, items, n_jobs, batch_size="auto"):
    """Aplica fn a cada item, em série ou em processos (joblib/loky).

    Processos (e não threads) porque as operações do DA são sobre vetores
    pequenos e seguram o GIL. Cada processo usa 1 thread de BLAS para não
    haver disputa entre processos. Os resultados são idênticos aos da
    execução em série: cada tarefa é determinística e independente."""
    n = resolve_n_jobs(n_jobs)
    if n == 1 or len(items) < 2:
        return [fn(*it) for it in items]
    from joblib import Parallel, delayed, parallel_config
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=n, batch_size=batch_size)(delayed(fn)(*it) for it in items)


def micro_stage(K_micro, bc, grid: Grid = Grid(), n_jobs: int = -1):
    """Pressão fina e tensores efetivos de todas as realizações."""
    NC, A = grid.nc, grid.alpha

    def one(k):
        p_ = tpfa(k, k, bc, grid.hf, grid.mu)
        Kxx_, Kyy_, Kxy_ = upscale_realisation(k, grid)
        for I in range(NC):
            for J in range(NC):
                ka = k[I * A:(I + 1) * A, J * A:(J + 1) * A].mean()
                if not np.isfinite(Kxx_[I, J]) or Kxx_[I, J] <= 0:
                    Kxx_[I, J] = ka
                if not np.isfinite(Kyy_[I, J]) or Kyy_[I, J] <= 0:
                    Kyy_[I, J] = ka
                if not np.isfinite(Kxy_[I, J]):
                    Kxy_[I, J] = 0.0
        return p_, Kxx_, Kyy_, Kxy_

    def block(Kb):
        return [one(k) for k in Kb]

    n = resolve_n_jobs(n_jobs)
    chunks = np.array_split(np.arange(K_micro.shape[0]), max(1, 4 * n))
    out = [o for part in _map(block, [(K_micro[c],) for c in chunks if c.size], n)
           for o in part]
    P = np.array([o[0] for o in out])
    return P, np.array([o[1] for o in out]), np.array([o[2] for o in out]), np.array([o[3] for o in out])


def _one_da(samp, params, I, J, dkey):
    yk, qk, hk, Tc = deterministic_annealing(samp, params, seed=da_seed(I, J, dkey))
    return dict(IJ=(I, J), h=hk, yk=yk, qk=qk, samples=samp, Tc=Tc)


def da_stage(Kxx_s, Kyy_s, Kxy_s, params: DAParams = DAParams(), n_jobs: int = -1):
    """DA escalar por macrocélula e direção; retorna dict dkey -> lista de entradas."""
    NC = Kxx_s.shape[1]
    pool = {"xx": Kxx_s, "yy": Kyy_s, "xy": Kxy_s}

    items = [(I, J, d) for I in range(NC) for J in range(NC) for d in DIRS]
    # uma tarefa por vez em cada processo, para equilibrar a carga
    args = [(pool[d][:, I, J].copy(), params, I, J, d) for I, J, d in items]
    res = _map(_one_da, args, n_jobs, batch_size=1)
    by = {(it[0], it[1], it[2]): r for it, r in zip(items, res)}
    return {d: [by[(I, J, d)] for I in range(NC) for J in range(NC)] for d in DIRS}


def macro_stage(da_all, bc, n_R, seed_macro, grid: Grid = Grid(), n_jobs: int = 1):
    """n_R macrorrealizações: sorteia Kxx e Kyy de cada codebook, independentes
    entre células e entre direções (como na tese), e resolve por TPFA."""
    NC = grid.nc
    cb = {d: {e["IJ"]: (e["yk"], e["qk"]) for e in da_all[d]} for d in ("xx", "yy")}
    rng = np.random.default_rng(seed_macro)
    seeds = rng.integers(0, 2**31 - 1, size=n_R)

    def one(seed_r):
        r = np.random.default_rng(int(seed_r))
        Kx, Ky = np.empty((NC, NC)), np.empty((NC, NC))
        for I in range(NC):
            for J in range(NC):
                yx, qx = cb["xx"][(I, J)]
                yy, qy = cb["yy"][(I, J)]
                Kx[I, J] = yx[r.choice(len(yx), p=qx / qx.sum())]
                Ky[I, J] = yy[r.choice(len(yy), p=qy / qy.sum())]
        return tpfa(Kx, Ky, bc, grid.hc, grid.mu)

    return np.array(_map(one, [(s,) for s in seeds], n_jobs))


def lml_stage(P_micro, P_macro, grid: Grid = Grid(), lml: LMLParams = LMLParams()):
    """Downscaling LML estocástico da tese, com correção de viés e reescala do
    desvio-padrão calibradas na própria referência fina (avaliação na amostra)."""
    Pmic_mean, Pmic_std = P_micro.mean(0), P_micro.std(0)
    n_R = P_macro.shape[0]
    _, ell_res = calibrate_lml_residual(P_micro, grid, lml)
    P_lml_det = np.array([lml_downscale(P_macro[r], grid, lml.degree)[0] for r in range(n_R)])
    var_lml = P_lml_det.var(axis=0)
    sigma_res_field = np.sqrt(np.maximum(Pmic_std ** 2 - var_lml, 0.0))
    bias_field = Pmic_mean - P_lml_det.mean(axis=0)
    L_chol = build_chol_corr_fine(ell_res, grid)
    N_STOC = max(n_R, lml.n_stoc_min)
    rng = np.random.default_rng(lml.seed)
    P_down_raw = np.empty((N_STOC, grid.nf, grid.nf))
    for r in range(N_STOC):
        P_down_raw[r] = lml_downscale_stochastic(P_macro[r % n_R], sigma_res_field,
                                                 L_chol, rng, grid, lml.degree) + bias_field
    mu_raw = P_down_raw.mean(axis=0)
    std_raw = P_down_raw.std(axis=0) + 1e-12
    P_down = (P_down_raw - mu_raw) * (Pmic_std / std_raw) + mu_raw
    return dict(ell_res=ell_res, P_lml_det_all=P_lml_det, var_lml_mc=var_lml,
                sigma_res_field=sigma_res_field, sigma_res=float(sigma_res_field.mean()),
                bias_field=bias_field, P_down_raw_first10=P_down_raw[:10],
                Pdown_mean=P_down.mean(axis=0), Pdown_std=P_down.std(axis=0))


def run_problem(name, bc, K_micro, grid: Grid = Grid(), da_params: DAParams = DAParams(),
                n_R=25, seed_macro=None, lml: LMLParams = LMLParams(), n_jobs: int = -1,
                P_macro=None):
    """Executa o pipeline e devolve um dicionário com as mesmas chaves do cache
    da tese (mais ``da_params`` e ``seed_macro``). ``P_macro`` permite reusar
    macrorrealizações já calculadas (a tese não fixava ``seed_macro``)."""
    N_R = K_micro.shape[0]
    P_micro, Kxx_s, Kyy_s, Kxy_s = micro_stage(K_micro, bc, grid, n_jobs)
    Pmic_mean, Pmic_std = P_micro.mean(0), P_micro.std(0)
    da_all = da_stage(Kxx_s, Kyy_s, Kxy_s, da_params, n_jobs)
    if n_R in (0, "auto", None):
        n_R, _ = recommend_n_R(da_all["xx"], da_all["yy"], da_all["xy"], N_R)
    n_R = int(n_R)
    if P_macro is None:
        P_macro = macro_stage(da_all, bc, n_R, seed_macro, grid, 1)
    n_R = P_macro.shape[0]
    L = lml_stage(P_micro, P_macro, grid, lml)
    NC = grid.nc
    nmac = {d: np.array([len(e["yk"]) for e in da_all[d]]).reshape(NC, NC) for d in DIRS}
    res = dict(
        name=name, bc=bc, N_R=N_R, n_R=n_R,
        rms=float(np.sqrt(np.mean((Pmic_mean - L["Pdown_mean"]) ** 2))),
        N_F=grid.nf, N_C=grid.nc, ALPHA=grid.alpha, HF=grid.hf, HC=grid.hc, MU=grid.mu, LX=grid.lx,
        K_micro_first_realiz=K_micro[0], K_micro_mean=K_micro.mean(axis=0),
        K_micro_std=K_micro.std(axis=0),
        Pmic_mean=Pmic_mean, Pmic_std=Pmic_std, P_micro_all=P_micro,
        Kxx_all=Kxx_s, Kyy_all=Kyy_s, Kxy_all=Kxy_s,
        Kxx_map=Kxx_s.mean(0), Kyy_map=Kyy_s.mean(0), Kxy_map=Kxy_s.mean(0),
        Kxx_std=Kxx_s.std(0), Kyy_std=Kyy_s.std(0), Kxy_std=Kxy_s.std(0),
        da_all_xx=da_all["xx"], da_all_yy=da_all["yy"], da_all_xy=da_all["xy"],
        n_macro=list(nmac["xx"].ravel()), nmac_map=nmac["xx"],
        nmac_map_xx=nmac["xx"], nmac_map_yy=nmac["yy"], nmac_map_xy=nmac["xy"],
        P_macro_all=P_macro, Pmac_mean=P_macro.mean(0), Pmac_std=P_macro.std(0),
        sig2_mean=lml_downscale(P_macro.mean(0), grid, lml.degree)[1],
        sig2_std=lml_downscale(P_macro.std(0), grid, lml.degree)[1],
        da_params=da_params.to_dict(), seed_macro=seed_macro,
    )
    res.update({k: v for k, v in L.items()})
    return res
