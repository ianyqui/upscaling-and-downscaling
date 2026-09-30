"""Busca aleatória dos hiperparâmetros do DA (critério minimax por quartil)."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import replace

import numpy as np

from .config import DAParams
from .da import deterministic_annealing
from .metrics import ks_exact, ks_quartis

DIRS = ("xx", "yy", "xy")
QS = ("q1", "q2", "q3", "q4")


def worst_cells(res: dict) -> dict:
    """Amostras da macrocélula de maior variância em cada direção."""
    NC = res["N_C"]
    out = {}
    for d, key in (("xx", "Kxx_all"), ("yy", "Kyy_all"), ("xy", "Kxy_all")):
        arr = res[key]
        I, J = divmod(int(np.argmax(arr.var(axis=0))), NC)
        out[d] = arr[:, I, J]
    return out


def evaluate(params: DAParams, samples_by_dir: dict, seeds) -> tuple[dict, dict]:
    ks, qd = {d: [] for d in DIRS}, {d: {q: [] for q in QS} for d in DIRS}
    for d, s in samples_by_dir.items():
        for seed in seeds:
            yk, qk, _, _ = deterministic_annealing(s, params, seed=seed)
            ks[d].append(ks_exact(s, yk, qk))
            qq = ks_quartis(s, yk, qk)
            for i, q in enumerate(QS, start=1):
                qd[d][q].append(qq[f"Q{i}"]["Linf"])
    return ({d: float(np.mean(v)) for d, v in ks.items()},
            {d: {q: float(np.mean(qd[d][q])) for q in QS} for d in DIRS})


def tune_da_random(res: dict, prob: str, n_iters: int = 15, n_seeds: int = 3,
                   base: DAParams = DAParams(), criterion: str = "quartile_minimax",
                   csv_path: str | None = None, json_path: str | None = None):
    """Mesmo espaço de busca e mesmo critério da tese. Semente determinística
    derivada do nome do problema. Retorna (melhor DAParams, objetivo)."""
    h = hashlib.sha256(f"tune-da-{prob}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "big") % (2**31))
    samples = worst_cells(res)
    rows, best, best_obj = [], None, np.inf
    t0 = time.perf_counter()
    for it in range(1, n_iters + 1):
        cfg = replace(base,
                      alpha=float(np.exp(rng.uniform(np.log(0.92), np.log(0.98)))),
                      n_inner=int(rng.choice([40, 75, 100, 150])),
                      perturb=float(np.exp(rng.uniform(np.log(5e-4), np.log(1e-2)))),
                      merge_tol=float(np.exp(rng.uniform(np.log(5e-5), np.log(1e-3)))),
                      f_lo=float(rng.uniform(0.7, 1.0)), f_hi=1.0)
        ks_d, q_d = evaluate(cfg, samples, list(range(n_seeds)))
        qmm = max(q_d[d][q] for d in DIRS for q in QS)
        obj = qmm if criterion == "quartile_minimax" else max(ks_d.values())
        rows.append({"iter": it, **cfg.to_dict(), **{f"ks_{d}": ks_d[d] for d in DIRS},
                     "quartile_minimax": qmm, "objetivo": obj})
        if obj < best_obj:
            best, best_obj = cfg, obj
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    if json_path:
        with open(json_path, "w") as f:
            json.dump({**best.to_dict(), "objetivo_at_tune": best_obj,
                       "criterion": criterion, "n_iters": n_iters, "n_seeds": n_seeds,
                       "prob": prob, "tempo_s": time.perf_counter() - t0}, f, indent=2)
    return best, best_obj
