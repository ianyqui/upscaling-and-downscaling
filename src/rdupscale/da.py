"""Recozimento Determinístico (Rose) escalar, como implementado na tese."""
from __future__ import annotations

import numpy as np

from .config import DAParams


def deterministic_annealing(samples, params: DAParams = DAParams(), seed=None,
                            T_max=None, T_min=None):
    """DA 1-D com subamostragem estocástica por temperatura.

    Retorna (yk, qk, hist, T_c1): valores representativos, massas (por
    atribuição rígida final de todas as amostras), histórico com colunas
    (T, D, I, K, m) e a primeira temperatura crítica T_c1 = 2 Var(amostras).
    Com ``DAParams.legacy()`` (tolerância de fusão absoluta) a numérica é
    idêntica à de ``deterministic_annealing`` da tese para a mesma semente."""
    p = params
    rng = np.random.default_rng(seed)
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    M = s.size
    sd = s.std() + 1e-12
    sn = s
    T_crit = 2.0 * float(np.var(sn, ddof=1))
    if T_max is None:
        T_max = p.T_max_ratio * T_crit
    if T_min is None:
        T_min = p.T_min_ratio * T_crit
    merge_atol = p.merge_atol(sd)
    # escala do critério de parada das iterações internas: absoluta na tese,
    # relativa ao desvio-padrão no modo relativo (invariância de escala)
    y_scale = sd if p.merge_relative else 1.0
    alpha, K_max, n_inner, perturb, f_lo, f_hi = (p.alpha, p.K_max, p.n_inner,
                                                  p.perturb, p.f_lo, p.f_hi)
    N_steps = max(int(np.ceil(np.log(T_min / T_max) / np.log(alpha))), 1)
    y = np.array([sn.mean()]); q = np.array([1.0])
    hist = []; T = T_max; step = 0
    while T > T_min:
        prog = min(max(step / N_steps, 0.0), 1.0)
        frac = f_lo + (f_hi - f_lo) * prog
        m = max(int(round(frac * M)), min(40, M))
        bt = sn[rng.choice(M, m, replace=False)] if m < M else sn
        for _ in range(n_inner):
            d = (bt[:, None] - y[None, :]) ** 2
            lw = np.log(q + 1e-300)[None, :] - d / T
            lw -= lw.max(axis=1, keepdims=True)
            P = np.exp(lw); P /= P.sum(axis=1, keepdims=True)
            qn = P.mean(axis=0)
            yn = (P * bt[:, None]).sum(axis=0) / (qn * m + 1e-300)
            if np.max(np.abs(yn - y)) / y_scale + np.max(np.abs(qn - q)) < 1e-7:
                y, q = yn, qn
                break
            y, q = yn, qn
        keep = q > 1e-6
        y, q = y[keep], q[keep]
        if y.size == 0:
            y = np.array([sn.mean()]); q = np.array([1.0])
        o = np.argsort(y); y, q = y[o], q[o]
        ky_, kq_ = [y[0]], [q[0]]
        for i in range(1, len(y)):
            if y[i] - ky_[-1] < merge_atol:
                kq_[-1] += q[i]
            else:
                ky_.append(y[i]); kq_.append(q[i])
        y = np.asarray(ky_); q = np.asarray(kq_); q /= q.sum()
        d = (bt[:, None] - y[None, :]) ** 2
        lw = np.log(q + 1e-300)[None, :] - d / T
        lw -= lw.max(axis=1, keepdims=True)
        P = np.exp(lw); P /= P.sum(axis=1, keepdims=True)
        D = float((P * d).sum(axis=1).mean())
        I = float((P * (np.log(P + 1e-300) - np.log(q[None, :] + 1e-300))).sum(axis=1).mean())
        hist.append((T, D, I, len(y), m))
        T *= alpha; step += 1
        if 2 * len(y) <= K_max:
            y = np.concatenate([y, y + perturb * sd * rng.standard_normal(len(y))])
            q = np.concatenate([q, q]) / 2.0
    nb = np.argmin((sn[:, None] - y[None, :]) ** 2, axis=1)
    q = np.bincount(nb, minlength=len(y)).astype(float)
    q /= q.sum()
    return y, q, np.asarray(hist), T_crit


def da_seed(I: int, J: int, dkey: str) -> int:
    """Semente do DA por macrocélula e direção, como na tese."""
    return (I * 9 + J) * 3 + {"xx": 0, "yy": 1, "xy": 2}[dkey]
