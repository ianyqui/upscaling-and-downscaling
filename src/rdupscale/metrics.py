"""Métricas de comparação entre a distribuição fina e o codebook do DA."""
from __future__ import annotations

import numpy as np

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _cdfs_on_union(samples, yk, qk):
    sX = np.sort(samples); FX = np.arange(1, len(sX) + 1) / len(sX)
    o = np.argsort(yk); sY = yk[o]; FY = np.cumsum(qk[o] / qk.sum())
    g = np.union1d(sX, sY)
    iX = np.searchsorted(sX, g, side="right") - 1
    iY = np.searchsorted(sY, g, side="right") - 1
    FXg = np.where(iX >= 0, FX[np.clip(iX, 0, len(FX) - 1)], 0.0)
    FYg = np.where(iY >= 0, FY[np.clip(iY, 0, len(FY) - 1)], 0.0)
    return g, sX, FXg, FYg


def ks_exact(samples, yk, qk) -> float:
    """Distância de Kolmogorov–Smirnov exata entre a amostra e o codebook."""
    _, _, FXg, FYg = _cdfs_on_union(samples, yk, qk)
    return float(np.max(np.abs(FXg - FYg)))


def ks_quartis(samples, yk, qk) -> dict:
    """Normas L_inf, L2 e L1 de |F_X - F_Y| em cada quartil de F_X e no global."""
    g, sX, FXg, FYg = _cdfs_on_union(samples, yk, qk)
    delta = np.abs(FXg - FYg)
    q25, q50, q75 = np.quantile(sX, [0.25, 0.5, 0.75])
    bounds = [(g.min(), q25), (q25, q50), (q50, q75), (q75, g.max())]

    def _norms(mask):
        if not mask.any():
            return {"Linf": 0.0, "L2": 0.0, "L1": 0.0}
        d = delta[mask]; gs = g[mask]
        Linf = float(np.max(d))
        if len(gs) >= 2:
            L1 = float(_trapz(d, gs))
            width = gs[-1] - gs[0]
            L2 = float(np.sqrt(_trapz(d ** 2, gs) / max(width, 1e-12)))
        else:
            L1 = L2 = float(d.mean())
        return {"Linf": Linf, "L2": L2, "L1": L1}

    out = {f"Q{i}": _norms((g >= lo) & (g <= hi))
           for i, (lo, hi) in enumerate(bounds, start=1)}
    out["GLOBAL"] = _norms(np.ones_like(g, dtype=bool))
    out.update(q25=float(q25), q50=float(q50), q75=float(q75),
               t_min=float(g.min()), t_max=float(g.max()))
    return out


def recommend_n_R(da_all_xx, da_all_yy, da_all_xy, N_R, q_factor=5.0,
                  strategy="p75"):
    """n_R sugerido pela regra de ter pelo menos q_factor amostras por valor."""
    q_thresh = q_factor / N_R
    Kv = np.array([int(((e["qk"] / e["qk"].sum()) >= q_thresh).sum())
                   for lst in (da_all_xx, da_all_yy, da_all_xy) for e in lst])
    stats = {
        "q_thresh": q_thresh, "n_macroels_total": len(Kv),
        "K_validos_min": int(Kv.min()), "K_validos_p25": int(np.percentile(Kv, 25)),
        "K_validos_median": int(np.median(Kv)), "K_validos_p75": int(np.percentile(Kv, 75)),
        "K_validos_max": int(Kv.max()), "K_validos_mean": float(Kv.mean()),
    }
    key = {"min": "K_validos_min", "p25": "K_validos_p25", "median": "K_validos_median",
           "p75": "K_validos_p75", "max": "K_validos_max"}.get(strategy, "K_validos_p75")
    return int(stats[key]), stats
