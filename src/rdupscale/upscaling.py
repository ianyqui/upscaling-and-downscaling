"""Upscaling de permeabilidade por dois experimentos canônicos (Durlofsky, 1991)."""
from __future__ import annotations

import numpy as np

from .config import Grid
from .fv import tpfa, darcy_velocity, grads
from .problems import BC_UPSCALE_X, BC_UPSCALE_Y


def upscale_realisation(k: np.ndarray, grid: Grid = Grid()):
    """Tensor efetivo (Kxx, Kyy, Kxy) por macrocélula, cada um (nc, nc).

    Idêntico à tese: médias de velocidade e gradiente por macrocélula nos
    experimentos com gradiente em x e em y; Kxy simetrizado; correções de
    positividade (diagonal > 0 e |Kxy| <= sqrt(Kxx Kyy))."""
    h, MU, NC, A = grid.hf, grid.mu, grid.nc, grid.alpha
    p_x = tpfa(k, k, BC_UPSCALE_X, h, MU)
    p_y = tpfa(k, k, BC_UPSCALE_Y, h, MU)
    vx_x, vy_x = darcy_velocity(p_x, k, h, MU)
    vx_y, vy_y = darcy_velocity(p_y, k, h, MU)
    gx_x, _ = grads(p_x, h)
    _, gy_y = grads(p_y, h)
    Kxx = np.zeros((NC, NC)); Kyy = np.zeros((NC, NC)); Kxy = np.zeros((NC, NC))
    eps = 1e-15
    for I in range(NC):
        for J in range(NC):
            sl = (slice(I * A, (I + 1) * A), slice(J * A, (J + 1) * A))
            vxx, vyx = vx_x[sl].mean(), vy_x[sl].mean()
            vxy, vyy = vx_y[sl].mean(), vy_y[sl].mean()
            gxx, gyy = gx_x[sl].mean(), gy_y[sl].mean()
            kxx_ = -MU * vxx / (gxx + np.copysign(eps, gxx))
            kyx_ = -MU * vyx / (gxx + np.copysign(eps, gxx))
            kxy_ = -MU * vxy / (gyy + np.copysign(eps, gyy))
            kyy_ = -MU * vyy / (gyy + np.copysign(eps, gyy))
            kxy_ = 0.5 * (kxy_ + kyx_)
            if not np.isfinite(kxx_) or kxx_ <= 0.0:
                kxx_ = float(np.nanmean(k[sl]))
            if not np.isfinite(kyy_) or kyy_ <= 0.0:
                kyy_ = float(np.nanmean(k[sl]))
            kbound = np.sqrt(kxx_ * kyy_)
            if not np.isfinite(kxy_):
                kxy_ = 0.0
            elif abs(kxy_) > kbound:
                kxy_ = float(np.sign(kxy_)) * kbound
            Kxx[I, J], Kyy[I, J], Kxy[I, J] = kxx_, kyy_, kxy_
    return Kxx, Kyy, Kxy
