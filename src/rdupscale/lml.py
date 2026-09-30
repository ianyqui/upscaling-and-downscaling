"""Downscaling por regressão linear de máxima verossimilhança (LML)."""
from __future__ import annotations

import numpy as np

from .config import Grid, LMLParams
from .geostat import correlation_matrix


def poly_basis(xi, eta, degree):
    """Base polinomial {xi^p eta^q}, p+q <= degree (triângulo de Pascal)."""
    cols = [xi ** p * eta ** q
            for d in range(degree + 1)
            for p in range(d + 1) for q in [d - p]]
    return np.column_stack(cols)


def centers_norm(n):
    """Coordenadas (xi, eta) em [-1, 1] dos centros de uma malha n x n."""
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    eta, xi = np.meshgrid(c, c, indexing="ij")
    return xi.ravel(), eta.ravel()


def lml_downscale(field_coarse, grid: Grid = Grid(), degree: int = 4):
    """Ajusta a base polinomial na macromalha e avalia na micromalha.
    Retorna (campo fino nf x nf, variância do resíduo do ajuste)."""
    xi_c, eta_c = centers_norm(grid.nc)
    Phi_c = poly_basis(xi_c, eta_c, degree)
    u = field_coarse.ravel()
    theta, *_ = np.linalg.lstsq(Phi_c, u, rcond=None)
    resid = u - Phi_c @ theta
    sigma2 = float(resid @ resid) / len(u)
    xi_f, eta_f = centers_norm(grid.nf)
    Phi_f = poly_basis(xi_f, eta_f, degree)
    return (Phi_f @ theta).reshape(grid.nf, grid.nf), sigma2


def build_chol_corr_fine(ell, grid: Grid = Grid()):
    """Fator de Cholesky da autocorrelação gaussiana isotrópica (escala ell)."""
    R = correlation_matrix(grid.nf, ell_x=ell, ell_y=ell)
    return np.linalg.cholesky(R + 1e-10 * np.eye(R.shape[0]))


def lml_downscale_stochastic(field_coarse, sigma_res, L_chol, rng,
                             grid: Grid = Grid(), degree: int = 4):
    """Média LML + resíduo gaussiano correlacionado com desvio sigma_res."""
    mu, _ = lml_downscale(field_coarse, grid, degree)
    w = rng.standard_normal(grid.nf * grid.nf)
    xi_unit = (L_chol @ w).reshape(grid.nf, grid.nf)
    return mu + xi_unit * sigma_res


def calibrate_lml_residual(P_micro_array, grid: Grid = Grid(),
                           lml: LMLParams = LMLParams()):
    """(sigma_res, ell_res) dos resíduos micro - LML(média por macrocélula)."""
    NC, A = grid.nc, grid.alpha
    residuos = np.empty_like(P_micro_array)
    for i in range(P_micro_array.shape[0]):
        pmac = P_micro_array[i].reshape(NC, A, NC, A).mean(axis=(1, 3))
        plml, _ = lml_downscale(pmac, grid, lml.degree)
        residuos[i] = P_micro_array[i] - plml
    sigma_res = float(residuos.std())
    rh = residuos - residuos.mean(axis=(1, 2), keepdims=True)
    num = (rh[:, :, :-1] * rh[:, :, 1:]).mean()
    den = (rh ** 2).mean() + lml.var_floor
    rho1 = max(min(num / den, lml.rho_clip_hi), lml.rho_clip_lo)
    ell_res = float(grid.hf / np.sqrt(-np.log(rho1)))
    return sigma_res, ell_res
