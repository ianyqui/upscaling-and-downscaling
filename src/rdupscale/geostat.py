"""Geração de campos log-normais de permeabilidade na micromalha."""
from __future__ import annotations

import numpy as np

from .config import Grid, GeostatParams


def correlation_matrix(n: int, ell_x: float, ell_y: float) -> np.ndarray:
    """Matriz de autocorrelação (n² x n²) do kernel gaussiano anisotrópico
    rho(dx, dy) = exp[-(dx/ell_x)² - (dy/ell_y)²], com lags em células.
    Índice linear = i*n + j, com i = linha (y) e j = coluna (x)."""
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    coord = np.column_stack([ii.ravel(), jj.ravel()]).astype(float)
    dy = coord[:, 0][:, None] - coord[:, 0][None, :]
    dx = coord[:, 1][:, None] - coord[:, 1][None, :]
    return np.exp(-(dx / ell_x) ** 2 - (dy / ell_y) ** 2)


def sample_permeability(N_R: int, grid: Grid = Grid(),
                        geo: GeostatParams = GeostatParams(),
                        seed=None) -> np.ndarray:
    """N_R realizações log-normais k = k_med * exp(sigma * Z), forma (N_R, nf, nf).

    nf <= 30: Cholesky exato (idêntico à tese). nf > 30: síntese espectral por
    FFT com imersão circulante 2x, como na tese (atenção:
    esse ramo normaliza cada realização e não deve ser usado sem revisão).
    """
    NF = grid.nf
    rng = np.random.default_rng(seed)
    if NF <= 30:
        R = correlation_matrix(NF, geo.ell_x, geo.ell_y)
        L = np.linalg.cholesky(R + 1e-10 * np.eye(R.shape[0]))
        W = rng.standard_normal((NF * NF, N_R))
        Z = (L @ W).T.reshape(N_R, NF, NF)
    else:
        M = 2 * NF
        x = np.arange(M) * 1.0
        dx = np.minimum(x, M - x)
        dy = dx.copy()
        DX, DY = np.meshgrid(dx, dy, indexing="ij")
        R2D = np.exp(-(DX ** 2) / (geo.ell_x ** 2) - (DY ** 2) / (geo.ell_y ** 2))
        S = np.clip(np.fft.fft2(R2D).real, 0.0, None)
        sqrtS = np.sqrt(S / (M * M))
        Z = np.empty((N_R, NF, NF))
        for i in range(N_R):
            W = (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))) / np.sqrt(2.0)
            Z[i] = np.fft.ifft2(sqrtS * W * (M * M)).real[:NF, :NF]
        Z = (Z - Z.mean(axis=(1, 2), keepdims=True)) / (Z.std(axis=(1, 2), keepdims=True) + 1e-12)
    return geo.k_med * np.exp(geo.sigma_lnk * Z)
