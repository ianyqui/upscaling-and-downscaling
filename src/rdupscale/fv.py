"""Volumes finitos com fluxo de dois pontos (TPFA) para -div[(1/mu) K grad p] = 0."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _harm(a, b):
    return 2.0 * a * b / (a + b)


def assemble(kx: np.ndarray, ky: np.ndarray, bc: dict, h: float, mu: float = 1.0):
    """Monta o sistema TPFA (A, b) para K = diag(kx, ky) numa malha n x n de
    células quadradas de lado h. Retorna (A em CSR, b)."""
    n = kx.shape[0]
    lam = 1.0 / mu
    N = n * n
    Tx = lam * _harm(kx[:, :-1], kx[:, 1:])
    Ty = lam * _harm(ky[:-1, :], ky[1:, :])
    diag = np.zeros((n, n))
    diag[:, :-1] += Tx; diag[:, 1:] += Tx
    diag[:-1, :] += Ty; diag[1:, :] += Ty
    rhs = np.zeros((n, n))
    for face, (kind, val) in bc.items():
        if kind != "D":
            continue
        if face == "esq":
            sl, kd = (slice(None), 0), kx[:, 0]
        elif face == "dir":
            sl, kd = (slice(None), -1), kx[:, -1]
        elif face == "baixo":
            sl, kd = (0, slice(None)), ky[0, :]
        elif face == "cima":
            sl, kd = (-1, slice(None)), ky[-1, :]
        else:
            raise ValueError(f"face desconhecida: {face}")
        Tb = lam * kd * 2.0          # meia transmissibilidade até a face
        diag[sl] += Tb
        rhs[sl] += Tb * val
    east = np.zeros((n, n)); east[:, :-1] = -Tx
    west = np.zeros((n, n)); west[:, 1:] = -Tx
    A = sp.diags([diag.ravel(), east.ravel()[:-1], west.ravel()[1:],
                  (-Ty).ravel(), (-Ty).ravel()],
                 [0, 1, -1, n, -n], shape=(N, N), format="csr")
    return A, rhs.ravel()


def tpfa(kx: np.ndarray, ky: np.ndarray, bc: dict, h: float, mu: float = 1.0) -> np.ndarray:
    """Resolve o problema de pressão por TPFA; retorna p com forma (n, n).

    Idêntico a ``tpfa`` da tese. Em células quadradas, h cancela nas
    transmissibilidades; o argumento é mantido por compatibilidade."""
    A, b = assemble(kx, ky, bc, h, mu)
    n = kx.shape[0]
    return spla.spsolve(A, b).reshape(n, n)


def face_fluxes(p, kx, ky, bc, h, mu: float = 1.0):
    """Fluxos TPFA nas faces internas e de contorno (por unidade de espessura).

    Retorna (Fx, Fy): Fx com forma (n, n+1) é o fluxo na direção +x através
    das faces verticais; Fy com forma (n+1, n) é o fluxo na direção +y."""
    n = p.shape[0]
    lam = 1.0 / mu
    Fx = np.zeros((n, n + 1)); Fy = np.zeros((n + 1, n))
    Fx[:, 1:-1] = -lam * _harm(kx[:, :-1], kx[:, 1:]) * (p[:, 1:] - p[:, :-1])
    Fy[1:-1, :] = -lam * _harm(ky[:-1, :], ky[1:, :]) * (p[1:, :] - p[:-1, :])
    for face, (kind, val) in bc.items():
        if kind != "D":
            continue
        if face == "esq":
            Fx[:, 0] = -lam * 2.0 * kx[:, 0] * (p[:, 0] - val)
        elif face == "dir":
            Fx[:, -1] = -lam * 2.0 * kx[:, -1] * (val - p[:, -1])
        elif face == "baixo":
            Fy[0, :] = -lam * 2.0 * ky[0, :] * (p[0, :] - val)
        elif face == "cima":
            Fy[-1, :] = -lam * 2.0 * ky[-1, :] * (val - p[-1, :])
    return Fx, Fy


def grads(p: np.ndarray, h: float):
    """Gradiente por célula: diferenças centradas no interior e laterais na borda."""
    gx = np.zeros_like(p); gy = np.zeros_like(p)
    gx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * h)
    gx[:, 0] = (p[:, 1] - p[:, 0]) / h
    gx[:, -1] = (p[:, -1] - p[:, -2]) / h
    gy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * h)
    gy[0, :] = (p[1, :] - p[0, :]) / h
    gy[-1, :] = (p[-1, :] - p[-2, :]) / h
    return gx, gy


def darcy_velocity(p: np.ndarray, k: np.ndarray, h: float, mu: float = 1.0):
    """Velocidade de Darcy por célula, v = -(1/mu) k grad p (gradiente de ``grads``)."""
    gx, gy = grads(p, h)
    lam = 1.0 / mu
    return -lam * k * gx, -lam * k * gy
