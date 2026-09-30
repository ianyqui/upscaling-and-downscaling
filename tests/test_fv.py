import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from rdupscale import tpfa, assemble, face_fluxes, PROBLEMS


def _tpfa_loop(kx, ky, bc, mu=1.0):
    """Montagem célula a célula (tpfa_solve_aniso da tese), usada como referência."""
    n = kx.shape[0]; lam = 1.0 / mu; N = n * n
    idx = lambda i, j: i * n + j
    harm = lambda a, b: 2.0 * a * b / (a + b)
    A = sp.lil_matrix((N, N)); rhs = np.zeros(N)
    for i in range(n):
        for j in range(n):
            c = idx(i, j)
            for di, dj, kk in ((0, 1, kx), (0, -1, kx), (1, 0, ky), (-1, 0, ky)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    T = lam * harm(kk[i, j], kk[ii, jj])
                    A[c, c] += T; A[c, idx(ii, jj)] -= T
    faces = {"baixo": [(0, j) for j in range(n)], "cima": [(n - 1, j) for j in range(n)],
             "esq": [(i, 0) for i in range(n)], "dir": [(i, n - 1) for i in range(n)]}
    for face, (kind, val) in bc.items():
        if kind != "D":
            continue
        kd = kx if face in ("esq", "dir") else ky
        for (i, j) in faces[face]:
            T = lam * kd[i, j] * 2.0
            A[idx(i, j), idx(i, j)] += T; rhs[idx(i, j)] += T * val
    return spla.spsolve(A.tocsr(), rhs).reshape(n, n)


def test_linear_solution_homogeneous():
    n = 12
    k = np.ones((n, n))
    p = tpfa(k, k, PROBLEMS["1.1"], h=1.0)
    exact = 1.0 - (np.arange(n) + 0.5) / n
    assert np.allclose(p, exact[None, :], atol=1e-12)


def test_vectorized_equals_loop():
    rng = np.random.default_rng(3)
    n = 9
    kx = np.exp(rng.normal(size=(n, n))); ky = np.exp(rng.normal(size=(n, n)))
    for bc in PROBLEMS.values():
        assert np.allclose(tpfa(kx, ky, bc, 1.0), _tpfa_loop(kx, ky, bc), atol=1e-12)


def test_matrix_symmetric_positive_definite():
    rng = np.random.default_rng(4)
    n = 8
    k = np.exp(rng.normal(size=(n, n)))
    A, _ = assemble(k, k, PROBLEMS["1.5"], 1.0)
    assert abs(A - A.T).max() < 1e-14
    assert np.linalg.eigvalsh(A.toarray()).min() > 0


def test_mass_conservation():
    rng = np.random.default_rng(5)
    n = 10
    k = np.exp(0.9 * rng.normal(size=(n, n)))
    for bc in PROBLEMS.values():
        p = tpfa(k, k, bc, 1.0)
        Fx, Fy = face_fluxes(p, k, k, bc, 1.0)
        div = (Fx[:, 1:] - Fx[:, :-1]) + (Fy[1:, :] - Fy[:-1, :])
        assert np.abs(div).max() < 1e-10
