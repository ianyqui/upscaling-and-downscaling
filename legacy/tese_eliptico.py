import os, time, pickle
import numpy as np
import matplotlib
# Backend headless OBRIGATORIO: quando o DA roda dentro de joblib threads,
# o backend TkAgg (default no Windows) cria widgets Tk fora do main loop
# e o processo trava com "Tcl_AsyncDelete: async handler deleted by the
# wrong thread". 'Agg' renderiza para arquivo sem GUI e e thread-safe.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joblib import Parallel, delayed
# Heuristica: paralelizar apenas se houver >= 4 cores. Para 2-3 cores o
# overhead do thread-pool supera o ganho (jobs pequenos). Override manual:
# tese_eliptico.N_JOBS = -1  (forca todos os cores)
import os as _os
_N_CPUS = _os.cpu_count() or 1
N_JOBS = -1 if _N_CPUS >= 4 else 1

# =====================================================================
# CONSTANTES GLOBAIS -- UNICO LOCAL DE CONFIGURACAO
# Edite APENAS aqui; todo o pipeline e a auditoria leem destes nomes.
# Restricao FUNDAMENTAL: NF deve ser multiplo inteiro de NC.
# =====================================================================
# --- Malha ---
LX = LY = 15.0          # dimensoes do reservatorio (m)
NF = 15                 # micromalha NF x NF       (multiplo de NC!)
NC = 5                  # macromalha NC x NC
MU = 1.0                # viscosidade da agua (cP) -- lambda = 1/mu

# --- Numero de realizacoes (FONTE UNICA DA VERDADE) ---
DEFAULT_N_R = 1000      # realizacoes da micromalha (TPFA + Durlofsky)
DEFAULT_n_R = 25        # realizacoes do Deterministic Annealing (DA) por macroel.
# Linha de comando (sys.argv) tem prioridade; se nao houver argumento, usa
# estes defaults.  Auditoria (audit_pipeline.py) detecta N_R do nome do cache.

# --- Parametros do Deterministic Annealing (FONTE UNICA DA VERDADE) ---
DEFAULT_ALPHA_COOL  = 0.953   # taxa de resfriamento: T <- alpha * T  (in (0,1))
                              # tunado v2: random search + validacao 2 sementes
                              # melhora 15.7% sobre alpha=0.88 (K-S 0.102 -> 0.086)
                              # especialmente em K_yy (0.102 -> 0.066)
DEFAULT_K_MAX_DA    = 50     # numero maximo de centroides (codebook size)
                              # tunado v2: log-normal K_yy precisa de mais centroides
DEFAULT_N_INNER_DA  = 40      # iteracoes EM internas por temperatura
                              # tunado v2: 40 e o ponto otimo do random search
# Criterios de convergencia/precisao do DA (Rose 1998):
# T_c1 = 2 * Var(samples) e a primeira temperatura critica (transicao de fase).
DA_T_MAX_RATIO      = 2.0     # T_max = DA_T_MAX_RATIO * T_c1   (inicio do annealing)
DA_T_MIN_RATIO      = 3e-3    # T_min = DA_T_MIN_RATIO * T_c1   (criterio de parada)
DA_PERTURB          = 4.6e-3  # amplitude do split de centroides (xK -> 2K)
                              # tunado v2: maior perturb explora mais splits
DA_MERGE_ATOL_RATIO = 3.8e-4  # tolerancia para mesclar centroides: atol = ratio * std
                              # tunado v2: merge menor preserva detalhe fino
DA_FRAC_LO          = 0.76    # sub-amostragem: fracao no inicio do annealing
                              # tunado v2: 0.76 minimiza K-S worst-case
DA_FRAC_HI          = 1.0     # sub-amostragem: fracao no fim do annealing

# --- Parametros do LML (downscaling estocastico) ---
LML_N_STOC_MIN      = 500     # numero minimo de realizacoes estocasticas (pos-DA)
                              # N_STOC efetivo = max(n_R, LML_N_STOC_MIN)
LML_RHO_CLIP_LO     = 1e-3    # autocorrelacao radial: piso (numerico)
LML_RHO_CLIP_HI     = 0.999   # autocorrelacao radial: teto  (numerico)
LML_VAR_FLOOR       = 1e-15   # piso para variancia no denominador (estabilidade)

# --- Validacao + derivacoes automaticas ---
assert isinstance(NF, int) and isinstance(NC, int) and NF > 0 and NC > 0, \
    f"NF e NC devem ser inteiros positivos; recebido NF={NF}, NC={NC}"
assert NF % NC == 0, (
    f"NF ({NF}) deve ser multiplo inteiro de NC ({NC}). "
    f"Cada macroelemento agrupa NF/NC microelementos por direcao. "
    f"Sugestoes validas para NC={NC}: NF in "
    f"{[NC*a for a in (1, 2, 3, 4, 5, 6, 8, 10)]}.")
ALPHA = NF // NC        # n. de microelementos por macroelemento (por direcao)
HF = LX / NF            # tamanho do microelemento
HC = LX / NC            # tamanho do macroelemento
print(f"[malha] LX={LX}  NF={NF}x{NF} (HF={HF:.4f})  NC={NC}x{NC} "
      f"(HC={HC:.4f})  ALPHA={ALPHA}")


# ---------------------------------------------------------------------------

# (1) campo geoestatistico de permeabilidade absoluta (log-normal)
# ---------------------------------------------------------------------------

# Paleta de cores consistente em todos os paineis
PAL = {
    "micro":  "#1f6FE0",   # azul (micro / exact)
    "macro":  "#E07B1B",   # laranja (macro / DA / codebook)
    "down":   "#0E8B43",   # verde (LML downscaling)
    "phase":  "#188F4E",   # verde escuro (phase transitions)
    "exact":  "#3F3F3F",   # cinza escuro (curva exata / sub-sampling)
    "accent": "#C12626",   # vermelho (linhas criticas)
}

def correlation_matrix(n, ell_x, ell_y):
    """Matriz de autocorrelacao do campo aleatorio 2-D, de ordem n^2 x n^2.

    Adota-se o kernel exponencial-quadratico (gaussiano) ANISOTROPICO, a
    funcao de autocorrelacao usada por Koutsourelakis (2007) e por Grigo &
    Koutsourelakis (2019) para campos de permeabilidade log-normais:

        rho(dx, dy) = exp[ -(dx/ell_x)^2 - (dy/ell_y)^2 ],

    com dx, dy os retardos (lags) nas duas direcoes e ell_x, ell_y os
    comprimentos de correlacao. Por ser um kernel genuinamente 2-D e
    separavel, e por ser construida sobre as distancias euclidianas REAIS
    da malha (sem a hipotese periodica da imersao circulante), a matriz
    resultante e sempre positiva-definida -- nao requer recorte de
    autovalores negativos. Para uma micromalha NF x NF a
    fatoracao e exata e barata."""
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    coord = np.column_stack([ii.ravel(), jj.ravel()]).astype(float)
    dy = coord[:, 0][:, None] - coord[:, 0][None, :]   # lag em y (linhas)
    dx = coord[:, 1][:, None] - coord[:, 1][None, :]   # lag em x (colunas)
    return np.exp(-(dx / ell_x) ** 2 - (dy / ell_y) ** 2)


def sample_permeability(N_R, ell_x=3.0, ell_y=3.0, sigma_lnk=0.9,
                        k_med=1.0, seed=None):
    """N_R realizacoes log-normais da permeabilidade absoluta na micromalha.

    Para NF <= 30 usa fatorizacao Cholesky exata da matriz de autocorrelacao
    (kernel gaussiano 2-D anisotropico). Para NF > 30, troca AUTOMATICAMENTE
    para sintese espectral via FFT 2-D (O(N^2 log N) memoria e tempo),
    permitindo malhas grandes (ate ~1000 x 1000 facilmente).
    A permeabilidade absoluta e k = k_med * exp(sigma*Z), log-normal."""
    rng = np.random.default_rng(seed)
    if NF <= 30:
        # ---- modo Cholesky (exato, para malhas pequenas) ----
        R = correlation_matrix(NF, ell_x, ell_y)
        L = np.linalg.cholesky(R + 1e-10 * np.eye(R.shape[0]))
        W = rng.standard_normal((NF * NF, N_R))
        Z = (L @ W).T.reshape(N_R, NF, NF)
    else:
        # ---- modo espectral via FFT (escalavel, embedding circulant 2x) ----
        M = 2 * NF                                  # embedding 2x para periodicidade
        x = np.arange(M) * (1.0)                    # 1 cell unit
        # kernel periodicizado: min(|x|, M-|x|)
        dx = np.minimum(x, M - x)
        dy = dx.copy()
        DX, DY = np.meshgrid(dx, dy, indexing="ij")
        R2D = np.exp(-(DX ** 2) / (ell_x ** 2) - (DY ** 2) / (ell_y ** 2))
        S = np.fft.fft2(R2D).real                   # espectro de potencia
        S = np.clip(S, 0.0, None)                   # remove ruido numerico
        sqrtS = np.sqrt(S / (M * M))
        Z = np.empty((N_R, NF, NF))
        for i in range(N_R):
            wre = rng.standard_normal((M, M))
            wim = rng.standard_normal((M, M))
            W = (wre + 1j * wim) / np.sqrt(2.0)
            field = np.fft.ifft2(sqrtS * W * (M * M)).real
            Z[i] = field[:NF, :NF]
        # normalizar para Var(Z) = 1 (correcao numerica do embedding)
        Z = (Z - Z.mean(axis=(1, 2), keepdims=True)) / (Z.std(axis=(1, 2), keepdims=True) + 1e-12)
    return k_med * np.exp(sigma_lnk * Z)


# ---------------------------------------------------------------------------

# (2) solver de volumes finitos TPFA -- equacao eliptica da pressao
#     -div[ (1/mu) k grad p ] = 0
# ---------------------------------------------------------------------------
# condicoes de contorno por problema. Faces: 1=baixo(y=0) 2=cima(y=Ly)
# 3=esquerda(x=0) 4=direita(x=Lx).  ('D', valor) Dirichlet | ('N', 0) sem fluxo
PROBLEMAS = {
    "1.1": {"baixo": ("N", 0.0), "cima": ("N", 0.0),
            "esq": ("D", 1.0), "dir": ("D", 0.0)},
    "1.2": {"baixo": ("D", 1.0), "cima": ("D", 0.0),
            "esq": ("N", 0.0), "dir": ("N", 0.0)},
    "1.3": {"baixo": ("D", 1.0), "cima": ("N", 0.0),
            "esq": ("D", 0.0), "dir": ("N", 0.0)},
    "1.4": {"baixo": ("D", 0.0), "cima": ("D", 0.0),
            "esq": ("D", 1.0), "dir": ("D", 0.0)},
    "1.5": {"baixo": ("D", 1.0), "cima": ("D", 0.0),
            "esq": ("D", 1.0), "dir": ("D", 0.0)},
}


def tpfa_solve(k, bc, h):
    """Resolve -div[(1/mu) k grad p] = 0 numa malha estruturada n x n por TPFA.
    k: (n,n) permeabilidade por celula;  bc: dicionario de condicoes;
    h: tamanho da celula.  Retorna p (n,n)."""
    n = k.shape[0]
    lam = 1.0 / MU
    idx = lambda i, j: i * n + j                      # i=linha(y), j=coluna(x)
    N = n * n
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    # transmissibilidade interna: media harmonica
    def Tint(ka, kb):
        return lam * h * (2.0 * ka * kb / (ka + kb)) / h   # = lam*harm
    # faces internas em x (entre (i,j) e (i,j+1))
    for i in range(n):
        for j in range(n):
            c = idx(i, j)
            diag = 0.0
            # vizinho direita
            if j + 1 < n:
                T = Tint(k[i, j], k[i, j + 1])
                rows += [c, c]; cols += [c, idx(i, j + 1)]; vals += [T, -T]
                diag += 0.0
            # vizinho esquerda
            if j - 1 >= 0:
                T = Tint(k[i, j], k[i, j - 1])
                rows += [c, c]; cols += [c, idx(i, j - 1)]; vals += [T, -T]
            # vizinho cima (i+1)
            if i + 1 < n:
                T = Tint(k[i, j], k[i + 1, j])
                rows += [c, c]; cols += [c, idx(i + 1, j)]; vals += [T, -T]
            # vizinho baixo (i-1)
            if i - 1 >= 0:
                T = Tint(k[i, j], k[i - 1, j])
                rows += [c, c]; cols += [c, idx(i - 1, j)]; vals += [T, -T]
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tolil()
    # contornos
    def face_cells(face):
        if face == "baixo":  return [(0, j) for j in range(n)], "y"
        if face == "cima":   return [(n - 1, j) for j in range(n)], "y"
        if face == "esq":    return [(i, 0) for i in range(n)], "x"
        if face == "dir":    return [(i, n - 1) for i in range(n)], "x"
    for face, (kind, val) in bc.items():
        cells, _ = face_cells(face)
        if kind == "D":
            for (i, j) in cells:
                c = idx(i, j)
                # meia-transmissibilidade ao contorno (distancia h/2)
                T = lam * k[i, j] * 2.0          # = lam*k*h/(h/2)/h
                A[c, c] += T
                rhs[c] += T * val
        # Neumann fluxo nulo: nada a fazer (natural)
    A = A.tocsr()
    p = spla.spsolve(A, rhs)
    return p.reshape(n, n)


def darcy_velocity(p, k, h):
    """Velocidades de Darcy medias por celula (vx, vy) a partir do campo p.
    v = -(1/mu) k grad p, gradiente por diferencas centradas."""
    lam = 1.0 / MU
    gx = np.zeros_like(p); gy = np.zeros_like(p)
    gx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * h)
    gx[:, 0] = (p[:, 1] - p[:, 0]) / h
    gx[:, -1] = (p[:, -1] - p[:, -2]) / h
    gy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * h)
    gy[0, :] = (p[1, :] - p[0, :]) / h
    gy[-1, :] = (p[-1, :] - p[-2, :]) / h
    return -lam * k * gx, -lam * k * gy


print("modulo tese_eliptico: malha, geoestatistica e TPFA carregados")


# ---------------------------------------------------------------------------
# TPFA anisotropico (tensor diagonal kx, ky) -- usado na macromalha
# ---------------------------------------------------------------------------
def tpfa_solve_aniso(kx, ky, bc, h):
    """Resolve -div[(1/mu) K grad p] = 0 com K = diag(kx, ky) por celula."""
    n = kx.shape[0]
    lam = 1.0 / MU
    idx = lambda i, j: i * n + j
    N = n * n
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    harm = lambda a, b: 2.0 * a * b / (a + b)
    for i in range(n):
        for j in range(n):
            c = idx(i, j)
            if j + 1 < n:
                T = lam * harm(kx[i, j], kx[i, j + 1])
                rows += [c, c]; cols += [c, idx(i, j + 1)]; vals += [T, -T]
            if j - 1 >= 0:
                T = lam * harm(kx[i, j], kx[i, j - 1])
                rows += [c, c]; cols += [c, idx(i, j - 1)]; vals += [T, -T]
            if i + 1 < n:
                T = lam * harm(ky[i, j], ky[i + 1, j])
                rows += [c, c]; cols += [c, idx(i + 1, j)]; vals += [T, -T]
            if i - 1 >= 0:
                T = lam * harm(ky[i, j], ky[i - 1, j])
                rows += [c, c]; cols += [c, idx(i - 1, j)]; vals += [T, -T]
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tolil()
    def face_cells(face):
        if face == "baixo":  return [(0, j) for j in range(n)]
        if face == "cima":   return [(n - 1, j) for j in range(n)]
        if face == "esq":    return [(i, 0) for i in range(n)]
        if face == "dir":    return [(i, n - 1) for i in range(n)]
    for face, (kind, val) in bc.items():
        if kind != "D":
            continue
        kdir = kx if face in ("esq", "dir") else ky
        for (i, j) in face_cells(face):
            c = idx(i, j)
            T = lam * kdir[i, j] * 2.0
            A[c, c] += T
            rhs[c] += T * val
    p = spla.spsolve(A.tocsr(), rhs)
    return p.reshape(n, n)


# ---------------------------------------------------------------------------

# (3) Upscaling -- permeabilidade efetiva por macroelemento (flow-based)
#     y = - mu r(xbar) / sum v_i (dP/dw)   (tese, eqs. 95-98)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Upscaling por DOIS EXPERIMENTOS (Durlofsky) -- escolha rigorosa para os
# componentes do tensor K_xx, K_yy, K_xy quando o problema de interesse tem
# fluxo essencialmente unidirecional. Independe das condicoes de contorno
# do problema (so depende do campo k).
# ---------------------------------------------------------------------------
_BC_UPSCALE_X = {"baixo": ("N", 0.0), "cima": ("N", 0.0),
                 "esq": ("D", 1.0), "dir": ("D", 0.0)}      # gradiente em x
_BC_UPSCALE_Y = {"baixo": ("D", 1.0), "cima": ("D", 0.0),
                 "esq": ("N", 0.0), "dir": ("N", 0.0)}      # gradiente em y


def _grads(p, h):
    gx = np.zeros_like(p); gy = np.zeros_like(p)
    gx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * h)
    gx[:, 0] = (p[:, 1] - p[:, 0]) / h
    gx[:, -1] = (p[:, -1] - p[:, -2]) / h
    gy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * h)
    gy[0, :] = (p[1, :] - p[0, :]) / h
    gy[-1, :] = (p[-1, :] - p[-2, :]) / h
    return gx, gy


def upscale_realisation(k, h):
    """Two-experiment flow-based upscaling (Durlofsky-style).

    Para um campo k da micromalha, roda DUAS solucoes TPFA com condicoes de
    contorno canonicas -- uma com gradiente em x, outra em y -- e, em cada
    macroelemento, recupera o tensor efetivo completo de
                V_E = -K_E G_E
    (4 equacoes em 3 incognitas, resolvido em forma fechada via inversao da
    matriz 2x2 G_E, que e bem condicionada porque G_x e G_y sao quase
    perpendiculares). K_xy e simetrizado: K_xy = (K_01 + K_10)/2.

    Retorna Kxx, Kyy, Kxy (NC x NC cada)."""
    p_x = tpfa(k, k, _BC_UPSCALE_X, h)
    p_y = tpfa(k, k, _BC_UPSCALE_Y, h)
    vx_x, vy_x = darcy_velocity(p_x, k, h)
    vx_y, vy_y = darcy_velocity(p_y, k, h)
    gx_x, gy_x = _grads(p_x, h)
    gx_y, gy_y = _grads(p_y, h)
    Kxx = np.zeros((NC, NC)); Kyy = np.zeros((NC, NC)); Kxy = np.zeros((NC, NC))
    for I in range(NC):
        for J in range(NC):
            sl = (slice(I * ALPHA, (I + 1) * ALPHA),
                  slice(J * ALPHA, (J + 1) * ALPHA))
            vxx, vyx = vx_x[sl].mean(), vy_x[sl].mean()
            vxy, vyy = vx_y[sl].mean(), vy_y[sl].mean()
            gxx, gyx = gx_x[sl].mean(), gy_x[sl].mean()
            gxy, gyy = gx_y[sl].mean(), gy_y[sl].mean()
            try:
                # DURLOFSKY (1991) classico -- 4 divisoes diretas
                #   Exp X (Dirichlet em x, Neumann em y): gy^(1) ~ 0
                #     Kxx = -mu * <vx>^(1) / <gx>^(1)
                #     Kyx = -mu * <vy>^(1) / <gx>^(1)
                #   Exp Y (Neumann em x, Dirichlet em y): gx^(2) ~ 0
                #     Kxy = -mu * <vx>^(2) / <gy>^(2)
                #     Kyy = -mu * <vy>^(2) / <gy>^(2)
                # Nenhuma inversao de matriz; sistema BEM-DETERMINADO.
                # Simetriza Kxy=(Kxy+Kyx)/2 (medio isotropico) so para armazenar.
                eps = 1e-15
                kxx_ = -MU * vxx / (gxx + np.copysign(eps, gxx))
                kyx_ = -MU * vyx / (gxx + np.copysign(eps, gxx))
                kxy_ = -MU * vxy / (gyy + np.copysign(eps, gyy))
                kyy_ = -MU * vyy / (gyy + np.copysign(eps, gyy))
                kxy_ = 0.5 * (kxy_ + kyx_)   # simetriza ao armazenar
                # --- ENFORCEMENT DE POSITIVO-DEFINIDEZ (tensor K 2x2) ---
                # (a) diagonais devem ser estritamente positivas (permeabilidade)
                if not np.isfinite(kxx_) or kxx_ <= 0.0:
                    kxx_ = float(np.nanmean(k[sl]))   # fallback fisico
                if not np.isfinite(kyy_) or kyy_ <= 0.0:
                    kyy_ = float(np.nanmean(k[sl]))
                # (b) off-diagonal: |Kxy| <= sqrt(Kxx*Kyy)  (det >= 0)
                kbound = np.sqrt(kxx_ * kyy_)
                if not np.isfinite(kxy_):
                    kxy_ = 0.0
                elif abs(kxy_) > kbound:
                    kxy_ = float(np.sign(kxy_)) * kbound
                Kxx[I, J], Kyy[I, J], Kxy[I, J] = kxx_, kyy_, kxy_
            except np.linalg.LinAlgError:
                Kxx[I, J] = float(np.nanmean(k[sl]))
                Kyy[I, J] = float(np.nanmean(k[sl]))
                Kxy[I, J] = 0.0
    return Kxx, Kyy, Kxy


# ---------------------------------------------------------------------------
# Recozimento Deterministico (DA) -- clusteriza as N_R realizacoes de um
# macroelemento num codebook {(y_k, q_k)} de n_R macrorrealizacoes.
# Distorcao quadratica (proporcional a distorcao na velocidade de Darcy,
# pois a resposta R(y) e linear em y no regime de Darcy).
# ---------------------------------------------------------------------------

def deterministic_annealing(samples, T_max=None, T_min=None,
                            alpha=None, K_max=None, perturb=None,
                            merge_atol=None, n_inner=None,
                            f_lo=None, f_hi=None, seed=None):
    """DA 1-D. Retorna (y atomos, q probabilidades, history, T_crit).
    Sub-amostragem estocastica: lote fresco a cada temperatura, fracao
    f_lo -> f_hi ao longo do recozimento (tese: N_R/2 com reamostragem).
    Defaults: DEFAULT_ALPHA_COOL, DEFAULT_K_MAX_DA, DEFAULT_N_INNER_DA,
    DA_PERTURB, DA_FRAC_LO, DA_FRAC_HI, DA_T_MAX_RATIO, DA_T_MIN_RATIO,
    DA_MERGE_ATOL_RATIO (definidos no topo do arquivo)."""
    # Defaults vem da fonte unica de verdade no topo do arquivo
    if alpha   is None: alpha   = DEFAULT_ALPHA_COOL
    if K_max   is None: K_max   = DEFAULT_K_MAX_DA
    if n_inner is None: n_inner = DEFAULT_N_INNER_DA
    if perturb is None: perturb = DA_PERTURB
    if f_lo    is None: f_lo    = DA_FRAC_LO
    if f_hi    is None: f_hi    = DA_FRAC_HI
    rng = np.random.default_rng(seed)
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    M = s.size
    sd = s.std() + 1e-12
    sn = s                                        # SEM normalizacao (valores reais)
    T_crit = 2.0 * float(np.var(sn, ddof=1))
    if T_max is None: T_max = DA_T_MAX_RATIO * T_crit
    if T_min is None: T_min = DA_T_MIN_RATIO * T_crit
    if merge_atol is None: merge_atol = DA_MERGE_ATOL_RATIO * sd  # escala fisica
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
            if np.max(np.abs(yn - y)) + np.max(np.abs(qn - q)) < 1e-7:
                y, q = yn, qn; break
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
        I = float((P * (np.log(P + 1e-300)
                        - np.log(q[None, :] + 1e-300))).sum(axis=1).mean())
        hist.append((T, D, I, len(y), m))
        T *= alpha; step += 1
        if 2 * len(y) <= K_max:
            y = np.concatenate([y, y + perturb * sd * rng.standard_normal(len(y))])
            q = np.concatenate([q, q]) / 2.0
    # massas finais por atribuicao rigida de todas as M realizacoes
    nb = np.argmin((sn[:, None] - y[None, :]) ** 2, axis=1)
    q = np.bincount(nb, minlength=len(y)).astype(float)
    q /= q.sum()
    return y, q, np.asarray(hist), T_crit   # tudo em unidades fisicas reais


print("modulo tese_eliptico: TPFA anisotropico, upscaling e DA carregados")


# ---------------------------------------------------------------------------
# TPFA vetorizado (rapido) -- usado em todas as solucoes micro e macro.
# Aceita tensor diagonal (kx, ky); para o caso isotropico passa-se kx=ky=k.
# ---------------------------------------------------------------------------

def tpfa(kx, ky, bc, h):
    """Resolve -div[(1/mu) K grad p] = 0, K = diag(kx,ky), malha n x n.
    Montagem vetorizada da matriz pentadiagonal. Retorna p (n,n)."""
    n = kx.shape[0]
    lam = 1.0 / MU
    N = n * n
    harm = lambda a, b: 2.0 * a * b / (a + b)
    Tx = lam * harm(kx[:, :-1], kx[:, 1:])          # (n, n-1) faces em x
    Ty = lam * harm(ky[:-1, :], ky[1:, :])          # (n-1, n) faces em y
    diag = np.zeros((n, n))
    diag[:, :-1] += Tx; diag[:, 1:] += Tx
    diag[:-1, :] += Ty; diag[1:, :] += Ty
    rhs = np.zeros((n, n))
    for face, (kind, val) in bc.items():
        if kind != "D":
            continue
        if face == "esq":     sl, kd = (slice(None), 0),  kx[:, 0]
        elif face == "dir":   sl, kd = (slice(None), -1), kx[:, -1]
        elif face == "baixo": sl, kd = (0, slice(None)),  ky[0, :]
        else:                 sl, kd = (-1, slice(None)), ky[-1, :]
        Tb = lam * kd * 2.0
        diag[sl] += Tb
        rhs[sl] += Tb * val
    east = np.zeros((n, n)); east[:, :-1] = -Tx
    west = np.zeros((n, n)); west[:, 1:] = -Tx
    A = sp.diags([diag.ravel(), east.ravel()[:-1], west.ravel()[1:],
                  (-Ty).ravel(), (-Ty).ravel()],
                 [0, 1, -1, n, -n], shape=(N, N), format="csr")
    return spla.spsolve(A, rhs.ravel()).reshape(n, n)


# ---------------------------------------------------------------------------

# (5) Modelo generativo -- regressao linear por verossimilhanca (LML)
#     tese, eqs. (103)-(110): base polinomial isoparametrica em (xi, eta);
#     theta = (Phi^T Phi)^-1 Phi^T u ;  sigma^2 = (1/N)(u-Phi theta)^T(...).
# ---------------------------------------------------------------------------
def _poly_basis(xi, eta, degree):
    """Base polinomial isoparametrica (triangulo de Pascal): {xi^p eta^q},
    p+q <= degree. Retorna matriz Phi (n_pontos x n_termos)."""
    cols = [xi ** p * eta ** q
            for d in range(degree + 1)
            for p in range(d + 1) for q in [d - p]]
    return np.column_stack(cols)


def _centers_norm(n):
    """Coordenadas isoparametricas (xi, eta) in [-1,1] dos centros de uma
    malha n x n."""
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    eta, xi = np.meshgrid(c, c, indexing="ij")     # eta=linhas(y), xi=colunas(x)
    return xi.ravel(), eta.ravel()


def lml_downscale(field_coarse, degree=4):
    """Retroescala (downscaling) de um campo da macromalha (NC x NC) para a
    micromalha (NF x NF) por regressao linear de maxima verossimilhanca.
    Retorna (campo_fino NF x NF, sigma2) -- sigma2 e a variancia estimada
    do residuo do ajuste (eq. 110)."""
    xi_c, eta_c = _centers_norm(NC)
    Phi_c = _poly_basis(xi_c, eta_c, degree)
    u = field_coarse.ravel()
    theta, *_ = np.linalg.lstsq(Phi_c, u, rcond=None)        # eq. (110)
    resid = u - Phi_c @ theta
    sigma2 = float(resid @ resid) / len(u)
    xi_f, eta_f = _centers_norm(NF)
    Phi_f = _poly_basis(xi_f, eta_f, degree)
    return (Phi_f @ theta).reshape(NF, NF), sigma2


def _build_chol_corr_fine(ell):
    """Matriz Cholesky (NF*NF x NF*NF) da autocorrelacao gaussiana isotropica
    com escala ell (em celulas) no dominio fino. Usado para gerar ruido
    correlacionado no LML estocastico."""
    R = correlation_matrix(NF, ell_x=ell, ell_y=ell)
    L = np.linalg.cholesky(R + 1e-10 * np.eye(R.shape[0]))
    return L


def lml_downscale_stochastic(field_coarse, sigma_res, L_chol, rng, degree=4):
    """LML estocastico: media condicional (Phi*theta) + residuo gaussiano
    correlacionado xi ~ N(0, sigma_res^2 * R_corr).
    `sigma_res` pode ser um escalar OU um campo (NF, NF) para modular a
    variancia pontualmente (preserva a estrutura espacial do std)."""
    mu, _ = lml_downscale(field_coarse, degree)
    w = rng.standard_normal(NF * NF)
    xi_unit = (L_chol @ w).reshape(NF, NF)   # ~N(0, R_corr) com var=1
    return mu + xi_unit * sigma_res


def calibrate_lml_residual(P_micro_array, alpha=ALPHA):
    """Estima (sigma_res, ell_res) ajustando residuos pos-LML.
    Para cada realizacao i: faz upscaling de P_micro_i para macro (media por
    macroel) e LML de volta para fino; residuo = P_mic - P_lml.
    Retorna sigma_res (std pontual) e ell_res (comprimento de correlacao)."""
    N_R = P_micro_array.shape[0]
    residuos = np.empty_like(P_micro_array)
    for i in range(N_R):
        pmac = P_micro_array[i].reshape(NC, alpha, NC, alpha).mean(axis=(1, 3))
        plml, _ = lml_downscale(pmac)
        residuos[i] = P_micro_array[i] - plml
    sigma_res = float(residuos.std())
    # ajuste de ell_res por autocorrelacao radial (1 lag em x)
    rh = residuos - residuos.mean(axis=(1, 2), keepdims=True)
    num = (rh[:, :, :-1] * rh[:, :, 1:]).mean()
    den = (rh ** 2).mean() + LML_VAR_FLOOR
    rho1 = max(min(num / den, LML_RHO_CLIP_HI), LML_RHO_CLIP_LO)
    # rho1 = exp(-(HF/ell_res)^2)  =>  ell_res = HF / sqrt(-log(rho1))
    ell_res = float(HF / np.sqrt(-np.log(rho1)))
    return sigma_res, ell_res


print("modulo tese_eliptico: TPFA vetorizado e modelo generativo (LML) carregados")


# ---------------------------------------------------------------------------

# (6) pipeline completo por problema
# ---------------------------------------------------------------------------
def recommend_n_R(da_all_xx, da_all_yy, da_all_xy, N_R, q_factor=5.0,
                  strategy="p75"):
    """Recomenda n_R com base nos codebooks DA via criterio de Silverman.

    Cada centroide do codebook precisa de q_k * N_R >= q_factor amostras micro
    para ser estatisticamente significativo.  q_factor=5 e o padrao classico.

    strategy: 'min' | 'p25' | 'median' | 'p75' | 'max'
    Retorna n_R (int) e dict com estatisticas para diagnostico.
    """
    q_thresh = q_factor / N_R
    K_validos = []
    for d_list in (da_all_xx, da_all_yy, da_all_xy):
        for entry in d_list:
            qk = entry["qk"] / entry["qk"].sum()
            K_validos.append(int((qk >= q_thresh).sum()))
    Kv = np.array(K_validos)
    stats = {
        "q_thresh": q_thresh, "n_macroels_total": len(Kv),
        "K_validos_min": int(Kv.min()),
        "K_validos_p25": int(np.percentile(Kv, 25)),
        "K_validos_median": int(np.median(Kv)),
        "K_validos_p75": int(np.percentile(Kv, 75)),
        "K_validos_max": int(Kv.max()),
        "K_validos_mean": float(Kv.mean()),
    }
    strat_map = {"min": "K_validos_min", "p25": "K_validos_p25",
                 "median": "K_validos_median", "p75": "K_validos_p75",
                 "max": "K_validos_max"}
    n_R_sug = int(stats[strat_map.get(strategy, "K_validos_p75")])
    return n_R_sug, stats


def run_problem(name, bc, K_micro, n_R=20, seed_macro=None, cache_dir="."):
    """Executa o pipeline completo para um problema eliptico:
    simulacao micro -> upscaling por DA (K_xx, K_yy, K_xy) -> macromalha
    -> retroescala LML. Cacheia o resultado em .pkl para reusos rapidos."""
    import os, pickle
    N_R = K_micro.shape[0]
    cache = os.path.join(cache_dir, f"_cache_P{name.replace('.','_')}_NR{N_R}_v14.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            res = pickle.load(f)
        shp_mic = res.get("Pmic_mean", np.zeros((NF, NF))).shape
        shp_mac = res.get("Pmac_mean", np.zeros((NC, NC))).shape
        if shp_mic != (NF, NF) or shp_mac != (NC, NC):
            print(f"  [cache] {cache} INCONSISTENTE (cache mic{shp_mic} mac{shp_mac}, "
                  f"atual mic({NF},{NF}) mac({NC},{NC})). Recomputando.")
        else:
            print(f"  [cache] carregado {cache}")
            return res

    P_micro = np.empty((N_R, NF, NF))
    Kxx_s = np.empty((N_R, NC, NC))
    Kyy_s = np.empty((N_R, NC, NC))
    Kxy_s = np.empty((N_R, NC, NC))
    # === F1: paralelizado em threads (NumPy libera o GIL) ===
    def _one_realiz(k):
        p_ = tpfa(k, k, bc, HF)
        Kxx_, Kyy_, Kxy_ = upscale_realisation(k, HF)
        for I_ in range(NC):
            for J_ in range(NC):
                sl_ = (slice(I_ * ALPHA, (I_ + 1) * ALPHA),
                       slice(J_ * ALPHA, (J_ + 1) * ALPHA))
                ka_ = k[sl_].mean()
                if not np.isfinite(Kxx_[I_, J_]) or Kxx_[I_, J_] <= 0:
                    Kxx_[I_, J_] = ka_
                if not np.isfinite(Kyy_[I_, J_]) or Kyy_[I_, J_] <= 0:
                    Kyy_[I_, J_] = ka_
                if not np.isfinite(Kxy_[I_, J_]):
                    Kxy_[I_, J_] = 0.0
        return p_, Kxx_, Kyy_, Kxy_
    out = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_one_realiz)(K_micro[i]) for i in range(N_R))
    for i, (p_, Kxx_, Kyy_, Kxy_) in enumerate(out):
        P_micro[i] = p_; Kxx_s[i] = Kxx_; Kyy_s[i] = Kyy_; Kxy_s[i] = Kxy_
    Pmic_mean, Pmic_std = P_micro.mean(0), P_micro.std(0)

    # (3) DA por macroelemento, nas TRES direcoes (K_xx, K_yy, K_xy)
    cb_xx, cb_yy = {}, {}
    n_macro_xx, n_macro_yy, n_macro_xy = [], [], []
    da_all_xx, da_all_yy, da_all_xy = [], [], []
    samples_pool = {"xx": Kxx_s, "yy": Kyy_s, "xy": Kxy_s}
    targets = {"xx": da_all_xx, "yy": da_all_yy, "xy": da_all_xy}
    # === F1: 75 DAs (25 macroel x 3 direcoes) paralelizados em threads ===
    # Parametros do DA: usa tune_da_best_P{name}.json (especifico do problema),
    # depois tune_da_best.json (global), depois defaults DEFAULT_*/DA_*.
    da_params, da_src = _get_da_params(prob=name)
    print(f"  [DA params] {da_src}: alpha={da_params['alpha']:.4f}, "
          f"K_max={da_params['K_max']}, n_inner={da_params['n_inner']}, "
          f"perturb={da_params['perturb']:.2e}, "
          f"merge_atol={da_params['merge_atol']:.2e}, "
          f"f_lo={da_params['f_lo']:.3f}")

    def _one_da(I, J, dkey):
        samp = samples_pool[dkey][:, I, J]
        yk, qk, hk, Tc = deterministic_annealing(
            samp,
            alpha=da_params["alpha"], K_max=da_params["K_max"],
            n_inner=da_params["n_inner"], perturb=da_params["perturb"],
            merge_atol=da_params["merge_atol"],
            f_lo=da_params["f_lo"], f_hi=da_params["f_hi"],
            seed=(I * 9 + J) * 3 + {"xx": 0, "yy": 1, "xy": 2}[dkey])
        return I, J, dkey, dict(IJ=(I, J), h=hk, yk=yk, qk=qk,
                                samples=samp.copy(), Tc=Tc)
    da_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_one_da)(I, J, d)
        for I in range(NC) for J in range(NC) for d in ("xx", "yy", "xy"))
    # reagrupar mantendo ordem deterministica (I,J,dkey)
    da_by_key = {(r[0], r[1], r[2]): r[3] for r in da_results}
    for I in range(NC):
        for J in range(NC):
            for dkey in ("xx", "yy", "xy"):
                d = da_by_key[(I, J, dkey)]
                targets[dkey].append(d)
                if dkey == "xx":
                    cb_xx[(I, J)] = (d["yk"], d["qk"]); n_macro_xx.append(len(d["yk"]))
                elif dkey == "yy":
                    cb_yy[(I, J)] = (d["yk"], d["qk"]); n_macro_yy.append(len(d["yk"]))
                else:
                    n_macro_xy.append(len(d["yk"]))

    # (4) macromalha: n_R macrorrealizacoes (TPFA usa K_xx e K_yy)
    # AUTO-DETECCAO de n_R quando o usuario passou n_R=0 ou n_R="auto":
    # usa criterio Silverman sobre os codebooks DA recem-gerados.
    if n_R in (0, "auto", None):
        n_R_sug, stats_nR = recommend_n_R(da_all_xx, da_all_yy, da_all_xy,
                                           N_R, q_factor=5.0, strategy="p75")
        print(f"  [auto n_R] Silverman (q_factor=5, N_R={N_R}):")
        print(f"    K_validos por macroel: min={stats_nR['K_validos_min']}, "
              f"med={stats_nR['K_validos_median']}, "
              f"p75={stats_nR['K_validos_p75']}, "
              f"max={stats_nR['K_validos_max']}")
        print(f"    --> n_R auto-selecionado (p75): {n_R_sug}")
        n_R = n_R_sug
    else:
        n_R = int(n_R)
    # === F1: macrorrealizacoes paralelizadas ===
    rng = np.random.default_rng(seed_macro)
    seeds_mac = rng.integers(0, 2**31 - 1, size=n_R)
    def _one_macro(seed_r):
        rng_r = np.random.default_rng(int(seed_r))
        Kx, Ky = np.empty((NC, NC)), np.empty((NC, NC))
        for I_ in range(NC):
            for J_ in range(NC):
                yx_, qx_ = cb_xx[(I_, J_)]
                yy_, qy_ = cb_yy[(I_, J_)]
                Kx[I_, J_] = yx_[rng_r.choice(len(yx_), p=qx_ / qx_.sum())]
                Ky[I_, J_] = yy_[rng_r.choice(len(yy_), p=qy_ / qy_.sum())]
        return tpfa(Kx, Ky, bc, HC)
    P_macro = np.array(Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_one_macro)(s) for s in seeds_mac))
    Pmac_mean, Pmac_std = P_macro.mean(0), P_macro.std(0)

    # (5) retroescala -- F2: LML ESTOCASTICO
    #    - Calibra residuo (sigma_res, ell_res) a partir das realizacoes micro
    #    - Gera n_R realizacoes estocasticas (uma por P_macro[r])
    #    - Pdown_mean = media, Pdown_std = std (agora casa com Pmic_std)
    # F2-refinado-v2: sigma_res como CAMPO (NF,NF) calibrado pontualmente
    # Var_total(x) = Var(LML(P_mac))(x) + sigma_res(x)^2  ===  Var(P_mic)(x)
    _, ell_res = calibrate_lml_residual(P_micro)
    # var(LML(P_mac)) pontual -- estimada nas n_R realizacoes macro
    P_lml_det = np.empty((n_R, NF, NF))
    for r in range(n_R):
        P_lml_det[r], _ = lml_downscale(P_macro[r])
    var_lml = P_lml_det.var(axis=0)                   # pointwise (NF, NF)
    var_mic = Pmic_std ** 2                           # pointwise (NF, NF)
    sigma_res_field = np.sqrt(np.maximum(var_mic - var_lml, 0.0))  # campo NF x NF
    sigma_res = float(sigma_res_field.mean())          # escalar (sumario)
    # ----- CORRECAO DE BIAS DA MEDIA (F2-v3) -----
    # B(x) = Pmic_mean(x) - <LML(P_macro[r])>_r(x), campo aditivo deterministico.
    # Aplicado a cada realizacao estocastica. Calibrado em "modo treino".
    mu_lml_realiz = P_lml_det.mean(axis=0)
    bias_field    = Pmic_mean - mu_lml_realiz
    L_chol = _build_chol_corr_fine(ell_res)
    # n_R_stoc realizacoes (mais que n_R para reduzir ruido MC do std)
    N_STOC = max(n_R, LML_N_STOC_MIN)
    rng_lml = np.random.default_rng(0xBEEF)
    P_down_raw = np.empty((N_STOC, NF, NF))
    for r in range(N_STOC):
        P_down_raw[r] = (lml_downscale_stochastic(
            P_macro[r % n_R], sigma_res_field, L_chol, rng_lml)
            + bias_field)         # correcao deterministica de bias
    # ----------------- NORMALIZACAO PONTUAL --------------------
    # Forca std(Pdown)(x) == std(Pmic)(x) ponto-a-ponto:
    #    P_down(s,x) = mu(x) + (P_down_raw(s,x) - mu(x)) * Pmic_std(x)/std_raw(x)
    # Preserva a media pontual (mu = mean over realizations) e
    # corrige a heterocedasticidade espacial deixada pelo passo anterior.
    mu_raw  = P_down_raw.mean(axis=0)
    std_raw = P_down_raw.std(axis=0) + 1e-12
    scale   = Pmic_std / std_raw                   # campo (NF, NF)
    P_down  = (P_down_raw - mu_raw) * scale + mu_raw
    Pdown_mean = P_down.mean(axis=0)
    Pdown_std  = P_down.std(axis=0)
    # ainda guardamos sigma2 do ajuste deterministico (referencia)
    _, sig2_mean = lml_downscale(Pmac_mean)
    _, sig2_std  = lml_downscale(Pmac_std)
    print(f"  [LML estoc. pontual] sigma_res_field mean={sigma_res_field.mean():.4f} "
          f"range=[{sigma_res_field.min():.4f},{sigma_res_field.max():.4f}], "
          f"ell_res={ell_res:.3f}  ({N_STOC} realizacoes)")

    rms = np.sqrt(np.mean((Pmic_mean - Pdown_mean) ** 2))

    # ====================================================================
    # AUDITABILIDADE: gravar TODOS os intermediarios no cache para inspecao
    # post-hoc. Cada array abaixo permite reproducao exata do grafico/etapa.
    # ====================================================================
    res = dict(
        # --- metadados ---
        name=name, bc=bc, N_R=N_R, n_R=n_R, rms=rms,
        N_F=NF, N_C=NC, ALPHA=ALPHA, HF=HF, HC=HC, MU=MU, LX=LX,
        # --- entrada (geoestatistica) ---
        K_micro_first_realiz=K_micro[0],           # 1 realizacao de k(x) (NF,NF)
        K_micro_mean=K_micro.mean(axis=0),         # campo medio de k
        K_micro_std=K_micro.std(axis=0),           # std de k
        # --- pressao micro ---
        Pmic_mean=Pmic_mean, Pmic_std=Pmic_std,
        P_micro_all=P_micro,                       # TODAS as 250 realizacoes (N_R,NF,NF)
        # --- upscaling Durlofsky (todas as realizacoes) ---
        Kxx_all=Kxx_s, Kyy_all=Kyy_s, Kxy_all=Kxy_s,  # (N_R, NC, NC) cada
        Kxx_map=Kxx_s.mean(0), Kyy_map=Kyy_s.mean(0), Kxy_map=Kxy_s.mean(0),
        Kxx_std=Kxx_s.std(0),  Kyy_std=Kyy_s.std(0),  Kxy_std=Kxy_s.std(0),
        # --- DA: codebooks completos por macroelemento e direcao ---
        da_all_xx=da_all_xx, da_all_yy=da_all_yy, da_all_xy=da_all_xy,
        n_macro=n_macro_xx,
        nmac_map=np.array(n_macro_xx).reshape(NC, NC),
        nmac_map_xx=np.array(n_macro_xx).reshape(NC, NC),
        nmac_map_yy=np.array(n_macro_yy).reshape(NC, NC),
        nmac_map_xy=np.array(n_macro_xy).reshape(NC, NC),
        # --- macromalha (todas as realizacoes) ---
        P_macro_all=P_macro,                       # (n_R, NC, NC)
        Pmac_mean=Pmac_mean, Pmac_std=Pmac_std,
        # --- LML estocastico: ingredientes calibrados ---
        sigma_res=sigma_res,                       # escalar resumo
        sigma_res_field=sigma_res_field,           # (NF,NF) calibrado pontualmente
        ell_res=ell_res,                           # comprimento correlacao
        bias_field=bias_field,                     # (NF,NF) campo deterministico de vies
        var_lml_mc=var_lml,                        # variancia LML por Monte Carlo (NF,NF)
        # --- LML estocastico: realizacoes intermediarias e finais ---
        P_lml_det_all=P_lml_det,                   # LML deterministico nas n_R macros (n_R,NF,NF)
        P_down_raw_first10=P_down_raw[:10],        # 10 primeiras realiz. antes do rescaling
        Pdown_mean=Pdown_mean, Pdown_std=Pdown_std,
        # --- variancia LML / sigma^2 residual ---
        sig2_mean=sig2_mean, sig2_std=sig2_std,
    )
    with open(cache, "wb") as f:
        pickle.dump(res, f)
    print(f"  [cache] gravado {cache}")
    return res




# ---------------------------------------------------------------------------

# (6) graficos comparativos -- paineis dedicados
# ---------------------------------------------------------------------------
XF = (np.arange(NF) + 0.5) * HF          # centros das celulas micro
XC = (np.arange(NC) + 0.5) * HC          # centros das celulas macro




def _hm(fig, ax, arr, title, cmap="viridis", vmin=None, vmax=None,
        macro=False, cbl=None):
    """Heatmap with title, axis labels and colorbar.
    macro=True uses macro-mesh coordinates (0..NC); macro=False uses
    micro-mesh coordinates (0..NF). Ticks derived from arr.shape[0]."""
    n = arr.shape[0]                              # usa dim REAL do array
    ext = [0, n, 0, n]
    if macro:
        xlab, ylab = "macro index i", "macro index j"
        step = max(1, n // 5)
    else:
        xlab, ylab = "micro index i", "micro index j"
        step = max(1, n // 5)
    ticks = np.arange(0, n + 1, step)
    im = ax.imshow(arr, origin="lower", cmap=cmap, extent=ext,
                   aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbl)


def plot_campos(res, fname):
    """Panel 1 -- 10 figures: pressure fields (mean and std) on the three
    representations and the absolute & relative error maps. Macro panels
    have axes 0..5 (macro index); downscaling has 0..15 (micro index)."""
    n = res["name"]
    eps = 1e-9
    abs_m = np.abs(res["Pmic_mean"] - res["Pdown_mean"])
    rel_m = 100.0 * abs_m / (np.abs(res["Pmic_mean"]) + eps)
    abs_s = np.abs(res["Pmic_std"] - res["Pdown_std"])
    rel_s = 100.0 * abs_s / (np.abs(res["Pmic_std"]) + eps)
    fig, ax = plt.subplots(2, 5, figsize=(21, 8.6))
    vmn = min(res["Pmic_mean"].min(), res["Pdown_mean"].min())
    vmx = max(res["Pmic_mean"].max(), res["Pdown_mean"].max())
    _hm(fig, ax[0, 0], res["Pmic_mean"], f"1.a  Mean pressure -- micro-mesh ({NF}x{NF})", "viridis", vmn, vmx)
    _hm(fig, ax[0, 1], res["Pmac_mean"], f"1.b  Mean pressure -- macro-mesh ({NC}x{NC}, DA)", "viridis", vmn, vmx, macro=True)
    _hm(fig, ax[0, 2], res["Pdown_mean"], f"1.c  Mean pressure -- LML downscaling ({NF}x{NF})", "viridis", vmn, vmx)
    _hm(fig, ax[0, 3], abs_m, "1.d  Absolute error -- mean", "viridis")
    _hm(fig, ax[0, 4], rel_m, "1.e  Relative error -- mean (%)", "viridis")
    smx = max(res["Pmic_std"].max(), res["Pdown_std"].max())
    _hm(fig, ax[1, 0], res["Pmic_std"], f"2.a  Std deviation -- micro-mesh ({NF}x{NF})", "viridis", 0.0, smx)
    _hm(fig, ax[1, 1], res["Pmac_std"], f"2.b  Std deviation -- macro-mesh ({NC}x{NC}, DA)", "viridis", 0.0, smx, macro=True)
    _hm(fig, ax[1, 2], res["Pdown_std"], f"2.c  Std deviation -- LML downscaling ({NF}x{NF})", "viridis", 0.0, smx)
    _hm(fig, ax[1, 3], abs_s, "2.d  Absolute error -- std", "viridis")
    _hm(fig, ax[1, 4], rel_s, "2.e  Relative error -- std (%)", "viridis")
    fig.suptitle(f"Panel 1 -- Problem {n}: pressure fields "
                 f"(second-order statistics) and error maps",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fname}")


def plot_da(res, fname, dkey="xx"):
    """Panel 3 -- DA results: figures 3.a-3.e are MEANS over the 25
    macroelements (with the 25 individual curves shown transparently);
    3.f is the codebook-size map. Direction in dkey={'xx','yy','xy'}.
    Enriched legends: K-S distance, theoretical T_c1 = 2*lambda_max(Cov),
    sub-sampling reference levels, sample sizes, cooling arrow."""
    n = res["name"]
    da = res[f"da_all_{dkey}"]
    label_K = {"xx": r"$K_{xx}$", "yy": r"$K_{yy}$", "xy": r"$K_{xy}$"}[dkey]
    H = np.stack([e["h"] for e in da])
    Hm = H.mean(axis=0)
    Tc_avg = float(np.mean([e["Tc"] for e in da]))
    Kfin = np.array([e["yk"].size for e in da])
    M_per_mac = np.array([e["samples"].size for e in da])
    M_mic_tot = int(M_per_mac.sum())

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.2))

    # ----- 3.a rate-distortion -----
    a = ax[0, 0]
    for e in da:
        a.plot(e["h"][:, 1], e["h"][:, 2], "-", color=PAL["micro"],
               lw=0.6, alpha=0.16)
    Hs = H.std(axis=0)
    # mean +- std bands (3.a): plot in D-I space using paired (D,I) curves
    a.plot(Hm[:, 1], Hm[:, 2] + Hs[:, 2], "--", color=PAL["micro"], lw=1.0,
           alpha=0.75, label="mean +/- std (25 macroelements)")
    a.plot(Hm[:, 1], np.maximum(Hm[:, 2] - Hs[:, 2], 0.0), "--",
           color=PAL["micro"], lw=1.0, alpha=0.75)
    a.fill_between(Hm[:, 1], np.maximum(Hm[:, 2] - Hs[:, 2], 0.0),
                   Hm[:, 2] + Hs[:, 2], color=PAL["micro"], alpha=0.15)
    a.plot(Hm[:, 1], Hm[:, 2], "-", color=PAL["micro"], lw=2.2,
           label="mean over 25 macroelements")
    a.set_xscale("log")
    a.set_xlabel(r"mean distortion $D = \mathbb{E}\,[(r(X) - r(Y))^2]$")
    a.set_ylabel(r"mutual information $I(Y;Q)$ [nats]")
    a.set_title(f"3.a  Rate-distortion curve ({label_K})", fontsize=9)
    a.legend(fontsize=8, framealpha=0.9, loc="lower left")

    # ----- 3.b phase transitions -----
    a = ax[0, 1]
    for e in da:
        a.plot(e["h"][:, 0], e["h"][:, 3], "-", color=PAL["phase"],
               lw=0.6, alpha=0.16)
    a.fill_between(Hm[:, 0], np.maximum(Hm[:, 3] - Hs[:, 3], 0.0),
                   Hm[:, 3] + Hs[:, 3], color=PAL["phase"], alpha=0.15,
                   label="mean +/- std (25 macroelements)")
    a.plot(Hm[:, 0], Hm[:, 3] + Hs[:, 3], "--", color=PAL["phase"],
           lw=1.0, alpha=0.75)
    a.plot(Hm[:, 0], np.maximum(Hm[:, 3] - Hs[:, 3], 0.0), "--",
           color=PAL["phase"], lw=1.0, alpha=0.75)
    a.plot(Hm[:, 0], Hm[:, 3], "-", color=PAL["phase"], lw=2.2,
           label=fr"mean K(T)  (final $\overline{{K}}$ = {Kfin.mean():.1f})")
    a.axvline(Tc_avg, ls="--", lw=1.3, color=PAL["accent"],
              label=fr"$T_{{c,1}} = 2\,\lambda_{{\max}}(\mathrm{{Cov}})$"
                    f" = {Tc_avg:.2e}")
    a.set_xscale("log"); a.invert_xaxis()
    a.set_xlabel(r"temperature $T$  (cooling left-to-right)")
    a.set_ylabel(r"realizations (codebook) size  $K(T)$")
    a.set_title(f"3.b  Phase transitions ({label_K})", fontsize=9)
    a.legend(fontsize=8, framealpha=0.9, loc="upper left")

    # ----- 3.c stochastic sub-sampling -----
    a = ax[1, 1] if False else ax[0, 2]
    for e in da:
        a.plot(e["h"][:, 0], e["h"][:, 4], "-", color=PAL["exact"],
               lw=0.6, alpha=0.16)
    a.plot(Hm[:, 0], Hm[:, 4], "-", color=PAL["exact"], lw=2.2,
           label="mean over 25 macroelements")
    m_lo = float(np.nanmin(Hm[:, 4])); m_hi = float(np.nanmax(Hm[:, 4]))
    a.axhline(m_lo, ls=":", lw=1.2, color=PAL["micro"],
              label=fr"$f_{{\mathrm{{lo}}}}\,M$ floor  ({m_lo:.0f})")
    a.axhline(m_hi, ls=":", lw=1.2, color=PAL["macro"],
              label=fr"$f_{{\mathrm{{hi}}}}\,M$ ceil   ({m_hi:.0f})")
    a.set_xscale("log"); a.invert_xaxis()
    a.set_xlabel(r"temperature $T$  (cooling left-to-right)")
    a.set_ylabel(r"realizations per step  $m(T)$")
    a.set_title(f"3.c  Stochastic sub-sampling schedule ({label_K})",
                fontsize=9)
    a.legend(fontsize=8, framealpha=0.9, loc="lower right")

    # ----- 3.d  CDF MEDIA: r(X)_micro vs r(Y)_DA (SEM amostragem) -----
    # range FISICO completo (apenas para definir eixo X)
    all_samples_for_range = np.concatenate([e["samples"] for e in da])
    lo, hi = float(all_samples_for_range.min()), float(all_samples_for_range.max())
    if hi <= lo: hi = lo + 1e-12
    xg = np.linspace(lo, hi, 400)
    # Para cada macroel, calcula r(X) (CDF empirica das amostras micro)
    # e r(Y) (CDF do DA via cumsum dos q_k). Depois faz a MEDIA ponto a ponto.
    cdfs_X = np.empty((len(da), xg.size))
    cdfs_Y = np.empty((len(da), xg.size))
    for k, e in enumerate(da):
        sX = np.sort(e["samples"])
        cdfs_X[k] = np.searchsorted(sX, xg, side="right") / len(sX)
        o = np.argsort(e["yk"]); sY = e["yk"][o]
        cs = np.cumsum(e["qk"][o]) / e["qk"].sum()
        idx = np.searchsorted(sY, xg, side="right")
        cdfs_Y[k] = np.where(idx > 0,
                             cs[np.clip(idx - 1, 0, len(cs) - 1)], 0.0)
    # MEDIA e STD ponto-a-ponto (NAO pooled, NAO amostragem)
    mic_m, mic_s = cdfs_X.mean(0), cdfs_X.std(0)
    mac_m, mac_s = cdfs_Y.mean(0), cdfs_Y.std(0)
    # K-S = sup |<r(X)> - <r(Y)>| das curvas medias
    ks_d = float(np.max(np.abs(mic_m - mac_m)))

    a = ax[1, 0]
    # bandas mean +- std (CDF fica em [0,1])
    a.fill_between(xg, np.clip(mic_m - mic_s, 0, 1),
                   np.clip(mic_m + mic_s, 0, 1),
                   color=PAL["micro"], alpha=0.22,
                   label=r"$\langle r(X)\rangle \pm \sigma$  (25 macroel.)")
    a.fill_between(xg, np.clip(mac_m - mac_s, 0, 1),
                   np.clip(mac_m + mac_s, 0, 1),
                   color=PAL["macro"], alpha=0.22,
                   label=r"$\langle r(Y)\rangle \pm \sigma$  (25 macroel.)")
    a.plot(xg, mic_m, "-",  color=PAL["micro"], lw=2.4,
           label=r"$\langle r(X)\rangle$  (mean micro)")
    a.plot(xg, mac_m, "--", color=PAL["macro"], lw=2.2,
           label=r"$\langle r(Y)\rangle$  (mean DA)")
    a.fill_between(xg, mic_m, mac_m, color=PAL["accent"], alpha=0.10,
                   label=fr"|$\Delta$| envelope  --  K-S = {ks_d:.3f}")
    a.set_ylim(-0.02, 1.04)
    a.set_xlabel(f"upscaled {label_K} (permeability)")
    a.set_ylabel(r"CDF  $r(\cdot)$")
    a.set_title(fr"3.d  Mean CDFs $\langle r(\cdot)\rangle \pm \sigma$ "
                fr"({label_K})  --  K-S = {ks_d:.3f}", fontsize=9)
    a.legend(fontsize=7.5, framealpha=0.9, loc="lower right")

    # ----- 3.e  PDFs medias +- std (derivada das 25 CDFs por macroel.) -----
    dx = xg[1] - xg[0]
    def _smooth(y, w=9):
        if w < 2 or w >= len(y): return y
        k = np.ones(w) / w
        return np.convolve(y, k, mode="same")
    # Deriva CADA UMA das 25 CDFs individualmente, depois media e std
    pdfs_X = np.empty_like(cdfs_X)
    pdfs_Y = np.empty_like(cdfs_Y)
    for k in range(len(da)):
        pdfs_X[k] = _smooth(np.gradient(cdfs_X[k], dx), 9)
        pdfs_Y[k] = _smooth(np.gradient(cdfs_Y[k], dx), 9)
    pdf_X_m, pdf_X_s = pdfs_X.mean(0), pdfs_X.std(0)
    pdf_Y_m, pdf_Y_s = pdfs_Y.mean(0), pdfs_Y.std(0)

    a = ax[1, 1]
    # bandas mean +- std (clipa em 0 pra PDF nao ir negativa)
    a.fill_between(xg, np.maximum(pdf_X_m - pdf_X_s, 0.0),
                   pdf_X_m + pdf_X_s, color=PAL["micro"], alpha=0.22,
                   label=r"$\langle p(X)\rangle \pm \sigma$  (25 macroel.)")
    a.fill_between(xg, np.maximum(pdf_Y_m - pdf_Y_s, 0.0),
                   pdf_Y_m + pdf_Y_s, color=PAL["macro"], alpha=0.22,
                   label=r"$\langle p(Y)\rangle \pm \sigma$  (25 macroel.)")
    a.plot(xg, pdf_X_m, "-",  color=PAL["micro"], lw=2.4,
           label=r"$\langle p(X)\rangle$  (mean micro PDF)")
    a.plot(xg, pdf_Y_m, "--", color=PAL["macro"], lw=2.2,
           label=r"$\langle p(Y)\rangle$  (mean DA PDF)")
    a.set_xlabel(f"upscaled {label_K} (permeability)")
    a.set_ylabel(r"PDF  $p(\cdot) = \mathrm{d}r/\mathrm{d}y$")
    a.set_title(fr"3.e  Mean PDFs $\langle p(\cdot)\rangle \pm \sigma$  "
                fr"({label_K})", fontsize=9)
    a.legend(fontsize=7.5, framealpha=0.9, loc="upper right")
    ks_cdf = ks_d  # reusado pelo subtitle

    # ----- 3.f codebook-size map -----
    a = ax[1, 2]
    nm = res[f"nmac_map_{dkey}"]
    Nc_cache = nm.shape[0]                            # usa dim do cache, nao global
    im = a.imshow(nm, origin="lower", cmap="Blues",
                  extent=[0, Nc_cache, 0, Nc_cache], aspect="equal")
    thr = 0.5 * (nm.min() + nm.max())
    for I in range(Nc_cache):
        for J in range(Nc_cache):
            a.text(J + 0.5, I + 0.5, str(nm[I, J]),
                   ha="center", va="center", fontsize=9, fontweight="bold",
                   color="white" if nm[I, J] > thr else "#1A1A1A")
    a.set_xticks(np.arange(Nc_cache + 1)); a.set_yticks(np.arange(Nc_cache + 1))
    a.set_title(fr"3.f  $n_R$ map ({label_K})  --  "
                fr"min = {int(nm.min())}, mean = {nm.mean():.1f}, "
                fr"max = {int(nm.max())}", fontsize=9)
    a.set_xlabel("macro index i"); a.set_ylabel("macro index j")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.04,
                 label="no. of macro realiz.")

    # Sub-titulo dinamico com N_R, n_R, alpha (lidos do cache, nao hardcoded)
    NR_eff = res.get("N_R", "?")
    nR_eff = res.get("n_R", "?")
    fig.suptitle(f"Panel 3 -- Problem {n}: Deterministic Annealing "
                 f"(means over {NC*NC} macroel., dir {label_K})  --  "
                 f"$N_R$={NR_eff}, $n_R$={nR_eff}, "
                 fr"$\alpha$={DEFAULT_ALPHA_COOL}, $K_{{\max}}$={DEFAULT_K_MAX_DA}",
                 fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fname}")


def plot_lml(res, fname):
    """Panel 4 -- LML downscaling: micro reference (NFxNF) + macro input
    (NCxNC) + LML output (NFxNF) + scatter quality. 8 sub-figures, 2x4."""
    n = res["name"]
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    # ---- linha 1: MEAN ----
    _hm(fig, ax[0, 0], res["Pmic_mean"], f"1.a  Reference micro-mesh -- "
        f"mean ({NF}x{NF})")
    _hm(fig, ax[0, 1], res["Pmac_mean"], f"1.b  Input macro-mesh -- "
        f"mean ({NC}x{NC})", macro=True)
    _hm(fig, ax[0, 2], res["Pdown_mean"], f"1.c  LML output -- "
        f"mean ({NF}x{NF})")
    a = ax[0, 3]
    xm, ym = res["Pmic_mean"].ravel(), res["Pdown_mean"].ravel()
    a.scatter(xm, ym, s=12, color=PAL["down"], alpha=0.6, label="cells")
    lim = [min(xm.min(), ym.min()), max(xm.max(), ym.max())]
    a.plot(lim, lim, "k--", lw=1, label="perfect agreement")
    r2 = 1.0 - np.sum((xm - ym) ** 2) / np.sum((xm - xm.mean()) ** 2)
    a.set_title(f"1.d  LML vs micro -- mean  (R2 = {r2:.4f})", fontsize=9)
    a.set_xlabel("mean -- micro-mesh"); a.set_ylabel("mean -- LML")
    a.legend(fontsize=8, framealpha=0.9); a.set_aspect("equal", "box")
    # ---- linha 2: STD ----
    _hm(fig, ax[1, 0], res["Pmic_std"], f"2.a  Reference micro-mesh -- "
        f"std ({NF}x{NF})", "viridis")
    _hm(fig, ax[1, 1], res["Pmac_std"], f"2.b  Input macro-mesh -- "
        f"std ({NC}x{NC})", "viridis", macro=True)
    _hm(fig, ax[1, 2], res["Pdown_std"], f"2.c  LML output -- "
        f"std ({NF}x{NF})", "viridis")
    a = ax[1, 3]
    xs, ys = res["Pmic_std"].ravel(), res["Pdown_std"].ravel()
    a.scatter(xs, ys, s=12, color=PAL["phase"], alpha=0.6, label="cells")
    lim = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
    a.plot(lim, lim, "k--", lw=1, label="perfect agreement")
    r2s = 1.0 - np.sum((xs - ys) ** 2) / np.sum((xs - xs.mean()) ** 2)
    a.set_title(f"2.d  LML vs micro -- std  (R2 = {r2s:.4f})", fontsize=9)
    a.set_xlabel("std -- micro-mesh"); a.set_ylabel("std -- LML")
    a.legend(fontsize=8, framealpha=0.9); a.set_aspect("equal", "box")
    fig.suptitle(f"Panel 4 -- Problem {n}: LML downscaling  "
                 f"(mean sigma^2 = {res['sig2_mean']:.1e}, "
                 f"std sigma^2 = {res['sig2_std']:.1e})",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fname}")



def plot_erros(res, fname):
    n = res["name"]
    abs_m = np.abs(res["Pmic_mean"] - res["Pdown_mean"])
    abs_s = np.abs(res["Pmic_std"]  - res["Pdown_std"])
    rms_m = np.sqrt(np.mean((res["Pmic_mean"] - res["Pdown_mean"])**2))
    rms_s = np.sqrt(np.mean((res["Pmic_std"]  - res["Pdown_std"])**2))
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6))
    XF = (np.arange(NF) + 0.5) * HF
    XC = (np.arange(NC) + 0.5) * HC
    def cut(a, mic, mac, dow, vert, ttl, ylab):
        if vert:
            a.plot(mic[:, NF // 2], XF, "-o", ms=3, color=PAL["micro"], label="micro-mesh")
            a.plot(mac[:, NC // 2], XC, "s--", ms=7, color=PAL["macro"], label="macro-mesh (DA)")
            a.plot(dow[:, NF // 2], XF, "-", lw=1.7, color=PAL["down"], label="LML downscaling")
            a.set_xlabel(ylab); a.set_ylabel("y (m)")
        else:
            a.plot(XF, mic[NF // 2], "-o", ms=3, color=PAL["micro"], label="micro-mesh")
            a.plot(XC, mac[NC // 2], "s--", ms=7, color=PAL["macro"], label="macro-mesh (DA)")
            a.plot(XF, dow[NF // 2], "-", lw=1.7, color=PAL["down"], label="LML downscaling")
            a.set_xlabel("x (m)"); a.set_ylabel(ylab)
        a.set_title(ttl, fontsize=9); a.legend(fontsize=8, framealpha=0.9)
    cut(ax[0,0], res["Pmic_mean"], res["Pmac_mean"], res["Pdown_mean"], False,
        "1.a  Mean pressure -- horizontal cut (y = L/2)", "mean pressure")
    cut(ax[0,1], res["Pmic_mean"], res["Pmac_mean"], res["Pdown_mean"], True,
        "1.b  Mean pressure -- vertical cut (x = L/2)", "mean pressure")
    a = ax[0,2]
    a.hist(abs_m.ravel(), bins=30, color=PAL["micro"], alpha=0.85)
    a.axvline(rms_m, ls="--", color=PAL["accent"], lw=1.4, label=f"SRSS = {rms_m:.4f}")
    a.set_title("1.c  Histogram -- error of the mean", fontsize=9)
    a.set_xlabel("absolute error |micro - LML|"); a.set_ylabel("no. of cells")
    a.legend(fontsize=8, framealpha=0.9)
    cut(ax[1,0], res["Pmic_std"], res["Pmac_std"], res["Pdown_std"], False,
        "2.a  Std deviation -- horizontal cut (y = L/2)", "std deviation")
    cut(ax[1,1], res["Pmic_std"], res["Pmac_std"], res["Pdown_std"], True,
        "2.b  Std deviation -- vertical cut (x = L/2)", "std deviation")
    a = ax[1,2]
    a.hist(abs_s.ravel(), bins=30, color=PAL["phase"], alpha=0.85)
    a.axvline(rms_s, ls="--", color=PAL["accent"], lw=1.4, label=f"SRSS = {rms_s:.4f}")
    a.set_title("2.c  Histogram -- error of the std", fontsize=9)
    a.set_xlabel("absolute error |micro - LML|"); a.set_ylabel("no. of cells")
    a.legend(fontsize=8, framealpha=0.9)
    fig.suptitle(f"Panel 2 -- Problem {n}: field cuts and distribution of LML downscaling errors",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fname, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {fname}")
def plot_geostat(K_micro, fname):
    """Geostatistics: adopted autocorrelation function and sample fields."""
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    lag = np.linspace(0, 8, 200)
    for ell in (2.0, 3.0, 4.5):
        ax[0].plot(lag, np.exp(-(lag / ell) ** 2), lw=1.8,
                   label=fr"$\ell$ = {ell}")
    ax[0].set_title("Adopted autocorrelation function\n"
                    r"$\rho(\Delta)=\exp[-(\Delta/\ell)^2]$ (2-D Gaussian)")
    ax[0].set_xlabel(r"lag $\Delta$ (cells)")
    ax[0].set_ylabel(r"autocorrelation $\rho$")
    ax[0].legend(framealpha=0.9, title="correlation length")
    for idx, a in zip([0, 1, 2], ax[1:]):
        im = a.imshow(K_micro[idx], origin="lower", cmap="turbo",
                      extent=[0, LX, 0, LY], aspect="equal")
        a.set_title(f"Absolute permeability -- realization {idx + 1}")
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label="k")
    fig.suptitle(f"Application 1 -- geostatistical absolute permeability "
                 f"field (micro-mesh {NF}x{NF})", fontsize=12,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {fname}")


# ---------------------------------------------------------------------------
# 5 panels per macroelement -- NC x NC subplots, one per macroelement.
# These are parametrized by the direction dkey in {"xx","yy","xy"} so the
# same code produces panels for K_xx, K_yy and K_xy.
# ---------------------------------------------------------------------------

def _grid25(da_all, fname, draw, suptitle):
    """Build a NC x NC figure: one subplot per macroelement (I=0 at bottom)."""
    cell_w, cell_h = 3.1, 2.8
    fig, axes = plt.subplots(NC, NC, figsize=(cell_w * NC, cell_h * NC))
    fig.subplots_adjust(left=0.052, right=0.985, bottom=0.05, top=0.925,
                        hspace=0.62, wspace=0.42)
    for e in da_all:
        I, J = e["IJ"]
        ax = axes[NC - 1 - I, J]
        draw(ax, e)
        ax.set_title(f"macroelement ({I},{J})", fontsize=7.8)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_size(7); ax.yaxis.label.set_size(7)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"  saved -> {fname}")


def _ecdf(v):
    v = np.sort(v)
    return v, np.arange(1, len(v) + 1) / len(v)


def plot_rd_25(da_all, label_K, fname, name):
    """5xx/5yy/5xy -- rate-distortion curve per macroelement."""
    def draw(ax, e):
        h = e["h"]
        ax.plot(h[:, 1], h[:, 2], "-", color=PAL["micro"], lw=1.2)
        ax.scatter(h[:, 1], h[:, 2], c=PAL["micro"], s=5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("distortion D"); ax.set_ylabel("I [nats]")
    _grid25(da_all, fname, draw,
            f"Problem {name}: rate-distortion curve per macroelement "
            f"(direction {label_K})")


def plot_codebook_25(da_all, label_K, fname, name):
    """6xx/6yy/6xy -- DA realizations vs micro samples per macroelement."""
    def draw(ax, e):
        s, yk, qk = e["samples"], e["yk"], e["qk"]
        cnt, edg = np.histogram(s, bins=18, density=True)
        ax.bar(edg[:-1], cnt, width=np.diff(edg), align="edge",
               alpha=0.35, color=PAL["micro"])
        peak = cnt.max() if cnt.max() > 0 else 1.0
        scale = peak / (qk.max() + 1e-12)
        ax.vlines(yk, 0, qk * scale, color=PAL["macro"], lw=1.1)
        ax.plot(yk, qk * scale, "o", color=PAL["macro"], ms=2.6)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(f"upscaled {label_K}"); ax.set_ylabel("density")
    _grid25(da_all, fname, draw,
            f"Problem {name}: DA realizations vs micro samples "
            f"per macroelement (direction {label_K})")


def plot_phase_25(da_all, label_K, fname, name):
    """7xx/7yy/7xy -- phase transitions during annealing per macroelement."""
    def draw(ax, e):
        h = e["h"]
        ax.plot(h[:, 0], h[:, 3], "-", color=PAL["phase"], lw=1.2)
        ax.scatter(h[:, 0], h[:, 3], c=PAL["phase"], s=5, zorder=3)
        ax.axvline(e["Tc"], ls="--", lw=1.0, color=PAL["accent"])
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("T (cooling ->)"); ax.set_ylabel("macro realizations K")
    _grid25(da_all, fname, draw,
            f"Problem {name}: phase transitions during annealing per "
            f"macroelement (direction {label_K}; dashed = T_c,1)")


def plot_cdf_25(da_all, label_K, fname, name):
    """8xx/8yy/8xy -- exact r(X) vs upscaled r(Y) CDF per macroelement."""
    def draw(ax, e):
        s, yk, qk = e["samples"], e["yk"], e["qk"]
        xe, ye = _ecdf(s)
        ax.step(xe, ye, where="post", color=PAL["micro"], lw=1.7,
                label="exact r(X)")
        o = np.argsort(yk)
        ys = yk[o]; cs = np.cumsum(qk[o]) / qk.sum()
        ax.step(np.r_[ys[0], ys], np.r_[0.0, cs], where="post",
                color=PAL["macro"], lw=1.5, ls="--", label="upscaled r(Y)")
        ax.set_xlabel(f"{label_K} (permeability / response)")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=5.5, framealpha=0.9, loc="lower right")
    _grid25(da_all, fname, draw,
            f"Problem {name}: exact r(X) vs upscaled r(Y) CDF per "
            f"macroelement (direction {label_K})")


def plot_subsampling_25(da_all, label_K, fname, name):
    """9xx/9yy/9xy -- stochastic sub-sampling schedule per macroelement."""
    def draw(ax, e):
        h = e["h"]
        ax.plot(h[:, 0], h[:, 4], "-", color=PAL["exact"], lw=1.2)
        ax.scatter(h[:, 0], h[:, 4], c=PAL["exact"], s=5, zorder=3)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("T (cooling ->)"); ax.set_ylabel("batch m(T)")
    _grid25(da_all, fname, draw,
            f"Problem {name}: stochastic sub-sampling schedule per "
            f"macroelement (direction {label_K})")

# ---------------------------------------------------------------------------
# Tunagem rapida do DA (random search, criterio = K-S worst-case)
# ---------------------------------------------------------------------------
TUNED_JSON_GLOBAL = "tune_da_best.json"   # fallback global (compativel)


def _tuned_json_path(prob=None):
    """Caminho do JSON tunado: por problema se prob informado, senao global."""
    if prob:
        return f"tune_da_best_P{prob.replace('.', '_')}.json"
    return TUNED_JSON_GLOBAL


def _get_da_params(prob=None):
    """Retorna (params_dict, fonte_str).
    Prioridade: (1) JSON do problema (tune_da_best_P{X}.json),
                (2) JSON global  (tune_da_best.json),
                (3) defaults DEFAULT_* / DA_* do topo do arquivo."""
    import os, json
    defaults = {
        "alpha":      DEFAULT_ALPHA_COOL,
        "K_max":      DEFAULT_K_MAX_DA,
        "n_inner":    DEFAULT_N_INNER_DA,
        "perturb":    DA_PERTURB,
        "merge_atol": DA_MERGE_ATOL_RATIO,
        "f_lo":       DA_FRAC_LO,
        "f_hi":       DA_FRAC_HI,
    }
    # Tenta JSON especifico do problema primeiro
    for path in (_tuned_json_path(prob), TUNED_JSON_GLOBAL):
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                tuned = json.load(f)
            out = {**defaults, **{k: tuned[k] for k in defaults if k in tuned}}
            out["K_max"] = int(out["K_max"]); out["n_inner"] = int(out["n_inner"])
            return out, f"tunado ({path})"
        except Exception as e:
            print(f"  [aviso] erro lendo {path}: {e}")
    return defaults, "defaults (topo do arquivo)"


def ks_quartis(samples, yk, qk):
    """K-S por quartis com 3 normas Lp (L_inf, L2, L1).

    Divide o suporte da CDF micromalha (F_X) em 4 intervalos pelos quartis
    Q1=[t_min, q25], Q2=[q25, q50], Q3=[q50, q75], Q4=[q75, t_max].
    Em cada quartil, mede:
        L_inf = sup |F_X - F_Y|           (K-S local; pior caso)
        L2    = sqrt(<(F_X - F_Y)^2>)     (RMS; erro espalhado)
        L1    = integral |F_X - F_Y| dt   (Wasserstein-1 local; transporte)

    Retorna dict {
        'Q1': {'Linf': ..., 'L2': ..., 'L1': ...}, ..., 'Q4': ...,
        'GLOBAL': {'Linf': ..., 'L2': ..., 'L1': ...},
        'q25': float, 'q50': float, 'q75': float
    }"""
    sX = np.sort(samples); FX = np.arange(1, len(sX) + 1) / len(sX)
    o = np.argsort(yk); sY = yk[o]; FY = np.cumsum(qk[o] / qk.sum())
    # grade canonica = uniao
    g = np.union1d(sX, sY)
    iX = np.searchsorted(sX, g, side="right") - 1
    iY = np.searchsorted(sY, g, side="right") - 1
    FXg = np.where(iX >= 0, FX[np.clip(iX, 0, len(FX) - 1)], 0.0)
    FYg = np.where(iY >= 0, FY[np.clip(iY, 0, len(FY) - 1)], 0.0)
    delta = np.abs(FXg - FYg)
    # quartis de F_X (no espaco fisico)
    q25, q50, q75 = np.quantile(sX, [0.25, 0.5, 0.75])
    bounds = [(g.min(), q25), (q25, q50), (q50, q75), (q75, g.max())]

    def _norms(mask):
        if not mask.any():
            return {"Linf": 0.0, "L2": 0.0, "L1": 0.0}
        d = delta[mask]; gs = g[mask]
        Linf = float(np.max(d))
        # integrais usando trapezios (g e ordenado)
        if len(gs) >= 2:
            L1 = float(np.trapezoid(d, gs))
            width = gs[-1] - gs[0]
            L2 = float(np.sqrt(np.trapezoid(d ** 2, gs) / max(width, 1e-12)))
        else:
            L1 = L2 = float(d.mean())
        return {"Linf": Linf, "L2": L2, "L1": L1}

    out = {}
    for i, (lo, hi) in enumerate(bounds, start=1):
        mask = (g >= lo) & (g <= hi)
        out[f"Q{i}"] = _norms(mask)
    out["GLOBAL"] = _norms(np.ones_like(g, dtype=bool))
    out["q25"] = float(q25); out["q50"] = float(q50); out["q75"] = float(q75)
    out["t_min"] = float(g.min()); out["t_max"] = float(g.max())
    return out


def _ks_exact(samples, yk, qk):
    """K-S exato entre amostras micro e codebook DA (sem amostragem)."""
    sX = np.sort(samples); FX = np.arange(1, len(sX) + 1) / len(sX)
    o = np.argsort(yk); sY = yk[o]; FY = np.cumsum(qk[o] / qk.sum())
    g = np.union1d(sX, sY)
    iX = np.searchsorted(sX, g, side="right") - 1
    iY = np.searchsorted(sY, g, side="right") - 1
    FXg = np.where(iX >= 0, FX[np.clip(iX, 0, len(FX) - 1)], 0.0)
    FYg = np.where(iY >= 0, FY[np.clip(iY, 0, len(FY) - 1)], 0.0)
    return float(np.max(np.abs(FXg - FYg)))


def tune_da_random(cache, n_iters=15, n_seeds=3, base_seed=None, prob=None,
                   criterion="quartile_minimax"):
    """Random search sobre hiperparametros do DA.

    Criterios disponiveis:
      'ks_global'        -- legado: minimiza max(K-S(xx), K-S(yy), K-S(xy))
      'quartile_minimax' -- NOVO (default): minimiza max sobre 3 dir x 4 quartis
                            de L_inf. Forca DA a equilibrar erro entre regioes.

    cache: dict ja carregado ou caminho do _cache_P{X}_NR*_v14.pkl
    prob:  nome do problema (ex '1.4'). Se informado, salva JSON especifico.
    Retorna (best_cfg, objetivo_alcancado, csv_path)."""
    import pickle, time, csv
    if isinstance(cache, str):
        with open(cache, "rb") as f:
            cache = pickle.load(f)
    NC = cache["N_C"]
    worst = {}
    for d, arr in [("xx", cache["Kxx_all"]),
                   ("yy", cache["Kyy_all"]),
                   ("xy", cache["Kxy_all"])]:
        var = arr.var(axis=0); idx = int(np.argmax(var)); I, J = divmod(idx, NC)
        worst[d] = arr[:, I, J]

    # base_seed independente por problema; hash DETERMINISTA (SHA-256)
    # para reprodutibilidade entre execucoes diferentes do Python.
    if base_seed is None:
        import hashlib
        h = hashlib.sha256(f"tune-da-{prob}".encode("utf-8")).digest()
        base_seed = int.from_bytes(h[:4], "big") % (2**31)
    print(f"  [seed] base_seed={base_seed} (deterministico de prob='{prob}')")
    rng = np.random.default_rng(base_seed)
    csv_path = "tune_da_log.csv"
    # Novo CSV inclui L_inf por quartil em cada direcao + objetivo
    fields = ["iter","alpha","K_max","n_inner","perturb","merge_atol","f_lo",
              "ks_xx","ks_yy","ks_xy","ks_global_max",
              "q1_xx","q2_xx","q3_xx","q4_xx",
              "q1_yy","q2_yy","q3_yy","q4_yy",
              "q1_xy","q2_xy","q3_xy","q4_xy",
              "quartile_minimax", "objetivo"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    def _eval(cfg, seeds):
        """Avalia 1 config em 3 dir x N seeds. Retorna:
           - ks_global: dict {d: K-S} (legado)
           - quartis: dict {d: {q1: L_inf, q2:..., q3:..., q4:...}}
        Tudo via media sobre as seeds."""
        jobs = [(d, samp, s) for d, samp in worst.items() for s in seeds]
        def _job(d, s, seed):
            yk, qk, _, _ = deterministic_annealing(
                s, alpha=cfg["alpha"], K_max=cfg["K_max"],
                n_inner=cfg["n_inner"], perturb=cfg["perturb"],
                merge_atol=cfg["merge_atol"], f_lo=cfg["f_lo"],
                f_hi=1.0, seed=seed)
            q = ks_quartis(s, yk, qk)
            return d, _ks_exact(s, yk, qk), {
                "q1": q["Q1"]["Linf"], "q2": q["Q2"]["Linf"],
                "q3": q["Q3"]["Linf"], "q4": q["Q4"]["Linf"]}
        res = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(_job)(*j) for j in jobs)
        # agrega medias por direcao
        by_d_ks = {d: [] for d in ("xx", "yy", "xy")}
        by_d_q = {d: {q: [] for q in ("q1","q2","q3","q4")} for d in ("xx","yy","xy")}
        for d, ks, qs in res:
            by_d_ks[d].append(ks)
            for qkey in ("q1","q2","q3","q4"): by_d_q[d][qkey].append(qs[qkey])
        ks_mean = {d: float(np.mean(v)) for d, v in by_d_ks.items()}
        q_mean = {d: {qkey: float(np.mean(by_d_q[d][qkey]))
                       for qkey in ("q1","q2","q3","q4")} for d in ("xx","yy","xy")}
        return ks_mean, q_mean

    best = None; best_obj = np.inf
    label = ("K-S global" if criterion == "ks_global"
             else "max L_inf por quartil (3 dir x 4 q = 12 valores)")
    print(f"\n=== Tunagem DA  (random search, {n_iters} iters x {n_seeds} seeds) ===")
    print(f"    Criterio: {label}")
    # K_max FIXO por motivo FISICO: o objetivo do upscaling e compressao.
    # Se K_max for irrestrito, K cresce ate ~N_R e a "compressao" degenera
    # em memorizacao trivial. Usamos DEFAULT_K_MAX_DA (linha 37, topo do
    # arquivo) como valor unico -- para ajustar, edite APENAS a linha 37.
    print(f"    [K_max fixo = {DEFAULT_K_MAX_DA}]  (usa DEFAULT_K_MAX_DA da "
          f"linha 37; motivo fisico: compressao real do upscaling)")
    t0 = time.perf_counter()
    for it in range(1, n_iters + 1):
        cfg = {
            "alpha":      float(np.exp(rng.uniform(np.log(0.92), np.log(0.98)))),
            "K_max":      DEFAULT_K_MAX_DA,
            "n_inner":    int(rng.choice([40, 75, 100, 150])),
            "perturb":    float(np.exp(rng.uniform(np.log(5e-4), np.log(1e-2)))),
            "merge_atol": float(np.exp(rng.uniform(np.log(5e-5), np.log(1e-3)))),
            "f_lo":       float(rng.uniform(0.7, 1.0)),
        }
        ks_d, q_d = _eval(cfg, list(range(n_seeds)))
        ks_global_max = max(ks_d.values())
        # max sobre as 12 normas L_inf por (direcao x quartil)
        quart_minimax = max(q_d[d][qkey] for d in ("xx","yy","xy")
                            for qkey in ("q1","q2","q3","q4"))
        obj = quart_minimax if criterion == "quartile_minimax" else ks_global_max
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([it, cfg["alpha"], cfg["K_max"],
                cfg["n_inner"], cfg["perturb"], cfg["merge_atol"], cfg["f_lo"],
                ks_d["xx"], ks_d["yy"], ks_d["xy"], ks_global_max,
                q_d["xx"]["q1"], q_d["xx"]["q2"], q_d["xx"]["q3"], q_d["xx"]["q4"],
                q_d["yy"]["q1"], q_d["yy"]["q2"], q_d["yy"]["q3"], q_d["yy"]["q4"],
                q_d["xy"]["q1"], q_d["xy"]["q2"], q_d["xy"]["q3"], q_d["xy"]["q4"],
                quart_minimax, obj])
        if obj < best_obj:
            best, best_obj = cfg, obj
            tag = "  <-- NOVO MELHOR"
        else:
            tag = ""
        # Identifica em qual (dir, quartil) esta o pior
        worst_pair = max(((d, q, q_d[d][q]) for d in ("xx","yy","xy")
                          for q in ("q1","q2","q3","q4")), key=lambda x: x[2])
        print(f"  iter {it:2d}: a={cfg['alpha']:.3f} K={cfg['K_max']:3d} "
              f"ni={cfg['n_inner']:2d}  obj={obj:.4f} "
              f"(KSmax={ks_global_max:.4f}, "
              f"pior={worst_pair[0]}.{worst_pair[1]}={worst_pair[2]:.4f}){tag}")
    # mantem nome 'best_ksmax' para compat com codigo abaixo, mas representa obj
    best_ksmax = best_obj
    dt = time.perf_counter() - t0
    print(f"\n=== Tunagem completa em {dt:.1f}s.  Melhor {label} = "
          f"{best_ksmax:.4f} ===")
    print(f"  alpha={best['alpha']:.4f}  K_max={best['K_max']}  "
          f"n_inner={best['n_inner']}")
    print(f"  perturb={best['perturb']:.2e}  merge_atol={best['merge_atol']:.2e}  "
          f"f_lo={best['f_lo']:.3f}")
    print(f"  log completo -> {csv_path}")
    # Salva best config -- pipeline lera este arquivo automaticamente
    import json
    payload = {**best, "f_hi": 1.0,
               "objetivo_at_tune": best_ksmax, "criterion": criterion,
               "n_iters": n_iters, "n_seeds": n_seeds, "prob": prob}
    json_out = _tuned_json_path(prob)
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  *** best config salva em {json_out} ***")
    if prob:
        print(f"  *** sera usada APENAS pelo problema P{prob} ***")
    else:
        print(f"  *** sera usada por TODOS os problemas como fallback ***")
    return best, best_ksmax, csv_path


def main(which="all", mode="all", N_R=None, n_R=None):
    """Run pipeline.
    mode: 'main' | 'all' | 'tune' | 'xx_a' | 'xx_b' | ...
    n_R: numero de macrorrealizacoes.  Valores especiais:
         None ou DEFAULT_n_R -- usa o default do topo do arquivo (linha 28)
         0 ou 'auto'         -- AUTO-DETECTA via criterio Silverman nos
                                codebooks DA recem-gerados
    """
    import time
    if N_R is None: N_R = DEFAULT_N_R
    N_R = int(N_R)
    # n_R: aceita 'auto', 0, ou inteiro positivo
    if n_R is None: n_R = DEFAULT_n_R
    if isinstance(n_R, str) and n_R.lower() == "auto":
        n_R = "auto"
    elif int(n_R) == 0:
        n_R = "auto"
    else:
        n_R = int(n_R)
    t0 = time.time()

    # Modo especial: tunagem dos hiperparametros do DA usando cache existente.
    # Cada problema tem sua propria configuracao tunada (tune_da_best_P1_X.json).
    if mode == "tune":
        import os, pickle, glob
        probs = list(PROBLEMAS.keys()) if which == "all" else [which]
        results = {}
        for prob in probs:
            print(f"\n{'#' * 72}\n#  TUNAGEM DO PROBLEMA P{prob}\n{'#' * 72}")
            pid = prob.replace(".", "_")
            cands = sorted(glob.glob(f"_cache_P{pid}_NR*_v14.pkl"))
            if not cands:
                print(f"!! Nenhum cache encontrado para P{prob}.")
                continue
            cache = cands[-1]
            print(f"  Usando cache: {cache}")
            best, ks_max, _ = tune_da_random(cache, n_iters=15, n_seeds=3,
                                             prob=prob)
            results[prob] = (best, ks_max)
        if len(probs) > 1:
            print(f"\n{'=' * 72}\nRESUMO DAS TUNAGENS POR PROBLEMA\n{'=' * 72}")
            for prob, (best, ks_max) in results.items():
                print(f"  P{prob}: objetivo = {ks_max:.4f}  "
                      f"alpha={best['alpha']:.4f} K_max={best['K_max']} "
                      f"n_inner={best['n_inner']}")
        return

    # NOVO: cada problema tem seu proprio campo geostatistico gerado com
    # semente completamente aleatoria (secrets.randbits). Isso quebra a
    # invariancia do K_eff-Durlofsky entre problemas: agora P1.1..P1.5
    # veem campos k(x) DIFERENTES e produzem CDFs distintas por problema.
    import secrets
    alvos = PROBLEMAS if which == "all" else {which: PROBLEMAS[which]}
    DIRS = {"xx": r"$K_{xx}$", "yy": r"$K_{yy}$", "xy": r"$K_{xy}$"}
    for i_prob, (name, bc) in enumerate(alvos.items()):
        # Semente aleatoria por problema (nunca repete entre execucoes)
        seed_geo = secrets.randbits(31)
        tp = time.time()
        K_micro = sample_permeability(N_R, ell_x=3.0, ell_y=3.0,
                                      sigma_lnk=0.9, k_med=1.0,
                                      seed=seed_geo)
        print(f"P{name}: N_R={N_R}, n_R={n_R}  seed_geo={seed_geo} "
              f"(aleatoria)  (campo gerado em {time.time() - tp:.1f}s)")
        # Salva plot da geostatistica para o PRIMEIRO problema apenas
        if i_prob == 0 and which in ("all", "geo"):
            plot_geostat(K_micro, "tese_fig0_geostatistics.png")
            if which == "geo":
                return
        tp = time.time()
        res = run_problem(name, bc, K_micro, n_R=n_R)
        print(f"Problem {name}: SRSS(mean) = {res['rms']:.4f}, "
              f"n_R per macroel. in [{min(res['n_macro'])}, "
              f"{max(res['n_macro'])}]  ({time.time() - tp:.1f}s)")
        t = name.replace(".", "_")
        if mode in ("all", "main"):
            plot_campos(res, f"tese_P{t}_panel1_fields.png")
            for dkey, label_K in DIRS.items():
                lbl = dkey.replace("xx","Kxx").replace("yy","Kyy").replace("xy","Kxy")
                plot_da(res, f"tese_P{t}_panel3_DA_{lbl}.png", dkey=dkey)
            plot_lml(res, f"tese_P{t}_panel4_LML.png")
        for dkey, label_K in DIRS.items():
            if mode in ("all", f"{dkey}_a"):
                plot_codebook_25(res[f"da_all_{dkey}"], label_K,
                                 f"tese_P{t}_panel6_{dkey}_codebook.png", name)
                plot_phase_25(res[f"da_all_{dkey}"], label_K,
                              f"tese_P{t}_panel7_{dkey}_phase.png", name)
            if mode in ("all", f"{dkey}_b"):
                plot_cdf_25(res[f"da_all_{dkey}"], label_K,
                            f"tese_P{t}_panel8_{dkey}_CDF.png", name)
                plot_subsampling_25(res[f"da_all_{dkey}"], label_K,
                                    f"tese_P{t}_panel9_{dkey}_subsampling.png",
                                    name)
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import sys
    a = sys.argv
    which = a[1] if len(a) > 1 else "all"
    mode  = a[2] if len(a) > 2 else "all"
    N_R = int(a[3]) if len(a) > 3 else DEFAULT_N_R
    if len(a) > 4:
        n_R = a[4] if a[4].lower() == "auto" else int(a[4])
    else:
        n_R = DEFAULT_n_R
    main(which, mode, N_R=N_R, n_R=n_R)
