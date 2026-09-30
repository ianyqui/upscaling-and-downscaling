"""Measured cost-accuracy study of the stochastic upscaling pipeline on a 120x120 mesh (coarsening factor 12).
Single-threaded timings. Accuracy is measured against a held-out fine-scale Monte Carlo reference for each scenario,
and every coarse variant is compared with plain Monte Carlo using n fine solves."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import time, json, sys, numpy as np
from scipy.stats import norm
import tese_eliptico as te
te.N_JOBS = 1
P = te._get_da_params()[0]

def set_mesh(NF, NC):
    te.NF, te.NC = NF, NC; te.ALPHA = NF//NC; te.HF = te.LX/NF; te.HC = te.LX/NC

def now(): return time.perf_counter()
BC_T = {"1.1": te.PROBLEMAS["1.1"], "1.3": te.PROBLEMAS["1.3"], "1.4": te.PROBLEMAS["1.4"], "1.2": te.PROBLEMAS["1.2"]}

# ------------------------------------------------------------------ solver scaling
out = {"threads": 1}
if "scaling" in sys.argv:
    sc = []
    bc = te.PROBLEMAS["1.1"]
    for n in (15, 30, 60, 120, 240, 480):
        k = np.exp(0.9*np.random.default_rng(0).standard_normal((n, n)))
        tf = []
        for r in range(3):
            t = now(); te.tpfa(k, k, bc, 15.0/n); tf.append(now()-t)
        nc = n//3; kc = k.reshape(nc, 3, nc, 3).mean(axis=(1, 3)); tc = []
        for r in range(3):
            t = now(); te.tpfa(kc, kc, bc, 15.0/nc); tc.append(now()-t)
        sc.append(dict(NF=n, cells=n*n, t_fine=float(np.median(tf)), t_coarse_a3=float(np.median(tc))))
        print(sc[-1], flush=True)
    out["scaling"] = sc
    x = np.log([s["cells"] for s in sc[-3:]]); y = np.log([s["t_fine"] for s in sc[-3:]])
    out["scaling_exponent_last3"] = float(np.polyfit(x, y, 1)[0])
    n = 480; k = np.exp(0.9*np.random.default_rng(0).standard_normal((n, n)))
    tf = []
    for r in range(3):
        t = now(); te.tpfa(k, k, bc, 15.0/n); tf.append(now()-t)
    tfm = float(np.median(tf)); ra = []
    for a in (2, 3, 4, 6, 8, 12, 16, 20, 24, 40):
        nc = n//a; kc = k.reshape(nc, a, nc, a).mean(axis=(1, 3)); tc = []
        for r in range(5):
            t = now(); te.tpfa(kc, kc, bc, 15.0/nc); tc.append(now()-t)
        ra.append(dict(alpha=a, t_coarse=float(np.median(tc)), ratio=tfm/float(np.median(tc))))
    out["ratio_vs_alpha_NF480"] = ra; print(ra, flush=True)
    json.dump(out, open("master_cost_scaling.json", "w"), indent=1)
    sys.exit()

# ------------------------------------------------------------------ main study
NF, NC = 120, 10; set_mesh(NF, NC); A = NF//NC
N = 500; cal = np.arange(250); tst = np.arange(250, 500)
T = {}
def spectral_fields(N, n, ell, sigma, seed):
    """Stationary Gaussian fields by circulant embedding (2x periodic extension), one global variance normalization.
    Unlike te.sample_permeability for NF > 30, realizations are NOT standardized one by one, which would force every
    field to have zero spatial mean and unit spatial variance and would remove the ensemble variability of the mean."""
    rng = np.random.default_rng(seed); M = 2*n; x = np.arange(M); d = np.minimum(x, M-x)
    DX, DY = np.meshgrid(d, d, indexing="ij"); S = np.clip(np.fft.fft2(np.exp(-(DX/ell)**2-(DY/ell)**2)).real, 0, None)
    sq = np.sqrt(S); Z = np.empty((N, n, n))
    for i in range(N):
        w = rng.standard_normal((M, M))+1j*rng.standard_normal((M, M))
        Z[i] = np.fft.ifft2(sq*w).real[:n, :n]*M/np.sqrt(M*M)
    Z /= Z.std()
    return np.exp(sigma*Z)
t = now(); Kf = spectral_fields(N, NF, 3.0, 0.9, 2026120); T["field_generation_total"] = now()-t
print("field stats: global sd of ln k", float(np.log(Kf).std()), " sd of realization means of ln k", float(np.log(Kf).mean(axis=(1,2)).std()), flush=True)

# offline 1: upscaling of the calibration realizations (2 fine solves each)
t = now()
Kxx = np.empty((250, NC, NC)); Kyy = np.empty_like(Kxx); Kxy = np.empty_like(Kxx)
for i, r in enumerate(cal):
    Kxx[i], Kyy[i], Kxy[i] = te.upscale_realisation(Kf[r], te.HF)
T["upscaling"] = now()-t; print("upscaling", T["upscaling"], flush=True)

# offline 2: DA compression of K_xx and K_yy in every coarse cell (the components the coarse solver uses)
t = now(); cb = {}
for I in range(NC):
    for J in range(NC):
        for d, arr in (("xx", Kxx), ("yy", Kyy)):
            yk, qk, _, _ = te.deterministic_annealing(arr[:, I, J], alpha=P["alpha"], K_max=P["K_max"], n_inner=P["n_inner"],
                                                      perturb=P["perturb"], merge_atol=P["merge_atol"], f_lo=P["f_lo"], f_hi=P["f_hi"],
                                                      seed=(I*NC+J)*3+{"xx": 0, "yy": 1}[d])
            o = np.argsort(yk); cb[(I, J, d)] = (yk[o], qk[o]/qk.sum())
T["DA_compression"] = now()-t; print("DA", T["DA_compression"], flush=True)
Kmean_card = float(np.mean([len(v[0]) for v in cb.values()]))

# offline 3: Gaussian copula on normal scores of the calibration tensors (dependence between coarse cells)
t = now()
X = np.concatenate([Kxx.reshape(250, -1), Kyy.reshape(250, -1)], axis=1)
U = (np.argsort(np.argsort(X, axis=0), axis=0)+0.5)/250.0
Z = norm.ppf(U); C = np.corrcoef(Z, rowvar=False); C = C+1e-8*np.eye(C.shape[0])
Lc = np.linalg.cholesky(C)
T["copula_fit"] = now()-t

def inv_cdf(y, q, u):
    c = np.cumsum(q); idx = np.searchsorted(c, u, side="left"); return y[np.clip(idx, 0, len(y)-1)]

YS = [cb[(I, J, d)][0] for d in ('xx', 'yy') for I in range(NC) for J in range(NC)]
CS = [np.cumsum(cb[(I, J, d)][1]) for d in ('xx', 'yy') for I in range(NC) for J in range(NC)]
def sample_coarse(nR, mode, rng):
    out = []
    for r in range(nR):
        if mode == "independent":
            u = rng.random(2*NC*NC)
        else:
            u = norm.cdf(Lc @ rng.standard_normal(2*NC*NC))
        v = np.array([YS[m][np.minimum(np.searchsorted(CS[m], u[m]), len(YS[m])-1)] for m in range(2*NC*NC)])
        out.append((v[:NC*NC].reshape(NC, NC), v[NC*NC:].reshape(NC, NC)))
    return out

def inflow(p, kx, ky, bc):
    Q = 0.0
    for face, (kind, val) in bc.items():
        if kind == "D" and val == 1.0:
            if face == "esq": Q += np.sum(2*kx[:, 0]*(1-p[:, 0]))
            elif face == "dir": Q += np.sum(2*kx[:, -1]*(1-p[:, -1]))
            elif face == "baixo": Q += np.sum(2*ky[0, :]*(1-p[0, :]))
            else: Q += np.sum(2*ky[-1, :]*(1-p[-1, :]))
    return Q

VL = {"1.1": True, "1.3": True, "1.4": True, "1.2": False}
def midflux(p, kx, ky, vl):
    h = p.shape[0]//2
    if vl:
        T = 2*kx[:, h-1]*kx[:, h]/(kx[:, h-1]+kx[:, h]); return float(abs(np.sum(T*(p[:, h-1]-p[:, h]))))
    T = 2*ky[h-1, :]*ky[h, :]/(ky[h-1, :]+ky[h, :]); return float(abs(np.sum(T*(p[h-1, :]-p[h, :]))))
def cavg(p): return p.reshape(NC, A, NC, A).mean(axis=(1, 3))
def rel(a, b): return float(np.linalg.norm((a-b).ravel())/np.linalg.norm(b.ravel()))
def ks(a, b):
    a = np.sort(a); b = np.sort(b); g = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, g, "right")/len(a)-np.searchsorted(b, g, "right")/len(b))))
def metrics(Pc, Q, ref):
    return dict(err_mean_p=rel(Pc.mean(0), ref["Pm"]), err_sd_p=rel(Pc.std(0), ref["Ps"]),
                err_mean_Q=float(abs(np.mean(Q)-ref["Qm"])/ref["Qm"]), err_sd_Q=float(abs(np.std(Q)-ref["Qs"])/ref["Qs"]),
                ks_Q=ks(Q, ref["Q"]))

res = dict(T_offline=T, mean_cardinality=Kmean_card, scenarios={})
for sname, bc in BC_T.items():
    S = {}
    # fine reference on the test half: brute-force Monte Carlo (the cost comparator)
    t = now(); Pt = []; Qt = []; Qin_t = []
    for r in tst:
        p = te.tpfa(Kf[r], Kf[r], bc, te.HF); Pt.append(cavg(p)); Qt.append(midflux(p, Kf[r], Kf[r], VL[sname])); Qin_t.append(inflow(p, Kf[r], Kf[r], bc))
    S["t_bruteforce_250"] = now()-t
    Pt = np.array(Pt); Qt = np.array(Qt)
    ref = dict(Pm=Pt.mean(0), Ps=Pt.std(0), Qm=float(Qt.mean()), Qs=float(Qt.std()), Q=Qt)
    # fine Monte Carlo pool on the calibration half, for the MC-with-n curves
    Pc_pool = []; Qc_pool = []
    for r in cal:
        p = te.tpfa(Kf[r], Kf[r], bc, te.HF); Pc_pool.append(cavg(p)); Qc_pool.append(midflux(p, Kf[r], Kf[r], VL[sname]))
    Pc_pool = np.array(Pc_pool); Qc_pool = np.array(Qc_pool)
    g = np.random.default_rng(5); mc = {}
    for n in (5, 10, 25, 50, 100, 250):
        ms = []
        for rep in range(40 if n < 250 else 1):
            sub = g.choice(250, n, replace=False) if n < 250 else np.arange(250)
            ms.append(metrics(Pc_pool[sub], Qc_pool[sub], ref))
        mc[n] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    S["mc_with_n"] = mc; S["ref_Q_mean"] = ref["Qm"]; S["ref_Q_sd"] = ref["Qs"]; S["ref_inflow_mean"] = float(np.mean(Qin_t))
    # coarse variants, 250 coarse realizations each
    variants = {}
    t = now(); Pc = []; Q = []; Qin = []
    for i in range(250):
        p = te.tpfa(Kxx[i], Kyy[i], bc, te.HC); Pc.append(p); Q.append(midflux(p, Kxx[i], Kyy[i], VL[sname])); Qin.append(inflow(p, Kxx[i], Kyy[i], bc))
    tt = now()-t; variants["upscaled_no_compression"] = dict(t_online=tt, err_mean_inflow=float(abs(np.mean(Qin)-np.mean(Qin_t))/np.mean(Qin_t)), **metrics(np.array(Pc), np.array(Q), ref))
    for mode in ("independent", "copula"):
        rng = np.random.default_rng(11); t = now(); samp = sample_coarse(250, mode, rng); ts_ = now()-t
        t = now(); Pc = []; Q = []
        for kx, ky in samp:
            p = te.tpfa(kx, ky, bc, te.HC); Pc.append(p); Q.append(midflux(p, kx, ky, VL[sname]))
        tt = now()-t
        variants[f"DA_{mode}"] = dict(t_online=tt+ts_, t_sampling=ts_, **metrics(np.array(Pc), np.array(Q), ref))
    # coarse-only downscaling (projection) cost at this mesh, per 250 realizations
    xi_c, eta_c = te._centers_norm(NC); xi_f, eta_f = te._centers_norm(NF)
    Proj = te._poly_basis(xi_f, eta_f, 4) @ np.linalg.pinv(te._poly_basis(xi_c, eta_c, 4))   # (NF*NF, NC*NC), built once offline
    t = now(); _ = np.array(Pc).reshape(len(Pc), -1) @ Proj.T
    variants["DA_copula"]["t_projection_downscaling"] = now()-t
    S["variants"] = variants
    res["scenarios"][sname] = S
    print(sname, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in variants.items()}), "BF", round(S["t_bruteforce_250"], 2), flush=True)
    json.dump(res, open("master_cost.json", "w"), indent=1, default=float)
