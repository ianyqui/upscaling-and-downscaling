"""Fine (15x15) versus coarse (5x5) pressure ensembles for the three problems: absolute and relative errors.
Reference: the 1000 fine-scale solutions of each problem (cached), averaged over each 3x3 block of the coarse cell.
Coarse variants (1000 coarse solves each unless stated):
  UPS   upscaled tensors of each realization, no compression (isolates the upscaling error)
  DA25  production run: 25 coarse realizations sampled independently from the representative values (cached)
  DAI   1000 coarse realizations sampled independently from the representative values
  DAC   1000 coarse realizations sampled from the representative values with a Gaussian copula (Kxx, Kyy of all cells)
Noise floor: two independent halves (500/500) of the fine ensemble compared with each other.
Also the projection of each coarse realization to the fine grid (degree-4 polynomial, no fine-scale statistics)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pickle, numpy as np
from scipy.stats import norm
import tese_eliptico as te
NC, A = 5, 3
def blk(P): return P.reshape(P.shape[0], NC, A, NC, A).mean(axis=(2, 4))
def metrics(ref, est):
    e = est-ref; ae = np.abs(e); re_ = ae/np.maximum(np.abs(ref), 1e-12)
    return dict(max_abs=float(ae.max()), mean_abs=float(ae.mean()), max_rel=float(100*re_.max()), mean_rel=float(100*re_.mean()),
                rel_L2=float(100*np.linalg.norm(e)/np.linalg.norm(ref)), R2=float(1-np.sum(e**2)/np.sum((ref-ref.mean())**2)))
def ks(a, b):
    a = np.sort(a); b = np.sort(b); g = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, g, "right")/len(a)-np.searchsorted(b, g, "right")/len(b))))
def inv(y, q, u):
    o = np.argsort(y); y = y[o]; c = np.cumsum(q[o]/q.sum()); return y[np.minimum(np.searchsorted(c, u), len(y)-1)]
out = {}; fields = {}
for name in ("1.1", "1.3", "1.4"):
    res = pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb")); bc = te.PROBLEMAS[name]
    Pf = res["P_micro_all"]; Pfc = blk(Pf); N = Pf.shape[0]
    cbx = {tuple(e["IJ"]): (e["yk"], e["qk"]) for e in res["da_all_xx"]}; cby = {tuple(e["IJ"]): (e["yk"], e["qk"]) for e in res["da_all_yy"]}
    Kxx, Kyy = res["Kxx_all"], res["Kyy_all"]
    V = {}
    V["UPS"] = np.array([te.tpfa(Kxx[i], Kyy[i], bc, te.HC) for i in range(N)])
    V["DA25"] = res["P_macro_all"]
    rng = np.random.default_rng(20260924)
    def sample(U):
        Kx = np.empty((NC, NC)); Ky = np.empty((NC, NC))
        for I in range(NC):
            for J in range(NC):
                Kx[I, J] = inv(*cbx[(I, J)], U[0, I, J]); Ky[I, J] = inv(*cby[(I, J)], U[1, I, J])
        return te.tpfa(Kx, Ky, bc, te.HC)
    V["DAI"] = np.array([sample(rng.random((2, NC, NC))) for _ in range(N)])
    Z = np.concatenate([Kxx.reshape(N, -1), Kyy.reshape(N, -1)], axis=1)
    R = (np.argsort(np.argsort(Z, axis=0), axis=0)+0.5)/N; C = np.corrcoef(norm.ppf(R), rowvar=False)
    L = np.linalg.cholesky(C + 1e-10*np.eye(C.shape[0]))
    V["DAC"] = np.array([sample(norm.cdf(L @ rng.standard_normal(2*NC*NC)).reshape(2, NC, NC)) for _ in range(N)])
    o = {"ref_mean_range": [float(Pfc.mean(0).min()), float(Pfc.mean(0).max())], "ref_sd_range": [float(Pfc.std(0).min()), float(Pfc.std(0).max())]}
    perm = np.random.default_rng(777).permutation(N); a, b = perm[:N//2], perm[N//2:]
    o["noise_floor"] = dict(mean=metrics(Pfc[a].mean(0), Pfc[b].mean(0)), sd=metrics(Pfc[a].std(0), Pfc[b].std(0)),
                            ks_mean=float(np.mean([ks(Pfc[a][:, I, J], Pfc[b][:, I, J]) for I in range(NC) for J in range(NC)])))
    for k, P in V.items():
        o[k] = dict(n=int(P.shape[0]), mean=metrics(Pfc.mean(0), P.mean(0)), sd=metrics(Pfc.std(0), P.std(0)),
                    sd_ratio_mean=float((P.std(0)/Pfc.std(0)).mean()),
                    ks_mean=float(np.mean([ks(Pfc[:, I, J], P[:, I, J]) for I in range(NC) for J in range(NC)])),
                    ks_max=float(np.max([ks(Pfc[:, I, J], P[:, I, J]) for I in range(NC) for J in range(NC)])))
        Pp = np.array([te.lml_downscale(p)[0] for p in P])
        o[k]["fine_mean"] = metrics(Pf.mean(0), Pp.mean(0)); o[k]["fine_sd"] = metrics(Pf.std(0), Pp.std(0))
        fields[(name, k)] = dict(mean=P.mean(0), sd=P.std(0), fmean=Pp.mean(0), fsd=Pp.std(0))
    fields[(name, "REF")] = dict(mean=Pfc.mean(0), sd=Pfc.std(0), fmean=Pf.mean(0), fsd=Pf.std(0))
    out[name] = o
    print(name, {k: (round(o[k]["mean"]["rel_L2"], 2), round(o[k]["sd"]["rel_L2"], 1), round(o[k]["ks_mean"], 3)) for k in V}, "floor", round(o["noise_floor"]["mean"]["rel_L2"], 2), round(o["noise_floor"]["sd"]["rel_L2"], 1), flush=True)
json.dump(out, open("main_flow.json", "w"), indent=1)
pickle.dump({f"{k[0]}|{k[1]}": {kk: vv.tolist() for kk, vv in v.items()} for k, v in fields.items()}, open("main_flow_fields.pkl", "wb"))
