"""Experiments for the methodological LML paper.
For each problem: DA on the training half -> coarse ensemble -> operator components.
(1) in-sample identity, (2) same-scenario hold-out vs split gap, (3) null operator,
(4) calibration-size experiment, (5) what the coarse scale contributes (no fine-scale statistics used)."""
import pickle, json, sys, numpy as np
from joblib import Parallel, delayed
import tese_eliptico as te

P = te._get_da_params()[0]; NC = te.NC
def rel(a, b): return float(np.linalg.norm((a-b).ravel())/np.linalg.norm(b.ravel()))

def coarse_ensemble(res, idx):
    K = {d: res[f"K{d}_all"][idx] for d in ("xx", "yy")}
    def one(I, J, d):
        yk, qk, _, _ = te.deterministic_annealing(K[d][:, I, J], alpha=P["alpha"], K_max=P["K_max"], n_inner=P["n_inner"],
                                                  perturb=P["perturb"], merge_atol=P["merge_atol"], f_lo=P["f_lo"], f_hi=P["f_hi"],
                                                  seed=(I*9+J)*3+{"xx": 0, "yy": 1}[d])
        return (I, J, d), (yk, qk)
    cb = dict(Parallel(n_jobs=2, prefer="threads")(delayed(one)(I, J, d) for I in range(NC) for J in range(NC) for d in ("xx", "yy")))
    n_R = int(res["n_R"]); seeds = np.random.default_rng(7).integers(0, 2**31-1, size=n_R); out = []
    for s in seeds:
        g = np.random.default_rng(int(s)); Kx = np.empty((NC, NC)); Ky = np.empty((NC, NC))
        for I in range(NC):
            for J in range(NC):
                yx, qx = cb[(I, J, "xx")]; yy, qy = cb[(I, J, "yy")]
                Kx[I, J] = yx[g.choice(len(yx), p=qx/qx.sum())]; Ky[I, J] = yy[g.choice(len(yy), p=qy/qy.sum())]
        out.append(te.tpfa(Kx, Ky, res["bc"], te.HC))
    return np.array(out)

def operator(Pmac, cal, rescale=True, bias=True, seed=0xBEEF):
    """LML operator of the pipeline, calibrated on the fine-scale realizations `cal` (n, NF, NF)."""
    m_c, s_c = cal.mean(0), cal.std(0)
    _, ell = te.calibrate_lml_residual(cal)
    det = np.array([te.lml_downscale(p)[0] for p in Pmac])
    sig = np.sqrt(np.maximum(s_c**2-det.var(0), 0.0))
    b = m_c-det.mean(0) if bias else 0.0
    L = te._build_chol_corr_fine(ell); NS = max(len(Pmac), te.LML_N_STOC_MIN); g = np.random.default_rng(seed)
    raw = np.array([te.lml_downscale_stochastic(Pmac[r % len(Pmac)], sig, L, g)+b for r in range(NS)])
    if not rescale: return raw, det, sig
    mu, sd = raw.mean(0), raw.std(0)+1e-12
    return (raw-mu)*(s_c/sd)+mu, det, sig

out = {}; arrays = {}
for name in sys.argv[1:]:
    res = pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb")); Pm = res["P_micro_all"]
    perm = np.random.default_rng(777).permutation(Pm.shape[0]); tr, ts = perm[:500], perm[500:]
    Pmac = coarse_ensemble(res, tr)
    mt, st = Pm[ts].mean(0), Pm[ts].std(0); mtr, strn = Pm[tr].mean(0), Pm[tr].std(0)
    down, det, sig = operator(Pmac, Pm[tr]); raw, _, _ = operator(Pmac, Pm[tr], rescale=False)
    o = {}
    o["insample"] = dict(mean=rel(down.mean(0), mtr), sd=rel(down.std(0), strn))
    o["holdout"] = dict(mean=rel(down.mean(0), mt), sd=rel(down.std(0), st))
    o["split_gap"] = dict(mean=rel(mtr, mt), sd=rel(strn, st))
    o["null_operator"] = dict(mean=rel(mtr, mt), sd=rel(strn, st))   # emits the calibration statistics, no coarse input
    o["raw_no_rescaling"] = dict(mean=rel(raw.mean(0), mt), sd=rel(raw.std(0), st))
    o["coarse_only"] = dict(mean=rel(det.mean(0), mt), sd=rel(det.std(0), st))   # projection of the coarse ensemble, no fine statistics
    o["clip_fraction"] = float(np.mean(sig <= 0))
    o["coarse_mean_sd_ratio"] = float(np.mean(det.std(0))/np.mean(st))
    # calibration-size experiment
    cs = {}
    rng = np.random.default_rng(99)
    for n in (10, 25, 50, 100, 250, 500):
        e_op_sd, e_null_sd, e_op_m, e_null_m = [], [], [], []
        for rep in range(12 if n < 500 else 1):
            sub = tr if n == 500 else rng.choice(tr, n, replace=False)
            d_, _, _ = operator(Pmac, Pm[sub], seed=1000+rep)
            e_op_sd.append(rel(d_.std(0), st)); e_null_sd.append(rel(Pm[sub].std(0), st))
            e_op_m.append(rel(d_.mean(0), mt)); e_null_m.append(rel(Pm[sub].mean(0), mt))
        cs[n] = dict(op_sd=float(np.mean(e_op_sd)), null_sd=float(np.mean(e_null_sd)), op_sd_sd=float(np.std(e_op_sd)),
                     op_mean=float(np.mean(e_op_m)), null_mean=float(np.mean(e_null_m)))
    o["calibration_size"] = cs
    out[name] = o; print(name, json.dumps(o, indent=1), flush=True)
    arrays[name] = dict(m_ref=mt, s_ref=st, m_down=down.mean(0), s_down=down.std(0), s_raw=raw.std(0), s_coarse=det.std(0),
                        m_coarse=det.mean(0), sig=sig, Pmac_mean=Pmac.mean(0), Pmac_std=Pmac.std(0))
json.dump(out, open("master_lml_paper.json", "w"), indent=1)
pickle.dump(arrays, open("lml_paper_arrays.pkl", "wb"))
