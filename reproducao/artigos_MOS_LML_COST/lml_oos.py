"""Out-of-sample assessment of the stochastic LML downscaling (500/500 split).
Everything fitted (DA codebooks, macro ensemble, bias field, residual scale, rescaling) uses the training half only."""
import pickle, json, sys, numpy as np
from joblib import Parallel, delayed
import tese_eliptico as te

P = te._get_da_params()[0]; NF, NC, A = te.NF, te.NC, te.ALPHA

def rel(a, b): return float(np.linalg.norm((a-b).ravel())/np.linalg.norm(b.ravel()))
def r2(a, b):
    a = a.ravel(); b = b.ravel(); return float(1-np.sum((a-b)**2)/np.sum((b-b.mean())**2))

out = {}
for name in sys.argv[1:]:
    res = pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb"))
    bc = res["bc"]; Pm = res["P_micro_all"]
    perm = np.random.default_rng(777).permutation(Pm.shape[0]); tr, ts = perm[:500], perm[500:]
    Ktr = {d: res[f"K{d}_all"][tr] for d in ("xx", "yy")}
    def one(I, J, d):
        yk, qk, _, _ = te.deterministic_annealing(Ktr[d][:, I, J], alpha=P["alpha"], K_max=P["K_max"], n_inner=P["n_inner"],
                                                  perturb=P["perturb"], merge_atol=P["merge_atol"], f_lo=P["f_lo"], f_hi=P["f_hi"],
                                                  seed=(I*9+J)*3+{"xx": 0, "yy": 1}[d])
        return (I, J, d), (yk, qk)
    cb = dict(Parallel(n_jobs=2, prefer="threads")(delayed(one)(I, J, d) for I in range(NC) for J in range(NC) for d in ("xx", "yy")))
    n_R = int(res["n_R"]); rng = np.random.default_rng(7); seeds = rng.integers(0, 2**31-1, size=n_R)
    Pmac = []
    for s in seeds:
        g = np.random.default_rng(int(s)); Kx = np.empty((NC, NC)); Ky = np.empty((NC, NC))
        for I in range(NC):
            for J in range(NC):
                yx, qx = cb[(I, J, "xx")]; yy, qy = cb[(I, J, "yy")]
                Kx[I, J] = yx[g.choice(len(yx), p=qx/qx.sum())]; Ky[I, J] = yy[g.choice(len(yy), p=qy/qy.sum())]
        Pmac.append(te.tpfa(Kx, Ky, bc, te.HC))
    Pmac = np.array(Pmac)
    mic_tr_m, mic_tr_s = Pm[tr].mean(0), Pm[tr].std(0); mic_ts_m, mic_ts_s = Pm[ts].mean(0), Pm[ts].std(0)
    _, ell_res = te.calibrate_lml_residual(Pm[tr])
    det = np.array([te.lml_downscale(p)[0] for p in Pmac])
    var_lml = det.var(0); sig = np.sqrt(np.maximum(mic_tr_s**2-var_lml, 0.0)); clip = float(np.mean(mic_tr_s**2-var_lml <= 0))
    bias = mic_tr_m-det.mean(0); L = te._build_chol_corr_fine(ell_res)
    NS = max(n_R, te.LML_N_STOC_MIN); g = np.random.default_rng(0xBEEF)
    raw = np.array([te.lml_downscale_stochastic(Pmac[r % n_R], sig, L, g)+bias for r in range(NS)])
    mu_raw, sd_raw = raw.mean(0), raw.std(0)+1e-12
    down = (raw-mu_raw)*(mic_tr_s/sd_raw)+mu_raw
    dm, ds = down.mean(0), down.std(0)
    o = dict(clip_fraction=clip, ell_res=ell_res,
             insample=dict(err_mean=rel(dm, mic_tr_m), err_sd=rel(ds, mic_tr_s), R2_mean=r2(dm, mic_tr_m)),
             outofsample=dict(err_mean=rel(dm, mic_ts_m), err_sd=rel(ds, mic_ts_s), R2_mean=r2(dm, mic_ts_m), R2_sd=r2(ds, mic_ts_s),
                              err_sd_no_rescaling=rel(sd_raw, mic_ts_s), err_mean_no_bias=rel(det.mean(0), mic_ts_m)),
             reference_split_gap=dict(err_mean_train_vs_test=rel(mic_tr_m, mic_ts_m), err_sd_train_vs_test=rel(mic_tr_s, mic_ts_s)),
             coarse_only=dict(err_sd_lml_det=rel(np.sqrt(var_lml), mic_ts_s)))
    out[name] = o; print(name, json.dumps(o, indent=1), flush=True)
json.dump(out, open("master_lml.json", "w"), indent=1)
