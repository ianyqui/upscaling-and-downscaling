"""Controlled sweep of the cardinality bound on the worst-variance cell (MOS article)."""
import pickle, json, sys, numpy as np
from joblib import Parallel, delayed
import tese_eliptico as te

P = te._get_da_params()[0]
KMAXES = [3, 4, 5, 6, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 46, 52, 58]

def mdl_terms(x, yk, q):
    N = x.size; h = 1.06*x.std(ddof=1)*N**(-0.2)
    d = (x[:, None]-yk[None, :])/h
    logk = -0.5*d**2 - np.log(h*np.sqrt(2*np.pi)) + np.log(q[None, :]+1e-300)
    mx = logk.max(axis=1, keepdims=True)
    ll = float((mx[:, 0] + np.log(np.exp(logk-mx).sum(axis=1))).sum())
    return -ll, 0.5*len(yk)*np.log(N)

def one(x, kmax, seed, xtest=None):
    yk, qk, h, Tc = te.deterministic_annealing(x, alpha=P["alpha"], K_max=kmax, n_inner=P["n_inner"],
                                               perturb=P["perturb"], merge_atol=P["merge_atol"],
                                               f_lo=P["f_lo"], f_hi=P["f_hi"], seed=seed)
    q = qk/qk.sum(); mis, pen = mdl_terms(x, yk, q)
    r = dict(K_max=kmax, K=int(len(yk)), misfit=mis, penalty=pen, MDL=mis+pen,
             KS=te._ks_exact(x, yk, q), Ks=int((q*x.size >= 5).sum()),
             H=float(-(q[q>0]*np.log(q[q>0])).sum()))
    if xtest is not None:
        r["KS_test"] = te._ks_exact(xtest, yk, q)
    return r

if __name__ == "__main__":
    out = {}
    for name in sys.argv[1:]:
        res = pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb"))
        arr = res["Kxx_all"]; var = arr.var(axis=0); I, J = np.unravel_index(int(np.argmax(var)), var.shape)
        x = arr[:, I, J]
        seed = (I*9+J)*3
        full = Parallel(n_jobs=2, prefer="threads")(delayed(one)(x, k, seed) for k in KMAXES)
        rng = np.random.default_rng(12345); perm = rng.permutation(x.size); tr, ts = x[perm[:500]], x[perm[500:]]
        half = Parallel(n_jobs=2, prefer="threads")(delayed(one)(tr, k, seed, ts) for k in KMAXES)
        out[name] = dict(cell=[int(I), int(J)], var=float(var[I, J]), full=full, split500=half)
        mdl = min(full, key=lambda r: r["MDL"]); ksm = min(r["KS"] for r in full)
        plateau = next(r for r in full if r["KS"] <= 1.05*ksm)
        kst = min(r["KS_test"] for r in half); plat_t = next(r for r in half if r["KS_test"] <= 1.05*kst)
        out[name]["summary"] = dict(K_MDL=mdl["K"], KS_at_MDL=mdl["KS"], K_KSplateau=plateau["K"], KS_plateau=plateau["KS"],
                                    KS_min=ksm, heldout_K_plateau=plat_t["K"], heldout_KS_min=kst)
        print(name, json.dumps(out[name]["summary"]), flush=True)
        for r in full: print("   ", r, flush=True)
    json.dump(out, open("master_mos.json", "w"), indent=1)
