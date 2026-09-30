import json, numpy as np
from scipy.stats import spearmanr
ref = json.load(open("main_search_ref.json")) if __import__("os").path.exists("main_search_ref.json") else {}
out = {}
for p in ("1.4", "1.1", "1.3"):
    try: S = json.load(open(f"main_search_P{p.replace('.','_')}.json"))
    except Exception: continue
    R = S["rows"]; n = len(R)
    J = np.array([r["J"] for r in R]); t = np.array([r["t_call"] for r in R]); K = np.array([r["K_mean"] for r in R])
    Jm = np.array([np.mean([np.mean(r["per"][c]["q"]) for c in r["per"]]) for r in R])
    X = {k: np.array([r["cfg"][k] for r in R]) for k in ("alpha", "n_inner", "perturb", "merge_atol", "f_lo")}
    o = np.argsort(t); best = np.inf; par = []
    for i in o:
        if J[i] < best: best = J[i]; par.append(int(i))
    fast, prec = par[0], par[-1]
    # knee: max normalized distance to chord
    tn = (t-t[fast])/(t[prec]-t[fast]+1e-12); jn = (J-J[prec])/(J[fast]-J[prec]+1e-12)
    dist = [abs(tn[i]+jn[i]-1)/np.sqrt(2) for i in par]; knee = par[int(np.argmax(dist))]
    arch = {k: dict(i=i, J=float(J[i]), t=float(t[i]), K=float(K[i]), **R[i]["cfg"]) for k, i in (("fast", fast), ("knee", knee), ("precise", prec))}
    rho = {k: float(spearmanr(v, J)[0]) for k, v in X.items()}; rhot = {k: float(spearmanr(v, t)[0]) for k, v in X.items()}
    o_ = dict(n=n, J_min=float(J.min()), J_max=float(J.max()), J_med=float(np.median(J)), t_min=float(t.min()), t_max=float(t.max()),
              pareto=par, arch=arch, rho_J=rho, rho_t=rhot, rho_J_Jm=float(spearmanr(J, Jm)[0]), argmin_J=int(np.argmin(J)), argmin_Jm=int(np.argmin(Jm)),
              rank_of_Jbest_in_Jm=int(np.argsort(np.argsort(Jm))[np.argmin(J)])+1, K_range=[float(K.min()), float(K.max())],
              argmax_quart=np.bincount([int(np.argmax(r["per"][c]["q"])) for r in R for c in r["per"]], minlength=4).tolist(), ref=ref.get(p))
    out[p] = o_; print(p, json.dumps({k: v for k, v in o_.items()}, indent=0, default=float)[:2500])
json.dump(out, open("main_search_an.json", "w"), indent=1)
