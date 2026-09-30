"""Complexity at the operating point, source statistics and quartile diagnostics from the cached production runs."""
import json, pickle, numpy as np
from scipy.special import gammaln
from scipy.stats import skew, kurtosis
import tese_eliptico as te
out = {}
for name in ("1.1", "1.3", "1.4"):
    res = pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb")); N = res["N_R"]; o = {}
    for d in ("xx", "yy", "xy"):
        rows = []
        for e in res[f"da_all_{d}"]:
            s = e["samples"]; y, q = e["yk"], e["qk"]; K = len(y)
            n = np.rint(q*N).astype(int); p = q[q > 0]; H = float(-(p*np.log(p)).sum())
            bits_masses = float((gammaln(N) - gammaln(K) - gammaln(N-K+1))/np.log(2))  # log2 C(N-1, K-1)
            bits = 64*K + bits_masses
            qd = te.ks_quartis(s, y, q)
            rows.append(dict(IJ=list(e["IJ"]), K=K, Ks=int((n >= 5).sum()), H=H, PP=float(np.exp(H)), bits=bits,
                             ks=te._ks_exact(s, y, q), qL=[qd[f"Q{i}"]["Linf"] for i in range(1, 5)], var=float(s.var())))
        A = lambda k: np.array([r[k] for r in rows], float)
        allS = res[f"K{d}_all"].ravel()
        iw = int(np.argmax(A("ks"))); iv = int(np.argmax(A("var")))
        o[d] = dict(K_mean=A("K").mean(), K_min=int(A("K").min()), K_max=int(A("K").max()), Ks_mean=A("Ks").mean(),
                    below_floor=float(1-A("Ks").sum()/A("K").sum()), H_mean=A("H").mean(), PP_mean=A("PP").mean(),
                    PP_over_K=float((A("PP")/A("K")).mean()), bits_mean=A("bits").mean(), raw_bits=64.0*N,
                    ratio_objects=float(N/A("K").mean()), ratio_bits=float(64.0*N/A("bits").mean()),
                    ks_mean=A("ks").mean(), ks_max=A("ks").max(), worst_ks_cell=rows[iw]["IJ"], worst_ks_quart=rows[iw]["qL"],
                    worst_var_cell=rows[iv]["IJ"], worst_var_quart=rows[iv]["qL"],
                    argmax_quartile_hist=np.bincount([int(np.argmax(r["qL"])) for r in rows], minlength=4).tolist(),
                    src_mean=float(allS.mean()), src_sd=float(allS.std()), frac_neg=float((allS < 0).mean()),
                    skew=float(np.mean([skew(res[f"K{d}_all"][:, I, J]) for I in range(5) for J in range(5)])),
                    exkurt=float(np.mean([kurtosis(res[f"K{d}_all"][:, I, J]) for I in range(5) for J in range(5)])),
                    corr_K_logvar=float(np.corrcoef(A("K"), np.log(A("var")))[0, 1]), cells=rows)
        print(name, d, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o[d].items() if k != "cells"})
    # cross-component correlation within cells (independent quantization discards it)
    C = []
    for I in range(5):
        for J in range(5):
            X = np.stack([res["Kxx_all"][:, I, J], res["Kyy_all"][:, I, J], res["Kxy_all"][:, I, J]])
            c = np.corrcoef(X); C.append([c[0, 1], c[0, 2], c[1, 2]])
    C = np.array(C); o["corr_xx_yy"] = float(C[:, 0].mean()); o["corr_xx_xy"] = float(C[:, 1].mean()); o["corr_yy_xy"] = float(C[:, 2].mean())
    # neighbour-cell correlation of Kxx (inter-cell dependence)
    a = res["Kxx_all"]; nb = [np.corrcoef(a[:, I, J], a[:, I, J+1])[0, 1] for I in range(5) for J in range(4)]
    o["corr_neighbour_xx"] = float(np.mean(nb))
    print(name, "corr", o["corr_xx_yy"], o["corr_xx_xy"], o["corr_yy_xy"], "neigh", o["corr_neighbour_xx"])
    out[name] = o
json.dump(out, open("main_summary.json", "w"), indent=1)
