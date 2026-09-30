"""Measure the temperature of the first observed split in the annealing (not computed from 2Var).
Replicates te.deterministic_annealing with identical arguments/seeds and records the atom spread."""
import pickle, json, sys, numpy as np
from joblib import Parallel, delayed
import tese_eliptico as te

def da_trace(samples, alpha, K_max, n_inner, perturb, merge_atol, f_lo, f_hi, seed):
    rng = np.random.default_rng(seed)
    sn = np.asarray(samples, float); M = sn.size; sd = sn.std() + 1e-12
    T_crit = 2.0*float(np.var(sn, ddof=1))
    T_max = te.DA_T_MAX_RATIO*T_crit; T_min = te.DA_T_MIN_RATIO*T_crit
    N_steps = max(int(np.ceil(np.log(T_min/T_max)/np.log(alpha))), 1)
    y = np.array([sn.mean()]); q = np.array([1.0]); T = T_max; step = 0; rec = []
    while T > T_min:
        prog = min(max(step/N_steps, 0.0), 1.0); frac = f_lo + (f_hi-f_lo)*prog
        m = max(int(round(frac*M)), min(40, M))
        bt = sn[rng.choice(M, m, replace=False)] if m < M else sn
        for _ in range(n_inner):
            d = (bt[:, None]-y[None, :])**2
            lw = np.log(q+1e-300)[None, :]-d/T; lw -= lw.max(axis=1, keepdims=True)
            P = np.exp(lw); P /= P.sum(axis=1, keepdims=True)
            qn = P.mean(axis=0); yn = (P*bt[:, None]).sum(axis=0)/(qn*m+1e-300)
            if np.max(np.abs(yn-y))+np.max(np.abs(qn-q)) < 1e-7:
                y, q = yn, qn; break
            y, q = yn, qn
        keep = q > 1e-6; y, q = y[keep], q[keep]
        if y.size == 0: y = np.array([sn.mean()]); q = np.array([1.0])
        o = np.argsort(y); y, q = y[o], q[o]
        ky, kq = [y[0]], [q[0]]
        for i in range(1, len(y)):
            if y[i]-ky[-1] < merge_atol: kq[-1] += q[i]
            else: ky.append(y[i]); kq.append(q[i])
        y = np.asarray(ky); q = np.asarray(kq); q /= q.sum()
        rec.append((T, len(y), float((y.max()-y.min())/sd)))
        T *= alpha; step += 1
        if 2*len(y) <= K_max:
            y = np.concatenate([y, y+perturb*sd*rng.standard_normal(len(y))]); q = np.concatenate([q, q])/2.0
    rec = np.array(rec)
    # first temperature at which the representative values separate by more than 5% of the sample sd
    idx = np.where(rec[:, 2] > 0.05)[0]
    T_split = float(rec[idx[0], 0]) if idx.size else np.nan
    return T_crit, T_split, rec

if __name__ == "__main__":
    P = te._get_da_params()[0]
    out = {}
    for name in sys.argv[1:]:
        res = pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb"))
        jobs = []
        for d in ("xx", "yy", "xy"):
            for e in res[f"da_all_{d}"]:
                I, J = e["IJ"]
                seed = (I*9+J)*3+{"xx": 0, "yy": 1, "xy": 2}[d]
                jobs.append((d, (I, J), e["samples"], seed))
        r = Parallel(n_jobs=2, prefer="threads")(delayed(lambda d, ij, s, sd: (d, ij, *da_trace(s, P["alpha"], P["K_max"], P["n_inner"], P["perturb"], P["merge_atol"], P["f_lo"], P["f_hi"], sd)[:2]))(*j) for j in jobs)
        o = {}
        for d in ("xx", "yy", "xy"):
            Tc = np.array([x[2] for x in r if x[0] == d]); Ts = np.array([x[3] for x in r if x[0] == d])
            o[d] = dict(Tc_theory_mean=float(Tc.mean()), Tsplit_measured_mean=float(np.nanmean(Ts)),
                        ratio_measured_over_theory_mean=float(np.nanmean(Ts/Tc)), ratio_min=float(np.nanmin(Ts/Tc)), ratio_max=float(np.nanmax(Ts/Tc)))
        o["ratio_diag_offdiag_measured"] = float(0.5*(o["xx"]["Tsplit_measured_mean"]+o["yy"]["Tsplit_measured_mean"])/o["xy"]["Tsplit_measured_mean"])
        out[name] = o
        print(name, json.dumps(o, indent=1), flush=True)
    json.dump(out, open("master_tc.json", "w"), indent=1)
