"""Experiments for the MAIN article (15x15 -> 5x5, cached ensembles of run_base.py).
  python main_exp.py tc              -> main_tc_cells.json  (per-cell measured first split, K(T) staircases)
  python main_exp.py search 1.4 40   -> main_search_P1_4.json (random search, quartile-KS objective, timing)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, time, pickle, hashlib, numpy as np
import tese_eliptico as te
te.N_JOBS = 1
from measure_tc import da_trace  # identical replica of te.deterministic_annealing with split tracking

def load(name): return pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb"))

def run_tc():
    P = te._get_da_params()[0]; out = {}
    for name in ("1.1", "1.3", "1.4"):
        res = load(name); o = {}
        for d in ("xx", "yy", "xy"):
            cells = []
            for e in res[f"da_all_{d}"]:
                I, J = e["IJ"]; seed = (I*9+J)*3+{"xx": 0, "yy": 1, "xy": 2}[d]
                Tc, Ts, rec = da_trace(e["samples"], P["alpha"], P["K_max"], P["n_inner"], P["perturb"], P["merge_atol"], P["f_lo"], P["f_hi"], seed)
                cells.append(dict(IJ=[int(I), int(J)], var=float(np.var(e["samples"], ddof=1)), Tc_theory=Tc, T_split=Ts,
                                  T=rec[:, 0].tolist(), K=rec[:, 1].astype(int).tolist()))
            o[d] = cells
        out[name] = o; print(name, "done", flush=True)
    json.dump(out, open("main_tc_cells.json", "w"))

def run_search(name, n_iter):
    res = load(name); NC = res["N_C"]; worst = {}
    for d in ("xx", "yy", "xy"):
        arr = res[f"K{d}_all"]; idx = int(np.argmax(arr.var(axis=0))); I, J = divmod(idx, NC)
        worst[d] = dict(IJ=[I, J], s=arr[:, I, J])
    base = int.from_bytes(hashlib.sha256(f"tune-da-{name}".encode()).digest()[:4], "big") % (2**31)
    rng = np.random.default_rng(base); rows = []
    for it in range(n_iter):
        cfg = dict(alpha=float(np.exp(rng.uniform(np.log(0.92), np.log(0.98)))), K_max=50,
                   n_inner=int(rng.choice([40, 75, 100, 150])),
                   perturb=float(np.exp(rng.uniform(np.log(5e-4), np.log(1e-2)))),
                   merge_atol=float(np.exp(rng.uniform(np.log(5e-5), np.log(1e-3)))),
                   f_lo=float(rng.uniform(0.7, 1.0)))
        r = dict(cfg=cfg, per={}); tt = []
        for d, w in worst.items():
            acc = dict(ks=[], q=[], K=[], H=[])
            for seed in range(3):
                t0 = time.perf_counter()
                yk, qk, _, _ = te.deterministic_annealing(w["s"], seed=seed, f_hi=1.0, **cfg)
                tt.append(time.perf_counter()-t0)
                q = te.ks_quartis(w["s"], yk, qk)
                acc["ks"].append(te._ks_exact(w["s"], yk, qk)); acc["q"].append([q[f"Q{i}"]["Linf"] for i in range(1, 5)])
                acc["K"].append(len(yk)); p = qk[qk > 0]; acc["H"].append(float(-(p*np.log(p)).sum()))
            r["per"][d] = dict(ks=float(np.mean(acc["ks"])), q=np.mean(acc["q"], axis=0).tolist(), K=float(np.mean(acc["K"])), H=float(np.mean(acc["H"])))
        r["J"] = float(max(max(r["per"][d]["q"]) for d in r["per"])); r["KSmax"] = float(max(r["per"][d]["ks"] for d in r["per"]))
        r["t_call"] = float(np.mean(tt)); r["K_mean"] = float(np.mean([r["per"][d]["K"] for d in r["per"]]))
        rows.append(r); print(name, it, f"J={r['J']:.4f} KS={r['KSmax']:.4f} t={r['t_call']:.2f}s K={r['K_mean']:.1f}", flush=True)
        json.dump(dict(base_seed=base, worst={d: w["IJ"] for d, w in worst.items()}, rows=rows), open(f"main_search_P{name.replace('.','_')}.json", "w"), indent=1)

if __name__ == "__main__":
    if sys.argv[1] == "tc": run_tc()
    else: run_search(sys.argv[2], int(sys.argv[3]))
