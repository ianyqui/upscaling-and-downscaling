"""Production configuration on the same worst-variance cells, two independent triplets of annealing seeds (noise level of J)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pickle, time, numpy as np, tese_eliptico as te
cfg = dict(alpha=0.953, K_max=50, n_inner=40, perturb=4.6e-3, merge_atol=3.8e-4, f_lo=0.76)
out = {}
for name in ("1.4", "1.1", "1.3"):
    res = pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb")); o = {}
    for trip in ((0, 1, 2), (3, 4, 5)):
        per = {}; tt = []
        for d in ("xx", "yy", "xy"):
            arr = res[f"K{d}_all"]; I, J = divmod(int(np.argmax(arr.var(axis=0))), 5); s = arr[:, I, J]; ks = []; qs = []
            for sd in trip:
                t0 = time.perf_counter(); y, q, _, _ = te.deterministic_annealing(s, seed=sd, f_hi=1.0, **cfg); tt.append(time.perf_counter()-t0)
                ks.append(te._ks_exact(s, y, q)); qq = te.ks_quartis(s, y, q); qs.append([qq[f"Q{i}"]["Linf"] for i in range(1, 5)])
            per[d] = dict(ks=float(np.mean(ks)), q=np.mean(qs, axis=0).tolist())
        o[str(trip)] = dict(J=max(p["ks"] for p in per.values()), Jm=float(np.mean([np.mean(p["q"]) for p in per.values()])), t=float(np.mean(tt)))
    out[name] = o; print(name, o, flush=True)
json.dump(out, open("main_search_ref.json", "w"), indent=1)
