"""Sweep of the cardinality bound with 5 annealing seeds per value (worst-variance K_xx cell), full sample and 500/500 split."""
import pickle, json, sys, numpy as np
from joblib import Parallel, delayed
import tese_eliptico as te
from mos_sweep import one, KMAXES
SEEDS = [0, 1, 2, 3, 4]
out = {}
for name in sys.argv[1:]:
    res = pickle.load(open(f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl", "rb"))
    arr = res["Kxx_all"]; var = arr.var(axis=0); I, J = np.unravel_index(int(np.argmax(var)), var.shape)
    x = arr[:, I, J]
    perm = np.random.default_rng(12345).permutation(x.size); tr, ts = x[perm[:500]], x[perm[500:]]
    jobs = [(k, s) for k in KMAXES for s in SEEDS]
    full = Parallel(n_jobs=2, prefer="threads")(delayed(one)(x, k, 1000+s) for k, s in jobs)
    half = Parallel(n_jobs=2, prefer="threads")(delayed(one)(tr, k, 1000+s, ts) for k, s in jobs)
    out[name] = dict(cell=[int(I), int(J)], var=float(var[I, J]), full=full, split500=half, jobs=jobs)
    print(name, "done", flush=True)
    json.dump(out, open("master_mos_multiseed.json", "w"), indent=1)
