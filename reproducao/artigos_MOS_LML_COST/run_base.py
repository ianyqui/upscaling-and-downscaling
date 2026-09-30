import sys, time, json, numpy as np
import tese_eliptico as te
SEEDS = {"1.1": 2026101, "1.3": 2026103, "1.4": 2026104}
for name in sys.argv[1:]:
    t=time.time()
    K = te.sample_permeability(1000, ell_x=3.0, ell_y=3.0, sigma_lnk=0.9, k_med=1.0, seed=SEEDS[name])
    np.save(f"Kmicro_P{name.replace('.','_')}.npy", K)
    res = te.run_problem(name, te.PROBLEMAS[name], K, n_R=25, seed_macro=7)
    print(name, "done in", round(time.time()-t,1), "s; SRSS", res["rms"], flush=True)
