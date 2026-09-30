"""Finite-sample check of the Shannon lower bound on a heavy-tailed source of known entropy (log-normal, sigma = 0.75: skewness 3.3, excess kurtosis 23)."""
import json, numpy as np, tese_eliptico as te
from main_rd import kl_entropy
s_ln = 0.75; h_true = 0.5*np.log(2*np.pi*np.e*s_ln**2)  # mu = 0
out = {}
for N in (1000, 10000):
    exc = []; exc_kl = []
    for seed in range(3 if N == 1000 else 1):
        x = np.exp(s_ln*np.random.default_rng(100+seed).standard_normal(N))
        y, q, H, Tc = te.deterministic_annealing(x, alpha=0.953, K_max=50, n_inner=40, perturb=4.6e-3, merge_atol=3.8e-4, f_lo=0.76, f_hi=1.0, seed=seed)
        T, D, I, K, m = H.T
        exc.append(float(np.max(h_true-0.5*np.log(2*np.pi*np.e*D)-I))); exc_kl.append(float(np.max(kl_entropy(x)-0.5*np.log(2*np.pi*np.e*D)-I)))
    out[N] = dict(h_true=h_true, max_exc_true_h=exc, max_exc_kl_h=exc_kl); print(N, out[N], flush=True)
json.dump(out, open("rd_lognormal_check.json", "w"), indent=1)
