import numpy as np, json, tese_eliptico as te
from main_rd import kl_entropy
out={}
for N in (1000, 10000):
    rng=np.random.default_rng(5); x=rng.standard_normal(N)
    y,q,H,Tc=te.deterministic_annealing(x, alpha=0.953, K_max=50, n_inner=40, perturb=4.6e-3, merge_atol=3.8e-4, f_lo=1.0, f_hi=1.0, seed=1)
    T,D,I,K,m=H.T; v=x.var(); g=0.5*np.log(v/D)
    sel=(D/v<0.5)&(D/v>1e-3)
    out[N]=dict(h_hat=kl_entropy(x), h_true=0.5*np.log(2*np.pi*np.e), rel_gap=[(float(d/v), float(i-gg)) for d,i,gg in zip(D[sel][::10],I[sel][::10],g[sel][::10])])
    print(N, out[N]["h_hat"], out[N]["h_true"]); print([ (round(a,4),round(b,3)) for a,b in out[N]["rel_gap"]])
json.dump(out, open("rd_gauss_check.json","w"))
