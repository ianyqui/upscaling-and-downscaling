import json, numpy as np
r = json.load(open("master_cost.json")); sc = json.load(open("master_cost_scaling.json"))
To = r["T_offline"]; T_off_DA = To["upscaling"]+To["DA_compression"]+To["copula_fit"]; T_off_up = To["upscaling"]
NAMES = {"1.1": "S1", "1.3": "S2", "1.4": "S3", "1.2": "S4"}
MET = ["err_mean_p", "err_sd_p", "err_mean_Q", "err_sd_Q", "ks_Q"]
def n_eq(mc, e, m):
    ns = np.array(sorted(int(k) for k in mc)); es = np.array([mc[str(k)][m] for k in ns])
    # monotone envelope of the MC curve
    es = np.minimum.accumulate(es)
    if e >= es[0]: return 5.0, "<="
    if e <= es[-1]: return 250.0, ">="
    i = np.where(es <= e)[0][0]
    x0, x1, y0, y1 = np.log(ns[i-1]), np.log(ns[i]), np.log(es[i-1]), np.log(es[i])
    return float(np.exp(x0+(np.log(e)-y0)*(x1-x0)/(y1-y0))), "="
out = {"T_off_DA": T_off_DA, "T_off_upscaling_only": T_off_up, "scen": {}}
for s, S in r["scenarios"].items():
    tf = S["t_bruteforce_250"]/250; o = {"t_fine": tf}
    for v in ("upscaled_no_compression", "DA_independent", "DA_copula"):
        V = S["variants"][v]; t_on = V["t_online"]; toff = T_off_up if v == "upscaled_no_compression" else T_off_DA
        o[v] = {}
        for m in MET:
            n, rel = n_eq(S["mc_with_n"], V[m], m); gain = n*tf-t_on
            o[v][m] = dict(err=V[m], n_eq=n, rel=rel, bf_cost=n*tf, t_on=t_on, S_star=(toff/gain if gain > 0 else None))
    out["scen"][NAMES[s]] = o
json.dump(out, open("master_cost_analysis.json", "w"), indent=1)
for s, o in out["scen"].items():
    print(s, "t_f=%.1f ms" % (1000*o["t_fine"]))
    for v in ("upscaled_no_compression", "DA_independent", "DA_copula"):
        print("   %-24s" % v, " | ".join(f"{m}:{o[v][m]['err']:.3f} n{o[v][m]['rel']}{o[v][m]['n_eq']:.0f} S*={'∞' if o[v][m]['S_star'] is None else round(o[v][m]['S_star'],1)}" for m in MET))
print("T_off DA", T_off_DA, "upscaling only", T_off_up)
