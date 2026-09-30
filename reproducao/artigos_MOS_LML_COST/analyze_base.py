"""Master metrics from the baseline caches (fixed seeds). Writes master_base.json."""
import pickle, json, glob, numpy as np
import tese_eliptico as te

PROBS = {"1.1": "P1 horizontal Darcy", "1.3": "P2 mixed (bottom/left Dirichlet)", "1.4": "P3 full Dirichlet"}
out = {}
for name, lab in PROBS.items():
    f = f"_cache_P{name.replace('.','_')}_NR1000_v14.pkl"
    res = pickle.load(open(f, "rb"))
    N = res["N_R"]
    r = {"label": lab, "n_R_macro": int(res["n_R"])}
    # --- fidelity of the pressure fields (in-sample, as the pipeline computes) ---
    for fld in ("mean", "std"):
        a = res[f"Pmic_{fld}"].ravel(); b = res[f"Pdown_{fld}"].ravel()
        r[f"R2_{fld}"] = float(1 - np.sum((a-b)**2)/np.sum((a-a.mean())**2))
        r[f"RMSE_{fld}"] = float(np.sqrt(np.mean((a-b)**2)))
        r[f"relL2_{fld}"] = float(np.linalg.norm(a-b)/np.linalg.norm(a))
    # --- per component DA statistics ---
    for d in ("xx", "yy", "xy"):
        da = res[f"da_all_{d}"]
        ks = []; K = []; H = []; PP = []; Ks = []; Tc = []; var = []; quart = []
        for e in da:
            s, yk, qk = e["samples"], e["yk"], e["qk"]
            q = qk/qk.sum()
            ks.append(te._ks_exact(s, yk, q))
            K.append(len(yk)); h = float(-(q[q>0]*np.log(q[q>0])).sum()); H.append(h); PP.append(np.exp(h))
            Ks.append(int((q*N >= 5).sum())); Tc.append(e["Tc"]); var.append(float(np.var(s, ddof=1)))
            qq = te.ks_quartis(s, yk, q); quart.append([qq[f"Q{i}"]["Linf"] for i in range(1,5)])
        ks = np.array(ks); quart = np.array(quart)
        w = int(np.argmax(var))
        r[d] = dict(KS_cellmean=float(ks.mean()), KS_cellmax=float(ks.max()),
                    K_mean=float(np.mean(K)), K_min=int(min(K)), K_max=int(max(K)),
                    Ks_mean=float(np.mean(Ks)),
                    H_mean=float(np.mean(H)), PP_mean_of_exp=float(np.mean(PP)), PP_exp_of_mean=float(np.exp(np.mean(H))),
                    Tc_theory_mean=float(np.mean(Tc)),
                    compression=float(N/np.mean(K)),
                    worst_var_cell=list(map(int, da[w]["IJ"])), quartiles_worst_var_cell=list(map(float, quart[w])),
                    worst_ks_cell=list(map(int, da[int(np.argmax(ks))]["IJ"])), quartiles_worst_ks_cell=list(map(float, quart[int(np.argmax(ks))])),
                    mean_sample=float(np.mean([e["samples"].mean() for e in da])),
                    sd_sample=float(np.mean([e["samples"].std() for e in da])),
                    frac_negative=float(np.mean([np.mean(e["samples"]<0) for e in da])))
    r["Tc_ratio_diag_offdiag"] = float(0.5*(r["xx"]["Tc_theory_mean"]+r["yy"]["Tc_theory_mean"])/r["xy"]["Tc_theory_mean"])
    r["sigma_res_field_zero_frac"] = float(np.mean(res["sigma_res_field"] <= 0))
    out[name] = r
ratios = [out[n]["Tc_ratio_diag_offdiag"] for n in PROBS]
out["Tc_ratio_mean"] = float(np.mean(ratios)); out["Tc_ratio_sd_pop"] = float(np.std(ratios))
json.dump(out, open("master_base.json", "w"), indent=1)
print(json.dumps(out, indent=1))
