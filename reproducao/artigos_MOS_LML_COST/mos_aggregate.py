import json, numpy as np
m = json.load(open("master_mos_multiseed.json"))
agg = {}
for p, v in m.items():
    rows = []
    kmaxes = sorted(set(r["K_max"] for r in v["full"]))
    for k in kmaxes:
        F = [r for r in v["full"] if r["K_max"] == k]; H = [r for r in v["split500"] if r["K_max"] == k]
        f = lambda L, key: (float(np.mean([r[key] for r in L])), float(np.std([r[key] for r in L])))
        rows.append(dict(K_max=k, K=f(F, "K"), misfit=f(F, "misfit"), penalty=f(F, "penalty"), MDL=f(F, "MDL"),
                         KS=f(F, "KS"), Ks=f(F, "Ks"), H=f(F, "H"), K_half=f(H, "K"), KS_in_half=f(H, "KS"), KS_test=f(H, "KS_test")))
    i_mdl = int(np.argmin([r["MDL"][0] for r in rows]))
    ksmin = min(r["KS"][0] for r in rows); i_pl = next(i for i, r in enumerate(rows) if r["KS"][0] <= 1.05*ksmin)
    kstmin = min(r["KS_test"][0] for r in rows); i_plt = next(i for i, r in enumerate(rows) if r["KS_test"][0] <= 1.05*kstmin)
    s = dict(cell=v["cell"], var=v["var"],
             MDL_opt=dict(K_max=rows[i_mdl]["K_max"], K=rows[i_mdl]["K"][0], KS=rows[i_mdl]["KS"][0], KS_test=rows[i_mdl]["KS_test"][0]),
             KS_plateau=dict(K_max=rows[i_pl]["K_max"], K=rows[i_pl]["K"][0], KS=rows[i_pl]["KS"][0]),
             KS_test_plateau=dict(K_max=rows[i_plt]["K_max"], K=rows[i_plt]["K_half"][0], KS_test=rows[i_plt]["KS_test"][0]),
             Ks_at_Kmax58=rows[-1]["Ks"][0], K_at_Kmax58=rows[-1]["K"][0])
    s["ratio_K"] = s["KS_plateau"]["K"]/s["MDL_opt"]["K"]; s["ratio_KS"] = s["MDL_opt"]["KS"]/s["KS_plateau"]["KS"]
    s["ratio_KS_test"] = s["MDL_opt"]["KS_test"]/s["KS_test_plateau"]["KS_test"]
    i9 = rows[i_mdl]; iL = rows[-1]
    s["misfit_change_after_opt"] = iL["misfit"][0]-i9["misfit"][0]; s["penalty_change_after_opt"] = iL["penalty"][0]-i9["penalty"][0]
    agg[p] = dict(rows=rows, summary=s)
    print(p, json.dumps(s, indent=0))
    print(" Kmax    K        misfit        pen      MDL         KS          Ks     KS_test")
    for r in rows:
        print(f" {r['K_max']:3d} {r['K'][0]:5.1f}±{r['K'][1]:4.1f} {r['misfit'][0]:8.1f}±{r['misfit'][1]:5.1f} {r['penalty'][0]:6.1f} {r['MDL'][0]:8.1f}±{r['MDL'][1]:5.1f} {r['KS'][0]:.3f}±{r['KS'][1]:.3f} {r['Ks'][0]:5.1f} {r['KS_test'][0]:.3f}±{r['KS_test'][1]:.3f}")
json.dump(agg, open("master_mos_agg.json", "w"), indent=1)
