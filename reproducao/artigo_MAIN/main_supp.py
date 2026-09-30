import pickle, tese_eliptico as te
import os; os.makedirs("figuras/supp", exist_ok=True)
PN = {"1.1": "1", "1.3": "2", "1.4": "3"}; LK = {"xx": "Kxx", "yy": "Kyy", "xy": "Kxy"}
for p, n in PN.items():
    res = pickle.load(open(f"_cache_P{p.replace('.','_')}_NR1000_v14.pkl", "rb"))
    for d in ("xx", "yy", "xy"):
        da = res[f"da_all_{d}"]
        te.plot_cdf_25(da, LK[d], f"figuras/supp/S_P{n}_{d}_a.png", n)
        te.plot_codebook_25(da, LK[d], f"figuras/supp/S_P{n}_{d}_b.png", n)
        te.plot_phase_25(da, LK[d], f"figuras/supp/S_P{n}_{d}_c.png", n)
