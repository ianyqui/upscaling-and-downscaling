"""Gera os dados de referência dos testes a partir dos caches .pkl.

Uso:
  python scripts/make_reference.py artigos <pasta_caches_artigos>   # P1.1, P1.3, P1.4 (sementes fixas)
  python scripts/make_reference.py tese    <pasta_caches_tese>      # P1.1–P1.5 (caches de 02/08/2026)
Guarda só o necessário para os testes (alguns MB)."""
import json, pickle, shutil, sys
from pathlib import Path
import numpy as np

kind, src = sys.argv[1], Path(sys.argv[2])
dst = Path(__file__).resolve().parents[1] / "data" / "reference" / kind
N_SUB = 50
ART_SEEDS = {"1_1": 2026101, "1_3": 2026103, "1_4": 2026104}
probs = list(ART_SEEDS) if kind == "artigos" else [f"1_{i}" for i in range(1, 6)]
for pid in probs:
    c = pickle.load(open(src / f"_cache_P{pid}_NR1000_v14.pkl", "rb"))
    out = dst / f"P{pid}"; out.mkdir(parents=True, exist_ok=True)
    if kind == "artigos":
        (out / "seed_geo.txt").write_text(f"prob={pid.replace('_', '.')}\nseed_geo={ART_SEEDS[pid]}\nN_R=1000\n")
        meta = {"seed_macro": 7, "da_params": "DAParams.legacy()", "n_R": int(c["n_R"])}
    else:
        shutil.copy(src / f"_cache_P{pid}_seed_geo.txt", out / "seed_geo.txt")
        shutil.copy(src / f"tune_da_best_P{pid}.json", out / "tune_da_best.json")
        shutil.copy(src / f"cache_da_params_P{pid}.json", out / "da_params.json")
        meta = {"seed_macro": None, "da_params": "da_params.json", "n_R": int(c["n_R"])}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    arrs = dict(K_micro_first_realiz=c["K_micro_first_realiz"], K_micro_mean=c["K_micro_mean"],
                K_micro_std=c["K_micro_std"], P_micro_sub=c["P_micro_all"][:N_SUB],
                Pmic_mean=c["Pmic_mean"], Pmic_std=c["Pmic_std"],
                Kxx_all=c["Kxx_all"], Kyy_all=c["Kyy_all"], Kxy_all=c["Kxy_all"],
                P_macro_all=c["P_macro_all"], Pdown_mean=c["Pdown_mean"], Pdown_std=c["Pdown_std"],
                rms=np.array(c["rms"]))
    for d in ("xx", "yy", "xy"):
        for e in c[f"da_all_{d}"]:
            I, J = e["IJ"]
            arrs[f"da_{d}_{I}{J}_yk"] = e["yk"]; arrs[f"da_{d}_{I}{J}_qk"] = e["qk"]
            arrs[f"da_{d}_{I}{J}_h"] = e["h"]
    np.savez_compressed(out / "reference.npz", **arrs)
    print(kind, pid, (out / "reference.npz").stat().st_size // 1024, "kB")
