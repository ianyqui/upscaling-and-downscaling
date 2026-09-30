"""Roda o pipeline da Aplicação 1 da tese (problemas P1.1 a P1.5).

Exemplos:
  python scripts/run_thesis.py --prob 1.1                       # base da tese (data/reference/tese)
  python scripts/run_thesis.py --base artigos --prob 1.4        # reproduz os caches dos artigos
  python scripts/run_thesis.py --prob all --out resultados --plots
  python scripts/run_thesis.py --prob 1.3 --seed-geo 123 --seed-macro 7 --da-params meu.json

Diferenças em relação ao script original: as sementes do campo geoestatístico
e das macrorrealizações são sempre fixadas e gravadas junto do resultado.
"""
import argparse
import json
import time
from pathlib import Path

from rdupscale import DAParams, Grid, PROBLEMS, run_problem, sample_permeability
from rdupscale.io import read_seed_file, save_cache

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", default="all", help="1.1 … 1.5 ou all")
    ap.add_argument("--base", choices=["tese", "artigos"], default="tese",
                    help="artigos: sementes 2026101/3/4, macro 7, DAParams.legacy()")
    ap.add_argument("--N_R", type=int, default=1000, help="realizações finas")
    ap.add_argument("--n_R", default="25", help="macrorrealizações (inteiro ou auto)")
    ap.add_argument("--seed-geo", type=int, default=None)
    ap.add_argument("--seed-macro", type=int, default=None)
    ap.add_argument("--da-params", default=None, help="JSON com hiperparâmetros do DA (padrão: da_params.json da referência da tese)")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--n-jobs", type=int, default=-1, help="processos (-1 = todos os núcleos)")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()

    grid = Grid()
    if a.base == "artigos":
        probs = ["1.1", "1.3", "1.4"] if a.prob == "all" else [a.prob]
    else:
        probs = list(PROBLEMS) if a.prob == "all" else [a.prob]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in probs:
        pid = name.replace(".", "_")
        ref = ROOT / "data" / "reference" / a.base / f"P{pid}"
        seed_geo = a.seed_geo if a.seed_geo is not None else read_seed_file(ref / "seed_geo.txt")["seed_geo"]
        if a.base == "artigos":
            seed_macro = a.seed_macro if a.seed_macro is not None else 7
            params = DAParams.from_json(a.da_params) if a.da_params else DAParams.legacy()
        else:
            seed_macro = a.seed_macro if a.seed_macro is not None else seed_geo + 1
            params = DAParams.from_json(a.da_params or ref / "da_params.json")
        n_R = "auto" if str(a.n_R).lower() == "auto" else int(a.n_R)
        t0 = time.time()
        K = sample_permeability(a.N_R, grid, seed=seed_geo)
        res = run_problem(name, PROBLEMS[name], K, grid, params, n_R=n_R,
                          seed_macro=seed_macro, n_jobs=a.n_jobs)
        res["seed_geo"] = seed_geo
        save_cache(res, out / f"P{pid}_NR{a.N_R}.pkl")
        (out / f"P{pid}_meta.json").write_text(json.dumps(
            {"prob": name, "seed_geo": seed_geo, "seed_macro": seed_macro, "N_R": a.N_R,
             "n_R": res["n_R"], "da_params": params.to_dict(), "rms": res["rms"],
             "tempo_s": round(time.time() - t0, 1)}, indent=2))
        print(f"P{name}: rms={res['rms']:.4g}  ({time.time() - t0:.1f}s)")
        if a.plots:
            from rdupscale import plots
            plots.plot_campos(res, out / f"P{pid}_panel1_fields.png")
            for d in ("xx", "yy", "xy"):
                plots.plot_da(res, out / f"P{pid}_panel3_DA_K{d}.png", dkey=d)
            plots.plot_lml(res, out / f"P{pid}_panel4_LML.png")


if __name__ == "__main__":
    main()
