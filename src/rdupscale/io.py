"""Leitura e gravação de caches e arquivos auxiliares."""
from __future__ import annotations

import pickle
from pathlib import Path


def read_seed_file(path) -> dict:
    """Lê arquivos ``_cache_P1_X_seed_geo.txt`` (linhas chave=valor)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    if "seed_geo" in out:
        out["seed_geo"] = int(out["seed_geo"])
    if "N_R" in out:
        out["N_R"] = int(out["N_R"])
    return out


def load_cache(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_cache(res: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
