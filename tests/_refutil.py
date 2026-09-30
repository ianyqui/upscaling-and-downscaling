from pathlib import Path
import json
import numpy as np

from rdupscale import DAParams, sample_permeability
from rdupscale.io import read_seed_file

REF = Path(__file__).resolve().parents[1] / "data" / "reference"
_cache = {}


def load(kind, pid):
    key = (kind, pid)
    if key not in _cache:
        d = REF / kind / f"P{pid}"
        ref = dict(np.load(d / "reference.npz"))
        meta = json.loads((d / "meta.json").read_text())
        seed = read_seed_file(d / "seed_geo.txt")["seed_geo"]
        prm = DAParams.legacy() if kind == "artigos" else DAParams.from_json(d / "da_params.json")
        _cache[key] = (ref, sample_permeability(1000, seed=seed), prm, meta)
    return _cache[key]


def da_all_from_ref(ref, nc=5):
    out = {}
    for d in ("xx", "yy", "xy"):
        out[d] = [dict(IJ=(I, J), yk=ref[f"da_{d}_{I}{J}_yk"], qk=ref[f"da_{d}_{I}{J}_qk"])
                  for I in range(nc) for J in range(nc)]
    return out
