# upscaling-and-downscaling (`rdupscale`)

Stochastic upscaling of permeability ensembles as rate–distortion coding, solved by
Deterministic Annealing (DA), and stochastic LML downscaling for single-phase Darcy
flow in heterogeneous porous media.

*Versão em português: [README.pt-BR.md](README.pt-BR.md).*

## What is in this repository

| Path | Content |
| --- | --- |
| `src/rdupscale/` | The Python package: log-normal permeability fields, TPFA finite volumes, two-experiment (Durlofsky) upscaling, Deterministic Annealing, LML downscaling, metrics, pipeline, plots |
| `scripts/run_thesis.py` | Command-line pipeline for the five elliptic test problems |
| `tests/` | Unit tests (TPFA, upscaling, DA) and regression tests against published results |
| `data/reference/` | Seeds, hyperparameters and compact reference arrays used by the regression tests |
| `reproducao/` | Scripts that produced the numbers and figures of the associated manuscripts |
| `legacy/tese_eliptico.py` | Original single-file implementation that the reproduction scripts import |

## Installation

```bash
git clone https://github.com/ianyqui/upscaling-and-downscaling.git
cd upscaling-and-downscaling
pip install -r requirements-lock.txt      # tested versions
pip install -e ".[test]"
```

or `conda env create -f environment.yml`. Python ≥ 3.10.

## Quick start

```python
import rdupscale as rd

K = rd.sample_permeability(1000, seed=2026104)               # 1000 fine-scale fields, 15 x 15
res = rd.run_problem("1.4", rd.PROBLEMS["1.4"], K,
                     da_params=rd.DAParams(), n_R=25, seed_macro=7)
print(res["rms"])
```

```bash
python scripts/run_thesis.py --base artigos --prob 1.4      # reproduces the published run bit for bit
python scripts/run_thesis.py --prob all --plots              # all five problems, with figures
```

The pipeline runs the 75 annealing problems of each test case in parallel processes
(`n_jobs=-1` uses every core); results are identical to a serial run.

### Merge tolerance of the annealing

`DAParams()` fuses two representative values when their distance is below
`merge_tol × standard deviation` of the cell's samples, which makes the annealing
invariant to the scale of the property. `DAParams.legacy()` uses the absolute
tolerance of the original implementation and is the setting that reproduces the
published results.

## Tests

```bash
pytest               # ~1.5 min: unit tests and regression against published results
pytest --runslow     # adds the full set of annealing codebooks and an end-to-end parallel run
```

## Reproducibility

All random seeds are fixed and stored with every result. The regression tests
check, against the published caches, the permeability fields, fine-scale pressures,
effective tensors, all 75 annealing codebooks per problem, the coarse realizations
and the downscaled fields.

## How to cite

See [`CITATION.cff`](CITATION.cff). A Zenodo DOI is assigned to each release.

## License

MIT — see [`LICENSE`](LICENSE).
