"""rdupscale — upscaling estocástico por taxa–distorção (Recozimento Determinístico)
e downscaling LML para escoamento monofásico em meios heterogêneos."""
from .config import Grid, GeostatParams, DAParams, LMLParams
from .geostat import sample_permeability, correlation_matrix
from .fv import tpfa, assemble, face_fluxes, darcy_velocity, grads
from .upscaling import upscale_realisation
from .da import deterministic_annealing, da_seed
from .lml import lml_downscale, lml_downscale_stochastic, calibrate_lml_residual
from .metrics import ks_exact, ks_quartis, recommend_n_R
from .pipeline import run_problem, micro_stage, da_stage, macro_stage, lml_stage
from .problems import PROBLEMS

__version__ = "0.2.0"
__all__ = [
    "Grid", "GeostatParams", "DAParams", "LMLParams",
    "sample_permeability", "correlation_matrix",
    "tpfa", "assemble", "face_fluxes", "darcy_velocity", "grads",
    "upscale_realisation", "deterministic_annealing", "da_seed",
    "lml_downscale", "lml_downscale_stochastic", "calibrate_lml_residual",
    "ks_exact", "ks_quartis", "recommend_n_R",
    "run_problem", "micro_stage", "da_stage", "macro_stage", "lml_stage",
    "PROBLEMS", "__version__",
]
