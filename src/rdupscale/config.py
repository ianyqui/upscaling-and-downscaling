"""Configuração da malha e dos hiperparâmetros do Recozimento Determinístico.

Substitui as constantes globais do script da tese (LX, NF, NC, MU, DEFAULT_*,
DA_*) por objetos imutáveis passados explicitamente às funções. Os valores
padrão são exatamente os da tese.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class Grid:
    """Malha estruturada quadrada: micromalha nf x nf e macromalha nc x nc."""

    lx: float = 15.0   # dimensão do domínio (m); o domínio é quadrado
    nf: int = 15       # células da micromalha por direção
    nc: int = 5        # células da macromalha por direção
    mu: float = 1.0    # viscosidade (cP); mobilidade = 1/mu

    def __post_init__(self):
        if not (isinstance(self.nf, int) and isinstance(self.nc, int)):
            raise TypeError("nf e nc devem ser inteiros")
        if self.nf <= 0 or self.nc <= 0:
            raise ValueError("nf e nc devem ser positivos")
        if self.nf % self.nc != 0:
            raise ValueError(f"nf ({self.nf}) deve ser múltiplo de nc ({self.nc})")

    @property
    def ly(self) -> float:
        return self.lx

    @property
    def alpha(self) -> int:
        """Número de microcélulas por macrocélula, por direção."""
        return self.nf // self.nc

    @property
    def hf(self) -> float:
        return self.lx / self.nf

    @property
    def hc(self) -> float:
        return self.lx / self.nc


@dataclass(frozen=True)
class GeostatParams:
    """Campo log-normal de permeabilidade (kernel gaussiano anisotrópico)."""

    ell_x: float = 3.0
    ell_y: float = 3.0
    sigma_lnk: float = 0.9
    k_med: float = 1.0


@dataclass(frozen=True)
class DAParams:
    """Hiperparâmetros do Recozimento Determinístico.

    Tolerância de fusão: com ``merge_relative=True`` (padrão), dois valores
    representativos são fundidos quando a distância entre eles é menor que
    ``merge_tol × desvio-padrão das amostras`` da macrocélula, o que torna o
    critério independente da escala da propriedade. Com ``merge_relative=False``
    a tolerância é absoluta, como no script da tese (use ``DAParams.legacy()``
    ou JSON antigos para reproduzir resultados já publicados).
    """

    alpha: float = 0.953
    K_max: int = 50
    n_inner: int = 40
    perturb: float = 4.6e-3
    merge_tol: float = 3.8e-4
    merge_relative: bool = True
    f_lo: float = 0.76
    f_hi: float = 1.0
    T_max_ratio: float = 2.0
    T_min_ratio: float = 3e-3

    @classmethod
    def legacy(cls, **kw) -> "DAParams":
        """Padrões do script da tese, com tolerância de fusão absoluta."""
        return cls(merge_relative=False, **kw)

    @classmethod
    def from_json(cls, path: str | Path) -> "DAParams":
        """Lê um JSON de hiperparâmetros. Arquivos da tese (chave ``merge_atol``,
        sem ``merge_relative``) são lidos com tolerância absoluta, como foram usados."""
        with open(path) as f:
            raw = json.load(f)
        keys = set(cls.__dataclass_fields__)
        kw = {k: raw[k] for k in keys if k in raw}
        if "merge_atol" in raw and "merge_tol" not in raw:
            kw["merge_tol"] = raw["merge_atol"]
            kw.setdefault("merge_relative", False)
        if "K_max" in kw:
            kw["K_max"] = int(kw["K_max"])
        if "n_inner" in kw:
            kw["n_inner"] = int(kw["n_inner"])
        return cls(**kw)

    def to_dict(self) -> dict:
        return asdict(self)

    def merge_atol(self, sd: float) -> float:
        """Tolerância absoluta efetiva para uma amostra com desvio-padrão sd."""
        return self.merge_tol * sd if self.merge_relative else self.merge_tol


@dataclass(frozen=True)
class LMLParams:
    """Parâmetros do downscaling LML estocástico (valores da tese)."""

    degree: int = 4
    n_stoc_min: int = 500
    rho_clip_lo: float = 1e-3
    rho_clip_hi: float = 0.999
    var_floor: float = 1e-15
    seed: int = 0xBEEF
