# rdupscale

Upscaling estocástico por taxa–distorção (Recozimento Determinístico) e downscaling
LML para escoamento monofásico em meios porosos heterogêneos. Pacote Python que
reorganiza o código da tese (`tese_eliptico.py`) sem alterar a numérica.

## Instalação

```bash
conda env create -f environment.yml      # ou: pip install -r requirements-lock.txt && pip install -e ".[test]"
conda activate rdupscale
```

## Uso

```bash
# reproduz um problema da Aplicação 1 (semente e hiperparâmetros de data/reference)
python scripts/run_thesis.py --prob 1.1 --out resultados --plots

# todos os problemas; usa todos os núcleos por padrão (--n-jobs -1)
python scripts/run_thesis.py --prob all
```

```python
import rdupscale as rd
K = rd.sample_permeability(1000, seed=1730619031)
res = rd.run_problem("1.1", rd.PROBLEMS["1.1"], K, da_params=rd.DAParams(), seed_macro=1)
# DAParams(): tolerância de fusão relativa (padrão novo)
# DAParams.legacy(): tolerância absoluta, como nos artigos publicados
```

## Testes

```bash
pytest                 # ~1,5 min: TPFA, upscaling, DA e regressão (artigos e tese)
pytest --runslow       # + codebooks completos e pipeline paralelo de ponta a ponta
```

## Estrutura

| Módulo | Conteúdo |
| --- | --- |
| `config.py` | `Grid`, `GeostatParams`, `DAParams`, `LMLParams` (substituem as constantes globais) |
| `geostat.py` | campo log-normal (Cholesky; FFT para malhas grandes) |
| `fv.py` | TPFA: montagem, solução, fluxos nas faces, velocidade de Darcy |
| `upscaling.py` | tensor efetivo por dois experimentos canônicos (Durlofsky) |
| `da.py` | Recozimento Determinístico escalar |
| `lml.py` | downscaling LML determinístico e estocástico |
| `metrics.py` | K-S exato, K-S por quartil, regra de n_R |
| `pipeline.py` | micro → upscaling → DA → macro → LML, em etapas separadas |
| `tuning.py` | busca aleatória dos hiperparâmetros do DA |
| `plots.py` | painéis da tese |
| `io.py` | caches e arquivos de semente |

`data/reference/artigos/` e `data/reference/tese/` guardam as sementes, os
hiperparâmetros e um `reference.npz` extraído dos caches (usados nos testes).
`reproducao/` contém os scripts que geraram os números dos quatro artigos e
`legacy/tese_eliptico.py`, o script original de que eles dependem.

## Estado da reprodução

Todos os resultados publicados são reproduzidos pelos testes de regressão (sementes fixas).

Versão 0.2.0: tolerância de fusão relativa por padrão, execução paralela em processos, referências dos artigos e da tese.
