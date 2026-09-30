# Scripts de reprodução dos quatro artigos

Estes são os scripts que geraram os números e as figuras dos artigos (MAIN, MOS,
LML e COST), copiados sem alteração dos pacotes de reprodução de 23–24/09/2026.
Eles importam `tese_eliptico`: antes de rodar, copie `legacy/tese_eliptico.py`
para a pasta do script (é o script da tese com `np.trapz` → `np.trapezoid`).

- `artigos_MOS_LML_COST/`: ver `LEIAME.txt` (ordem de execução; sementes 2026101,
  2026103, 2026104 e 7).
- `artigo_MAIN/`: ver `LEIAME.md`. Os caches `_cache_P1_x_NR1000_v14.pkl` não foram
  incluídos (10 MB); `python run_base.py 1.4 1.1 1.3` os regenera, ou use
  `scripts/run_thesis.py` do pacote, que produz resultados idênticos (ver testes
  `test_regression.py::test_articles_end_to_end_parallel`).

Os JSON incluídos são os resultados exatos usados nos artigos. A migração destes
scripts para a API do pacote `rdupscale` é o próximo passo da Fase 0.
