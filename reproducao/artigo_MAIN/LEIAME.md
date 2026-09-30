# Pacote de reprodução — artigo principal (MAIN)

"Stochastic Upscaling as Rate–Distortion Coding: Phase Transitions, Achievability Bounds and the Complexity of Coarse-Scale Permeability Ensembles"

Ambiente usado: Python 3.11, NumPy 2.4, SciPy 1.17, Matplotlib 3.10, joblib 1.5. Os tempos da busca aleatória (Seção 5.5) foram medidos em fio único; defina OMP_NUM_THREADS=1 antes de rodar.

## Conteúdo

| Arquivo | Papel |
|:--|:--|
| tese_eliptico.py | Implementação original (malha, geoestatística, TPFA, upscaling por experimentos canônicos, recozimento determinístico). Única alteração: np.trapz → np.trapezoid (NumPy ≥ 2.0). |
| run_base.py | Gera os 3 × 1000 campos (sementes 2026101, 2026103, 2026104) e roda o fluxo de produção → _cache_P1_x_NR1000_v14.pkl |
| _cache_P1_1/1_3/1_4_NR1000_v14.pkl | Resultados de produção já calculados (tensores, representações por célula, históricos do recozimento). Problema 1 = 1.1, Problema 2 = 1.3, Problema 3 = 1.4 |
| measure_tc.py | Réplica do recozimento que registra a primeira divisão (usada por main_exp.py) |
| main_exp.py | `tc` → main_tc_cells.json (Seção 5.1, Tabela 1, Figura 2); `search <prob> <n>` → main_search_P1_x.json (Seção 5.5) |
| main_search_ref.py | Controles de produção nas mesmas células, dois trios de sementes → main_search_ref.json |
| main_search_an.py | Fronteira acurácia–custo e correlações de postos → main_search_an.json (Tabela 5) |
| main_rd.py | Limites de Shannon, negentropia, trajetórias (Seção 5.2, Tabela 2, Figuras 3–4) → main_rd.json |
| main_summary.py | Complexidade no ponto de operação, quartis, estatísticas das fontes, correlações (Seções 2.2, 5.1, 5.4, 5.6; Tabela 3) → main_summary.json |
| picard_check.py | Saturação das iterações internas (Seção 5.1) → picard_check.json |
| rd_gauss_check.py, rd_lognormal_check.py | Otimismo de amostra finita frente aos limites, fonte gaussiana e log-normal de entropia conhecida (Seção 5.2) |
| main_flow.py | Micromalha × macromalha (Seção 5.7, Tabelas 6–7): 1000 soluções finas contra quatro ensembles grossos (sem compressão; DA 25 e 1000 independentes; DA 1000 com cópula gaussiana), erros absolutos e relativos por célula, piso amostral → main_flow.json, main_flow_fields.pkl |
| main_flow_figures.py | Figuras 7–9 (mapas de pressão média e desvio-padrão, erros absolutos e relativos, resumo) → figuras/ |
| main_flow_xlsx.py | Planilha Resultados_Micro_x_Macro_MAIN.xlsx: valores por célula, erros absolutos e relativos em fórmulas e resumo por problema |
| main_figures.py | Figuras 1–6 → figuras/ |
| main_supp.py | 27 figuras suplementares por célula → figuras/supp/ |

## Ordem de execução

```
export OMP_NUM_THREADS=1
# opcional (os caches já estão incluídos; ~4 min por problema):
python run_base.py 1.4 1.1 1.3
python main_exp.py tc                      # ~10 min
python main_exp.py search 1.4 30           # ~20 min; o artigo usa 30 configurações
python main_exp.py search 1.1 20
python main_exp.py search 1.3 20
python main_search_ref.py
python main_search_an.py
python main_rd.py
python main_summary.py
python picard_check.py
python rd_gauss_check.py
python rd_lognormal_check.py
python main_flow.py                       # ~1 min
python main_flow_figures.py
python main_flow_xlsx.py
python main_figures.py
python main_supp.py
```

Os arquivos JSON incluídos são os resultados exatos usados no artigo. Todas as sementes são fixas: por célula e componente no recozimento de produção, por problema na busca aleatória (hash SHA-256 do nome do problema). Os tempos de execução dependem da máquina; as razões entre configurações são o que o artigo usa.

## Correspondência entre números do artigo e arquivos

- Tabela 1, Figura 2 e "220 de 225 execuções": main_tc_cells.json (razão T_split/Tc_theory por célula)
- Tabela 2, Figura 3: main_rd.json; dispersão entre estimadores de entropia: recalculada com scipy.stats.differential_entropy (vasicek, ebrahimi) sobre as amostras dos caches
- Tabela 3, Seção 5.6, correlações 0,99 e 0,39: main_summary.json
- Tabela 5, Figura 6: main_search_P1_x.json, main_search_ref.json, main_search_an.json
- Saturação de 89%/91% (40 iterações) e 76%/83% (150): picard_check.json
- Otimismo de amostra finita (0,025 nats gaussiana; 0,002/0,05/0,09 nats log-normal): rd_gauss_check.json, rd_lognormal_check.json
- Tabelas 6–7, Figuras 7–9 e Seção 5.7 (erros absolutos e relativos micromalha × macromalha): main_flow.json e Resultados_Micro_x_Macro_MAIN.xlsx
