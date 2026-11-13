# RodoFauna: predição de atropelamento de fauna

> 🚧 **Projeto em desenvolvimento.** Limpeza, EDA, dados externos, datasets de modelagem, os modelos clássicos, a MLP e a otimização de hiperparâmetros com Optuna estão implementados. O estudo de regularização ainda não existe, e a estrutura, os scripts e os resultados podem mudar sem aviso.

Pipeline de aprendizado de máquina para estudar atropelamentos de fauna silvestre em rodovias do Distrito Federal. Os registros de campo vêm do monitoramento RodoFauna (IBRAM) e são combinados com dados abertos de vias (OpenStreetMap), clima (INMET e NASA POWER), relevo (SRTM) e cobertura do solo (MapBiomas).

Os três problemas planejados são:

- **Multiclasse:** a classe taxonômica do animal atropelado.
- **Binário:** se um trecho de rodovia é um hotspot no mês.
- **Regressão:** quantos atropelamentos ocorrem por trecho e mês.

## Estado atual

| Parte | Situação |
| --- | --- |
| Limpeza raw → interim (`01a_limpeza`) | ✅ pronta |
| Análise exploratória das ocorrências (`01b_eda`): classes, espaço, tempo e associações com a classe | ✅ pronta |
| Dados externos e segmentação das rodovias (`01c_external`) | ✅ pronta |
| EDA de revisão: esforço, zeros, sazonalidade, painel, blocos e alvos (`01d_eda_revisao`) | ✅ pronta |
| Datasets de modelagem e splits (`02_build_datasets`) | ✅ pronta |
| Modelos de referência e clássicos nos três problemas, com ablação (`04`–`06`) | ✅ pronta |
| MLP (PyTorch) nos três problemas, comparada com os clássicos (`src/mlp.py`, `04`–`06`) | ✅ pronta |
| Otimização dos hiperparâmetros da MLP com Optuna, nos três problemas (`07_optuna`) | ✅ pronta |
| Estudo de regularização e overfitting | ⏳ a fazer |

## Ambiente

```bash
python3.11 -m venv venv
venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Execução

Rode os scripts da raiz do projeto, como módulos e nesta ordem:

```bash
venv/bin/python -m experiments.01a_limpeza    # raw → data/interim/silvestres_limpo.csv
venv/bin/python -m experiments.01b_eda        # figuras e tabelas da EDA
venv/bin/python -m experiments.01c_external   # dados externos; precisa de internet
venv/bin/python -m experiments.01d_eda_revisao # EDA de esforço, zeros, sazonalidade, painel, blocos e alvos
venv/bin/python -m experiments.02_build_datasets # atributos e splits → data/processed/
venv/bin/python -m experiments.04_multiclasse   # clássicos + MLP: classe do animal (~2 + 2 min)
venv/bin/python -m experiments.05_binario       # clássicos + MLP: hotspot por segmento × mês (~4 + 6 min)
venv/bin/python -m experiments.06_regressao     # clássicos + MLP: n por segmento × mês (~5 + 5 min)
venv/bin/python -m experiments.07_optuna        # Optuna na MLP: 80 trials por problema e retreino da melhor (~40 min)
```

Os scripts `04`–`06` aceitam `--modelos classicos`, `--modelos mlp` ou `--modelos todos` (padrão). Com `mlp`, só a rede é treinada e as linhas dos clássicos em `outputs/tables/resultados.csv` são preservadas.

O `07_optuna` guarda os estudos em `outputs/optuna.db` (fora do git). Rodar de novo continua de onde parou e, com os 80 trials completos, só refaz a exportação e o retreino; `--reiniciar` apaga o estudo e recomeça, e `--problema` escolhe um dos três. O `04`–`06` com a MLP precisam rodar antes, porque a comparação usa a MLP original em `resultados.csv`.

- **Dados baixados automaticamente:** na primeira execução, o `01c_external` baixa OSM, NASA POWER e SRTM e guarda tudo em cache em `data/external/raw/`.
- **Dados baixados à mão:** os CSVs do INMET e o raster do MapBiomas. O passo a passo está em [`data/external/FONTES.md`](data/external/FONTES.md).

## Regras do projeto

- Só o `01a_limpeza` lê o CSV original; os demais scripts leem `data/interim/` ou `data/processed/`.
- Nada ajustado aos dados (scaler, imputação, limiar de hotspot) é salvo nos CSVs. Esses ajustes ficam dentro dos experimentos e usam só o conjunto de treino.
- Nenhum atributo usa informação do futuro em relação ao que se prevê:
  - clima do mês anterior e cobertura do solo do ano anterior;
  - os lags olham só para o passado e ficam fora do split espacial (grupo `historico`);
  - o split temporal treina até 2013 e testa em 2014–2015 e é o teste final;
  - a busca de hiperparâmetros só enxerga o treino: ajusta até 2011, para o treino por 2012 e mede em 2013.
- Split espacial: blocos de ~5 km ao longo da rodovia com faixa de exclusão de 1 km (`src/splits.py`); o split aleatório é só referência.
- O dicionário das colunas processadas está em `outputs/tables/dicionario_dados.csv`.
- Caminhos, `SEED` e parâmetros de agregação e de split ficam em `src/config.py`.

## Estrutura

```
data/
  original/       CSV original do RodoFauna, nunca alterado
  interim/        silvestres_limpo.csv (1 linha por ocorrência) e campanhas.csv (esforço por campanha)
  processed/      ocorrencias.csv (multiclasse) e segmentos_periodo.csv (binário + regressão), com os splits
  external/       raw/ e processed/ dos dados externos
src/              config, preprocessing, external, features, splits, models, mlp, evaluation
experiments/      scripts numerados na ordem de execução
outputs/          figures/ (eda/, modelos/, optuna/), tables/ (resultados.csv e resumos), models/, optuna.db
```
