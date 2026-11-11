# RodoFauna: predição de atropelamento de fauna

> 🚧 **Projeto em desenvolvimento.** Por enquanto só a limpeza dos dados, a análise exploratória e a coleta de dados externos estão implementadas. Os datasets de modelagem, os modelos e a avaliação ainda não existem, e a estrutura, os scripts e os resultados podem mudar sem aviso.

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
| Modelos: multiclasse, binário e regressão | ⏳ a fazer |
| Otimização de hiperparâmetros e estudo de regularização | ⏳ a fazer |

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
```

- **Dados baixados automaticamente:** na primeira execução, o `01c_external` baixa OSM, NASA POWER e SRTM e guarda tudo em cache em `data/external/raw/`.
- **Dados baixados à mão:** os CSVs do INMET e o raster do MapBiomas. O passo a passo está em [`data/external/FONTES.md`](data/external/FONTES.md).

## Regras do projeto

- Só o `01a_limpeza` lê o CSV original; os demais scripts leem `data/interim/` ou `data/processed/`.
- Nada ajustado aos dados (scaler, imputação, limiar de hotspot) é salvo nos CSVs. Esses ajustes ficam dentro dos experimentos e usam só o conjunto de treino.
- Nenhum atributo usa informação do futuro em relação ao que se prevê:
  - clima do mês anterior e cobertura do solo do ano anterior;
  - os lags olham só para o passado e ficam fora do split espacial (grupo `historico`);
  - o split temporal treina até 2013 e testa em 2014–2015 e é o teste final.
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
src/              config, preprocessing, external, features, splits
experiments/      scripts numerados na ordem de execução
outputs/          figures/, tables/, models/
```
