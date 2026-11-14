from pathlib import Path

# =============================================================================
# CAMINHOS
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / 'data'
RAW_CSV = DATA_DIR / 'original' / 'rodofauna_original.csv'
INTERIM_CSV = DATA_DIR / 'interim' / 'silvestres_limpo.csv'
CAMPANHAS_CSV = DATA_DIR / 'interim' / 'campanhas.csv'
OCORRENCIAS_CSV = DATA_DIR / 'processed' / 'ocorrencias.csv'
SEGMENTOS_CSV = DATA_DIR / 'processed' / 'segmentos_periodo.csv'
EXTERNAL_RAW_DIR = DATA_DIR / 'external' / 'raw'
EXTERNAL_PROCESSED_DIR = DATA_DIR / 'external' / 'processed'

OUTPUTS_DIR = ROOT / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
EDA_FIG_DIR = FIGURES_DIR / 'eda'
TABLES_DIR = OUTPUTS_DIR / 'tables'
MODELS_DIR = OUTPUTS_DIR / 'models'
DECISOES_LIMPEZA_CSV = TABLES_DIR / 'decisoes_limpeza.csv'
QUALIDADE_ORIGINAL_CSV = TABLES_DIR / 'qualidade_base_original.csv'

# =============================================================================
# PARÂMETROS GERAIS
# =============================================================================
SEED = 42

# Sistema de coordenadas dos dados do IBRAM (SIRGAS 2000 / UTM 23S)
CRS = 'EPSG:31983'

# Unidade de análise do binário e da regressão (Etapa 2): segmento de 500 m da
# malha do OSM × mês. Alvo da regressão: n = atropelamentos no segmento no mês.
# Hotspot: n >= HOTSPOT_MIN_N (com 80% de zeros, qualquer percentil do treino
# cai em 0 ou 1, então o limiar é fixo e não precisa ser ajustado no treino);
# n >= 2 entra só como análise de sensibilidade
SEGMENT_SIZE_M = 500
AGG_PERIOD = 'M'
HOTSPOT_MIN_N = 1
HOTSPOT_MIN_N_SENSITIVITY = 2

# Splits. Bloco espacial = trecho contínuo de ~5 km ao longo da rodovia (a
# correlação entre segmentos some por volta de 5 km); em cada rodada, os
# segmentos de treino a até SPATIAL_BUFFER_M do bloco de validação saem do treino
SPATIAL_BLOCK_M = 5000
SPATIAL_BUFFER_M = 1000
SPATIAL_N_FOLDS = 5
TEMPORAL_SPLIT_LAST_TRAIN_YEAR = 2013

# =============================================================================
# DADOS EXTERNOS
# =============================================================================
AOI_BUFFER_M = 3000          # margem da área de interesse em torno dos registros
ROUTE_MIN_RECORDS = 20       # rodovia entra como rota monitorada com >= 20 registros
ROUTE_BUFFER_M = 1000        # faixa em torno dos registros que define a extensão da rota
PARALLEL_TOL_M = 40          # pistas duplas: trechos paralelos a menos disso viram um só
SEGMENT_MIN_M = 250          # trecho menor que isso é anexado ao vizinho
SEGMENT_MAX_GAP_M = 60       # distância máxima até o vizinho para anexar
ASSIGN_MAX_DIST_M = 200      # registro → segmento mais próximo, até esta distância
INMET_MIN_COVERAGE = 0.9     # mês do INMET com menos horas válidas que isso usa o NASA POWER
CLIMATE_YEARS = (2009, 2015) # 2009 para ter a chuva do mês anterior a abril de 2010
INMET_A001_LATLON = (-15.78944444, -47.92583332)
CLIMATE_WINDOW = ('2010-04', '2015-03')   # período das campanhas
ELEV_VALID_RANGE_M = (600, 1400)          # fora disso no DF é artefato do SRTM → sem dado
MAPBIOMAS_YEARS = list(range(2010, 2016))
VEG_RADIUS_M = 500           # raio da % de vegetação nativa em volta de cada pixel
# Vegetação nativa na legenda do MapBiomas: formações florestal (3), savânica (4),
# mangue (5), floresta alagável (6), restinga arbórea (49), campo alagado (11),
# formação campestre (12), outras não florestais (13), afloramento rochoso (29),
# apicum (32) e restinga herbácea (50)
MAPBIOMAS_NATIVE_CLASSES = [3, 4, 5, 6, 11, 12, 13, 29, 32, 49, 50]

# =============================================================================
# COLUNAS
# =============================================================================
# Proibidas como preditores pela especificação. 'mes' não é proibida, mas é
# redundante (sai de data_dt) e fica fora pelo mesmo motivo
PROHIBITED_FEATURES = [
    'id', 'silvestre_', 'hora', 'sexo', 'posicao', 'ordem', 'familia',
    'nome_popul', 'campanha', 'periodo',
]
REDUNDANT_COLUMNS = ['mes']

# Mantidas no interim só como colunas auxiliares, nunca como atributos
AUX_COLUMNS = ['especie_nm', 'campanha']

INTERIM_COLUMNS = ['data', 'ano', 'classe', 'especie_nm', 'uc', 'x', 'y', 'campanha', 'dia_da_sem']

# =============================================================================
# ATRIBUTOS DOS DATASETS PROCESSADOS (Etapa 4)
# =============================================================================
# Regra: nenhum atributo usa informação posterior ao que se prevê. Clima entra
# com o mês anterior; a cobertura do solo (anual) entra com o ano anterior
# (2010 usa o próprio ano: não há banda de 2009); os lags usam só o passado.
VEG_YEAR_LAG = 1

# Grupos de atributos, usados na ablação da Etapa 5. 'historico' usa o alvo
# passado do próprio segmento e por isso fica fora do split espacial
FEATURE_GROUPS = {
    'base': ['x', 'y', 'x_mais_y', 'x_menos_y', 'uc', 'mes_sin', 'mes_cos', 'seco'],
    'clima': ['prec_mes_anterior_mm', 'temp_mes_anterior_c'],
    'via': ['tipo_via', 'velocidade_max', 'faixas'],
    'relevo_agua': ['elevacao_m', 'dist_agua_m'],
    'vegetacao': ['veg_nativa_pct'],
    'historico': ['n_lag_1', 'n_media_12m', 'n_mesmo_mes_ano_anterior', 'meses_historico'],
}
# Só nas ocorrências (a data exata é conhecida)
OCCURRENCE_EXTRA_FEATURES = ['dia_ano_sin', 'dia_ano_cos']
# Só no painel (tamanho do segmento)
PANEL_EXTRA_FEATURES = ['comprimento_m']
CATEGORICAL_FEATURES = ['uc', 'tipo_via']

# Colunas que não são atributos: alvos, identificadores e splits
TARGET_COLUMNS = {'ocorrencias': 'classe', 'segmentos_periodo': 'n'}
SPLIT_COLUMNS = ['bloco_espacial', 'fold_espacial', 'fold_aleatorio', 'split_temporal']
META_COLUMNS = {
    'ocorrencias': ['data', 'ano', 'segmento_id'],
    'segmentos_periodo': ['segmento_id', 'ano_mes', 'ano', 'mes', 'n_campanhas'],
}
# Colunas que nunca podem aparecer nos arquivos processados: as proibidas pela
# especificação e as que entregariam o alvo (especie_nm define a classe)
FORBIDDEN_IN_PROCESSED = PROHIBITED_FEATURES + ['especie_nm', 'dia_da_sem', 'mes_num']

# =============================================================================
# EXPERIMENTOS (Etapa 5 em diante)
# =============================================================================
MODEL_FIG_DIR = FIGURES_DIR / 'modelos'
RESULTS_CSV = TABLES_DIR / 'resultados.csv'
# Conjuntos de atributos da ablação dos dados externos. O grupo 'historico'
# (lags do próprio segmento) é acrescentado só nos splits temporal e aleatório
ABLATION_SETS = {
    'base': ['base'],
    'base+clima': ['base', 'clima'],
    'base+via': ['base', 'via'],
    'base+paisagem': ['base', 'relevo_agua', 'vegetacao'],
    'completo': ['base', 'clima', 'via', 'relevo_agua', 'vegetacao'],
}
HISTORY_GROUP = 'historico'
SPLITS = ['aleatorio', 'espacial', 'temporal']
PERMUTATION_REPEATS = 10

# =============================================================================
# MLP (Etapa 6)
# =============================================================================
# Configuração inicial: rede pequena para dados tabulares com poucos milhares
# de linhas; Adam com lr padrão; early stopping pela loss de validação.
# Dropout e weight decay ficam em zero aqui e são estudados na Etapa 8
MLP_DEFAULTS = dict(hidden=(64, 32), activation='relu', dropout=0.0, weight_decay=0.0, optimizer='adam',
                    lr=1e-3, batch_size=64, max_epochs=300, patience=20, early_stopping=True)
MLP_SEEDS_CV = [42, 43, 44]
MLP_SEEDS_TEST = [42, 43, 44, 45, 46]
MLP_FEATURE_SETS = ['base', 'completo']
MLP_INNER_VAL_FRACTION = 0.15   # validação do early stopping no split aleatório
TORCH_THREADS = 4

# =============================================================================
# OPTUNA (Etapa 7)
# =============================================================================
OPTUNA_FIG_DIR = FIGURES_DIR / 'optuna'
OPTUNA_DB = OUTPUTS_DIR / 'optuna.db'
OPTUNA_N_TRIALS = 80
OPTUNA_STARTUP_TRIALS = 10        # trials sorteados antes de o TPE começar a propor
OPTUNA_SEEDS_TRIAL = [42, 43, 44]  # cada trial é a média de 3 treinos
OPTUNA_SEEDS_CONFIRM = list(range(100, 110))  # sementes fora da busca, para confirmar o ganho na validação
# Espaço de busca. Limites pensados para ~5 mil linhas e ~20 atributos:
#   n_layers / n_units: de uma camada de 16 (quase um modelo linear) a 256 → 128 → 64,
#       já muito acima do que os dados sustentam; cada camada tem a metade da anterior
#   lr: uma década para cada lado do padrão do Adam (1e-3)
#   batch_size: de 32 (gradiente ruidoso) a 256 (~20 passos por época)
#   dropout e weight_decay: de praticamente nenhum a regularização forte
OPTUNA_SEARCH_SPACE = dict(
    n_layers=(1, 3), n_units=(16, 256), activation=['relu', 'tanh'], optimizer=['adam', 'rmsprop', 'sgd'],
    lr=(1e-4, 1e-2), batch_size=[32, 64, 128, 256], dropout=(0.0, 0.5), weight_decay=(1e-6, 1e-2),
)
# Primeiro trial: a configuração inicial da Etapa 6 (weight decay no piso do espaço)
OPTUNA_FIRST_TRIAL = dict(n_layers=2, n_units=64, activation='relu', optimizer='adam', lr=1e-3,
                          batch_size=64, dropout=0.0, weight_decay=1e-6)

# =============================================================================
# REGULARIZAÇÃO (Etapa 8)
# =============================================================================
REG_FIG_DIR = FIGURES_DIR / 'regularizacao'
REG_MAX_EPOCHS_NO_STOPPING = 200   # a configuração A treina até aqui, sem early stopping
# Uma mudança por vez em relação a B; o resto vem de MLP_DEFAULTS (Adam, lr 1e-3, batch 64, paciência 20)
REG_CONFIGS = {
    'A. Sem regularização': dict(hidden=(128, 64), dropout=0.0, weight_decay=0.0, early_stopping=False,
                                 max_epochs=REG_MAX_EPOCHS_NO_STOPPING),
    'B. Early stopping': dict(hidden=(128, 64), dropout=0.0, weight_decay=0.0, early_stopping=True),
    'C. Dropout 0,3': dict(hidden=(128, 64), dropout=0.3, weight_decay=0.0, early_stopping=True),
    'D. L2 1e-3': dict(hidden=(128, 64), dropout=0.0, weight_decay=1e-3, early_stopping=True),
    'E. Rede menor (32)': dict(hidden=(32,), dropout=0.0, weight_decay=0.0, early_stopping=True),
}

# =============================================================================
# HOTSPOTS PREVISTOS (Etapa 9)
# =============================================================================
HOTSPOT_FIG_DIR = FIGURES_DIR / 'hotspots'
HOTSPOT_TOP_FRACTIONS = [0.1, 0.2, 0.3]   # fração inspecionada nas tabelas de captura
HOTSPOT_MAP_FRACTION = 0.2                # segmentos destacados no mapa
