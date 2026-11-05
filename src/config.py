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

# Provisórios: a unidade espacial e temporal é fechada na Etapa 2
SEGMENT_SIZE_M = 500
AGG_PERIOD = 'M'

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
