import numpy as np
import pandas as pd

from src.config import INTERIM_COLUMNS

DATE_FORMAT = '%d/%m/%y'
WEEKDAYS = {0: 'seg', 1: 'ter', 2: 'qua', 3: 'qui', 4: 'sex', 5: 'sab', 6: 'dom'}


def load_raw(path) -> pd.DataFrame:
    """Lê o CSV original (o arquivo vem com BOM, por isso 'utf-8-sig')."""
    return pd.read_csv(path, encoding='utf-8-sig')


def remove_arcgis_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove a coluna OBJECTID gerada pelo servidor GIS e elimina as
    linhas duplicadas (reduzindo de 10.328 para 5.164 ocorrências únicas).
    """
    df_unique = df.copy()
    if 'OBJECTID' in df_unique.columns:
        df_unique = df_unique.drop(columns=['OBJECTID'])

    return df_unique.drop_duplicates().reset_index(drop=True)


def standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza todas as colunas de texto para minúsculas e sem espaços nas bordas.
    Útil para rodar antes da análise geral (silvestres + domésticos).
    """
    df_std = df.copy()
    str_cols = df_std.select_dtypes(include=['object', 'str']).columns
    for col in str_cols:
        df_std[col] = df_std[col].str.strip().str.lower()
    return df_std


def filter_wild_animals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra apenas os registros referentes à fauna silvestre e faz
    verificação cruzada contra espécies domésticas conhecidas.
    """
    df_wild = standardize_strings(df)

    col_silvestre = 'silvestre_' if 'silvestre_' in df_wild.columns else 'silvestre'
    df_wild = df_wild[df_wild[col_silvestre] == 'silvestre'].copy()

    # Double-check taxonômico: espécies que a própria base marca como domésticas
    especies_domesticas = {
        'canis_familiaris', 'felis_catus', 'columba_livia', 'gallus_gallus_domesticus',
        'passer_domesticus', 'bos_taurus', 'anas_platyrhynchos', 'equus_ferus_caballus',
        'rattus_norvegicus', 'oryctolagus_cuniculus', 'sus_scrofa_domesticus', 'numida_meleagris',
    }
    if 'especie_nm' in df_wild.columns:
        df_wild = df_wild[~df_wild['especie_nm'].isin(especies_domesticas)]

    return df_wild.reset_index(drop=True)


def weekday_from_date(df: pd.DataFrame) -> pd.Series:
    """Dia da semana ('seg'...'dom') calculado a partir de data_dt."""
    datas = pd.to_datetime(df['data_dt'], format=DATE_FORMAT)
    return datas.dt.dayofweek.map(WEEKDAYS)


def fix_weekday_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcula dia_da_sem a partir de data_dt (corrige valores inválidos como 'eg_')."""
    df = df.copy()
    df['dia_da_sem'] = weekday_from_date(df)
    return df


def generate_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela com tipos, nulos (NaN), 'nao_identificado' e valores únicos por coluna."""
    total = len(df)
    nao_id_count = df.apply(
        lambda col: (col.astype(str).str.strip().str.lower() == 'nao_identificado').sum()
    )
    return pd.DataFrame({
        'Tipo': df.dtypes.astype(str),
        'Nulos (NaN)': df.isnull().sum(),
        '% Nulos': (df.isnull().sum() / total * 100).round(2),
        'nao_identificado': nao_id_count,
        '% nao_identificado': (nao_id_count / total * 100).round(2),
        'Valores Únicos': df.nunique()
    }).rename_axis('coluna')


def remove_repeated_records(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove registros idênticos em todas as colunas exceto 'id' (mesma campanha,
    data, espécie, ponto, hora, sexo e posição). Os ids desses pares são quase
    sempre consecutivos, o que indica digitação dupla. Devolve também as linhas removidas.
    """
    repetido = df.duplicated([c for c in df.columns if c != 'id'], keep='first')
    return df[~repetido].reset_index(drop=True), df[repetido]


def remove_unidentified_class(df: pd.DataFrame) -> pd.DataFrame:
    """Remove os registros sem classe taxonômica (alvo do problema multiclasse)."""
    return df[df['classe'] != 'nao_identificado'].reset_index(drop=True)


def impute_uc_nearest(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Atribui a UC dos registros com uc = 'nao_identificado' pela UC do registro
    identificado mais próximo (distância euclidiana nas coordenadas UTM).
    Devolve também, para cada registro imputado, a UC e a distância usadas.
    """
    df = df.copy()
    missing = df['uc'] == 'nao_identificado'
    known = df[~missing]
    known_xy = known[['x', 'y']].to_numpy(dtype=float)

    imputations = []
    for idx, row in df[missing].iterrows():
        dist = np.hypot(known_xy[:, 0] - row['x'], known_xy[:, 1] - row['y'])
        nearest = dist.argmin()
        uc = known['uc'].iloc[nearest]
        df.loc[idx, 'uc'] = uc
        imputations.append({'x': row['x'], 'y': row['y'], 'uc': uc, 'distancia_m': float(dist[nearest])})
    return df, imputations


def fix_spatially_inconsistent_uc(df: pd.DataFrame, k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corrige a UC de registros cujo rótulo contradiz a vizinhança: nenhum dos
    `k` vizinhos mais próximos tem a mesma UC e todos os `k` concordam numa
    outra. Esses pontos estão no circuito de outra UC (erro de rótulo).
    Devolve também a lista das correções.
    """
    df = df.copy()
    xy = df[['x', 'y']].to_numpy(dtype=float)
    ucs = df['uc'].to_numpy()
    correcoes = []
    for i in range(len(df)):
        dist = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        dist[i] = np.inf
        vizinhos = ucs[np.argsort(dist)[:k]]
        if (vizinhos != ucs[i]).all() and (vizinhos == vizinhos[0]).all():
            correcoes.append({'indice': i, 'x': xy[i, 0], 'y': xy[i, 1], 'uc_antes': ucs[i], 'uc_depois': vizinhos[0]})
    for c in correcoes:
        df.loc[c['indice'], 'uc'] = c['uc_depois']
    return df, pd.DataFrame(correcoes)


def select_interim_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte data_dt em 'data' e mantém só as colunas usadas adiante.
    'especie_nm' e 'campanha' ficam apenas como auxiliares (nunca atributos);
    'dia_da_sem' fica só para a EDA do calendário das campanhas.
    """
    df = df.copy()
    df['data'] = pd.to_datetime(df['data_dt'], format=DATE_FORMAT)
    return df[INTERIM_COLUMNS]


def build_campaign_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uma linha por campanha (esforço de campo), a partir da base sem duplicatas
    com silvestres e domésticos: o esforço independe do tipo de animal, e uma
    campanha só com domésticos ainda é um percurso feito. Campanha sem nenhum
    registro não aparece na base e por isso não entra aqui.
    """
    datas = pd.to_datetime(df['data_dt'], format=DATE_FORMAT)
    tabela = (df.assign(data=datas, silvestre=df['silvestre_'] == 'silvestre')
                .groupby('campanha')
                .agg(data=('data', 'min'), n_datas=('data', 'nunique'),
                     n_registros=('data', 'size'), n_silvestres=('silvestre', 'sum'))
                .reset_index())
    tabela['mes'] = tabela['data'].dt.to_period('M').astype(str)
    return tabela


def build_interim(raw_path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executa a limpeza raw → interim e devolve a base limpa, a tabela de
    decisões (etapa, linhas antes, linhas depois e motivo), a tabela de
    qualidade da base original sem duplicatas (todas as colunas, silvestres
    e domésticos), que a EDA usa por não ler o raw, e a tabela de campanhas
    (esforço de campo por campanha).
    """
    decisoes = []

    def registrar(etapa, antes, depois, motivo):
        decisoes.append({'etapa': etapa, 'linhas_antes': antes, 'linhas_depois': depois, 'motivo': motivo})

    df_raw = load_raw(raw_path)
    registrar('carregar_raw', len(df_raw), len(df_raw), 'CSV original')

    df = remove_arcgis_duplicates(df_raw)
    copias = df_raw.drop(columns='OBJECTID').value_counts()
    assert (copias == 2).all(), 'Esperava cada linha do raw exatamente 2 vezes (exportação do ArcGIS)'
    registrar('remover_duplicatas', len(df_raw), len(df),
              'Base veio duplicada do ArcGIS: cada linha aparecia exatamente 2 vezes; '
              'removida a coluna OBJECTID e as cópias')

    df = standardize_strings(df)
    registrar('padronizar_strings', len(df), len(df), 'Texto em minúsculas e sem espaços nas bordas')
    qualidade = generate_quality_table(df)
    campanhas = build_campaign_table(df)

    divergente = weekday_from_date(df) != df['dia_da_sem']
    n_div_silv = (divergente & (df['silvestre_'] == 'silvestre')).sum()
    df = fix_weekday_from_date(df)
    registrar('recalcular_dia_da_sem', len(df), len(df),
              f'dia_da_sem recalculado de data_dt: {divergente.sum()} divergências corrigidas '
              f'({n_div_silv} na fauna silvestre), incluindo valores inválidos como eg_')

    antes = len(df)
    df = filter_wild_animals(df)
    registrar('filtrar_silvestres', antes, len(df),
              f'Mantida só a fauna silvestre ({antes - len(df)} registros domésticos removidos)')

    antes = len(df)
    df, repetidos = remove_repeated_records(df)
    pares = ', '.join(str(i) for i in repetidos['id'].head(5))
    registrar('remover_registros_repetidos', antes, len(df),
              f'{len(repetidos)} registros idênticos a outro em todas as colunas exceto id '
              f'(mesma campanha, data, espécie, ponto, hora, sexo e posição), em geral com ids '
              f'consecutivos: provável digitação dupla. Ex.: ids removidos {pares}')

    antes = len(df)
    df = remove_unidentified_class(df)
    registrar('remover_classe_nao_identificada', antes, len(df), 'Registro sem classe (alvo do multiclasse)')

    df, imputacoes = impute_uc_nearest(df)
    detalhes = '; '.join(f"({i['x']}, {i['y']}) → {i['uc']} a {i['distancia_m']:.0f} m" for i in imputacoes)
    registrar('imputar_uc', len(df), len(df),
              f'{len(imputacoes)} registros com uc = nao_identificado receberam a UC do vizinho mais próximo: {detalhes}')

    df, correcoes = fix_spatially_inconsistent_uc(df)
    resumo = correcoes.groupby(['uc_antes', 'uc_depois']).size() if len(correcoes) else pd.Series(dtype=int)
    resumo = '; '.join(f'{a} → {d}: {n}' for (a, d), n in resumo.items())
    registrar('corrigir_uc_incoerente', len(df), len(df),
              f'{len(correcoes)} registros com UC que contradiz os 10 vizinhos mais próximos '
              f'(nenhum com a mesma UC, todos de outra) receberam a UC dos vizinhos: {resumo}')

    df = select_interim_columns(df)
    registrar('selecionar_colunas', len(df), len(df),
              'data_dt convertida em data; descartadas as colunas proibidas como preditores '
              '(campanha e especie_nm mantidas só como auxiliares)')

    return df, pd.DataFrame(decisoes), qualidade, campanhas
