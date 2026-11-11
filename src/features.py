"""
Atributos dos datasets processados (Etapa 4).

Tudo aqui é cálculo determinístico sobre dados crus: nenhuma função ajusta
nada aos dados (scaler, imputação, one-hot ficam nos experimentos, só no
treino). Regra de tempo: nenhum atributo usa informação posterior ao que se
prevê. O clima entra com o mês anterior, a cobertura do solo com o ano
anterior e os lags só com o passado do próprio segmento.
"""
import numpy as np
import pandas as pd
import geopandas as gpd

from src import config
from src.external import nearest_distance, points_gdf, sample_raster


# =============================================================================
# TEMPO
# =============================================================================
def cyclic(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Seno e cosseno de uma variável cíclica (mês, dia do ano)."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def month_features(month: pd.Series) -> pd.DataFrame:
    """mes_sin, mes_cos e seco (maio a setembro no DF) a partir do mês 1–12."""
    sin, cos = cyclic(month - 1, 12)
    return pd.DataFrame({'mes_sin': sin, 'mes_cos': cos, 'seco': month.between(5, 9).astype(int)})


def day_of_year_features(dates: pd.Series) -> pd.DataFrame:
    sin, cos = cyclic(dates.dt.dayofyear - 1, 365.25)
    return pd.DataFrame({'dia_ano_sin': sin, 'dia_ano_cos': cos})


# =============================================================================
# CLIMA: SEMPRE DO MÊS ANTERIOR
# =============================================================================
def previous_month_climate(clima: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela ano_mes → clima do mês anterior. As colunas do mês anterior vêm do
    01c, que desloca a série completa (de 2009 em diante) antes de recortar a
    janela das campanhas; por isso o primeiro mês também tem valor.
    """
    clima = clima.sort_values('ano_mes').reset_index(drop=True)
    out = pd.DataFrame({
        'ano_mes': clima['ano_mes'].astype(str),
        'prec_mes_anterior_mm': clima['precipitacao_mes_anterior_mm'],
        'temp_mes_anterior_c': clima['temperatura_mes_anterior_c'],
    })
    # Conferência de que o deslocamento é de um mês exato
    mes_anterior = (pd.PeriodIndex(clima['ano_mes'], freq='M') - 1).astype(str)
    esperado = clima.set_index(clima['ano_mes'].astype(str))['precipitacao_mm']
    confere = out['prec_mes_anterior_mm'].values[1:] == esperado.reindex(mes_anterior[1:]).values
    assert confere.all(), 'precipitacao_mes_anterior_mm não bate com a precipitação do mês anterior'
    return out


# =============================================================================
# ESPAÇO: SEGMENTO, RASTERS E HIDROGRAFIA
# =============================================================================
def assign_segment(df: pd.DataFrame, segmentos: gpd.GeoDataFrame, max_dist_m: float) -> pd.DataFrame:
    """Segmento mais próximo de cada registro (NaN além de max_dist_m) e a distância."""
    pontos = points_gdf(df)
    junto = gpd.sjoin_nearest(pontos[['geometry']], segmentos[['segmento_id', 'geometry']],
                              how='left', max_distance=max_dist_m, distance_col='dist_segmento_m')
    junto = junto[~junto.index.duplicated()]
    return df.assign(segmento_id=junto['segmento_id'], dist_segmento_m=junto['dist_segmento_m'])


def segment_uc(ocorrencias: pd.DataFrame, segmentos: gpd.GeoDataFrame) -> pd.Series:
    """
    UC predominante de cada segmento: moda das ocorrências atribuídas; para
    segmentos sem ocorrência, a UC da ocorrência mais próxima do centro.
    """
    moda = ocorrencias.dropna(subset=['segmento_id']).groupby('segmento_id')['uc'].agg(lambda s: s.mode().iloc[0])
    centros = gpd.GeoDataFrame(segmentos[['segmento_id']],
                               geometry=gpd.points_from_xy(segmentos['x_centro'], segmentos['y_centro']),
                               crs=segmentos.crs)
    vizinha = gpd.sjoin_nearest(centros, points_gdf(ocorrencias)[['uc', 'geometry']])
    vizinha = vizinha[~vizinha.index.duplicated()].set_index('segmento_id')['uc']
    uc = moda.reindex(segmentos['segmento_id'])
    return uc.fillna(vizinha).rename('uc')


def static_site_features(xs, ys, agua: gpd.GeoDataFrame, dem_path) -> pd.DataFrame:
    """Elevação e distância à água num conjunto de pontos (registros ou centros de segmentos)."""
    pontos = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=config.CRS)
    return pd.DataFrame({
        'elevacao_m': sample_raster(dem_path, xs, ys),
        'dist_agua_m': nearest_distance(pontos, agua).reindex(pontos.index).values,
    })


def vegetation_by_year(xs, ys, years, veg_path) -> np.ndarray:
    """
    % de vegetação nativa de cada ponto na banda do ano anterior ao informado
    (VEG_YEAR_LAG), limitada ao primeiro ano disponível. Rasters anuais usam
    imagens do ano inteiro, então o do próprio ano conteria meses futuros.
    """
    years = np.asarray(years)
    ano_raster = np.clip(years - config.VEG_YEAR_LAG, min(config.MAPBIOMAS_YEARS), max(config.MAPBIOMAS_YEARS))
    out = np.full(len(years), np.nan)
    for banda, ano in enumerate(config.MAPBIOMAS_YEARS, start=1):
        mask = ano_raster == ano
        if mask.any():
            out[mask] = sample_raster(veg_path, np.asarray(xs)[mask], np.asarray(ys)[mask], banda)
    return out


# =============================================================================
# PAINEL SEGMENTO × MÊS E LAGS
# =============================================================================
def build_panel(ocorrencias: pd.DataFrame, segmentos: gpd.GeoDataFrame, meses: list[str]) -> pd.DataFrame:
    """Uma linha por segmento × mês com a contagem n (zeros incluídos)."""
    com_segmento = ocorrencias.dropna(subset=['segmento_id'])
    contagem = com_segmento.groupby([com_segmento['segmento_id'].astype(int),
                                     com_segmento['data'].dt.to_period('M').astype(str)]).size()
    indice = pd.MultiIndex.from_product([segmentos['segmento_id'].astype(int), meses], names=['segmento_id', 'ano_mes'])
    painel = contagem.reindex(indice, fill_value=0).rename('n').reset_index()
    painel['ano'] = painel['ano_mes'].str[:4].astype(int)
    painel['mes'] = painel['ano_mes'].str[5:].astype(int)
    return painel


def add_history_features(painel: pd.DataFrame) -> pd.DataFrame:
    """
    Lags do alvo que usam só o passado do próprio segmento: n do mês anterior,
    média dos 12 meses anteriores (com o que houver), n do mesmo mês do ano
    anterior (NaN no primeiro ano) e quantos meses de histórico existem.
    """
    painel = painel.sort_values(['segmento_id', 'ano_mes']).reset_index(drop=True)
    g = painel.groupby('segmento_id')['n']
    painel['n_lag_1'] = g.shift(1)
    painel['n_media_12m'] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    painel['n_mesmo_mes_ano_anterior'] = g.shift(12)
    painel['meses_historico'] = painel.groupby('segmento_id').cumcount()
    return painel
