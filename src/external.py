import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import MultiLineString, box
from shapely.ops import linemerge, substring

from src.config import CRS

MAIN_ROADS = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary'}

# Servidor principal e espelhos do Overpass (o principal recusa conexões em picos de uso)
OVERPASS_URLS = [
    'https://overpass-api.de/api',
    'https://overpass.private.coffee/api',
    'https://maps.mail.ru/osm/tools/overpass/api',
]


def _first(value):
    """OSM às vezes guarda listas em highway/maxspeed/lanes; fica o primeiro valor."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _to_number(value):
    """Extrai o número de maxspeed/lanes ('60', '60;80', ['2', '3'] → 60.0, 2.0)."""
    value = _first(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    digits = str(value).split(';')[0].strip().split(' ')[0]
    try:
        return float(digits)
    except ValueError:
        return np.nan


def points_gdf(df: pd.DataFrame, crs: str = CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['x'], df['y']), crs=crs)


def area_of_interest(points: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoSeries:
    """Retângulo que envolve os pontos, com margem, no CRS dos dados."""
    return gpd.GeoSeries([box(*points.total_bounds).buffer(buffer_m)], crs=points.crs)


# =============================================================================
# OPENSTREETMAP
# =============================================================================
def download_osm_roads(area: gpd.GeoSeries, cache_dir: Path) -> gpd.GeoDataFrame:
    """Baixa a malha viária de carros do OSM na área (com cache local do osmnx)."""
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    polygon = area.to_crs(4326).iloc[0]
    graph = ox.graph_from_polygon(polygon, network_type='drive', simplify=True, retain_all=True)
    graph = ox.convert.to_undirected(graph)
    edges = ox.graph_to_gdfs(graph, nodes=False).reset_index(drop=True)
    edges['highway'] = edges['highway'].map(_first)
    edges['ref'] = edges['ref'].map(lambda v: ';'.join(v) if isinstance(v, list) else v)
    edges['maxspeed'] = edges['maxspeed'].map(_to_number)
    edges['lanes'] = edges['lanes'].map(_to_number)
    return edges[['highway', 'ref', 'name', 'maxspeed', 'lanes', 'geometry']].to_crs(CRS)


def download_osm_water(area: gpd.GeoSeries, cache_dir: Path) -> gpd.GeoDataFrame:
    """Rios, córregos e corpos d'água do OSM na área (com cache local do osmnx)."""
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)

    tags = {'waterway': ['river', 'stream', 'canal'], 'natural': 'water'}
    polygon = area.to_crs(4326).iloc[0]
    for i, url in enumerate(OVERPASS_URLS):
        ox.settings.overpass_url = url
        try:
            water = ox.features_from_polygon(polygon, tags=tags).reset_index()
            break
        except requests.exceptions.RequestException:
            if i == len(OVERPASS_URLS) - 1:
                raise
    water = water[water.geometry.geom_type.isin(['LineString', 'MultiLineString', 'Polygon', 'MultiPolygon'])]
    water['tipo'] = water['waterway'].fillna(water['natural']) if 'waterway' in water else water['natural']
    return water[['tipo', 'geometry']].to_crs(CRS)


def nearest_distance(points: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> pd.Series:
    """Distância (m) de cada ponto à geometria mais próxima de `lines`."""
    joined = gpd.sjoin_nearest(points[['geometry']], lines[['geometry']], distance_col='dist')
    return joined[~joined.index.duplicated()]['dist']


def check_crs(df: pd.DataFrame, roads: gpd.GeoDataFrame, candidates: dict[str, str]) -> pd.DataFrame:
    """
    Compara hipóteses de CRS das coordenadas do IBRAM: a correta deixa os
    pontos em cima das vias do OSM (menor distância ponto → via).
    """
    rows = []
    for name, crs in candidates.items():
        pts = points_gdf(df, crs).to_crs(CRS)
        dist = nearest_distance(pts, roads)
        rows.append({
            'hipotese': name, 'epsg': crs,
            'dist_mediana_m': round(dist.median(), 1),
            'dist_p90_m': round(dist.quantile(0.9), 1),
            'pct_ate_20m': round((dist <= 20).mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def select_monitored_roads(roads: gpd.GeoDataFrame, points: gpd.GeoDataFrame,
                           min_records: int, buffer_m: float) -> gpd.GeoDataFrame:
    """
    Aproxima as rotas monitoradas: vias principais cuja rodovia (ref) recebeu
    pelo menos `min_records` registros, recortadas à faixa de `buffer_m` em
    torno dos registros (inclui trechos da rota sem registro, que são zeros).
    """
    main = roads[roads['highway'].isin(MAIN_ROADS) & roads['ref'].notna()].copy()
    nearest = gpd.sjoin_nearest(points[['geometry']], main[['ref', 'geometry']])
    nearest = nearest[~nearest.index.duplicated()]
    counts = nearest['ref'].value_counts()
    refs = counts[counts >= min_records].index

    monitored = main[main['ref'].isin(refs)]
    corridor = points.buffer(buffer_m).union_all()
    monitored = monitored.clip(corridor)
    return monitored[~monitored.geometry.is_empty].explode(index_parts=False).reset_index(drop=True)


def remove_parallel_carriageways(roads: gpd.GeoDataFrame, tol_m: float, min_piece_m: float = 20) -> list:
    """
    Pistas duplas (e rodovias que dividem a mesma pista, como BR-020 e DF-128)
    aparecem no OSM como linhas paralelas ou repetidas. Percorre as vias da mais
    longa para a mais curta e mantém só a parte de cada uma que fica a mais de
    `tol_m` das já mantidas; sobras menores que `min_piece_m` são descartadas.
    """
    kept, kept_area = [], None
    for geom in roads.geometry.iloc[roads.length.sort_values(ascending=False).index]:
        piece = geom if kept_area is None else geom.difference(kept_area)
        parts = [g for g in getattr(piece, 'geoms', [piece]) if not g.is_empty and g.length >= min_piece_m]
        if parts:
            kept.extend(parts)
            area = MultiLineString(parts).buffer(tol_m)
            kept_area = area if kept_area is None else kept_area.union(area)
    return kept


def split_into_segments(lines: list, size_m: float, min_m: float, max_gap_m: float) -> gpd.GeoDataFrame:
    """
    Une as linhas contínuas e corta cada uma em trechos iguais de ~`size_m`.
    Trechos menores que `min_m` (pontas e entroncamentos) são anexados ao
    trecho mais próximo, se estiver a até `max_gap_m`.
    """
    merged = linemerge(MultiLineString(lines))
    segments = []
    for line in getattr(merged, 'geoms', [merged]):
        n = max(1, round(line.length / size_m))
        step = line.length / n
        segments.extend(substring(line, i * step, (i + 1) * step) for i in range(n))

    while True:
        lengths = np.array([g.length for g in segments])
        short = [i for i in np.argsort(lengths) if lengths[i] < min_m]
        merged_any = False
        for i in short:
            dists = np.array([segments[i].distance(g) if j != i else np.inf for j, g in enumerate(segments)])
            j = int(dists.argmin())
            if dists[j] <= max_gap_m:
                union = segments[j].union(segments[i])
                segments[j] = linemerge(union) if union.geom_type == 'MultiLineString' else union
                del segments[i]
                merged_any = True
                break
        if not merged_any:
            break
    return gpd.GeoDataFrame(geometry=segments, crs=CRS)


def add_segment_attributes(segments: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Centro, comprimento e atributos da via (rodovia, tipo, velocidade, faixas)
    da via monitorada mais próxima do centro. O `segmento_id` segue a ordem
    (rodovia, x, y) do centro, para não depender da ordem em que o OSM devolve as vias.
    """
    seg = segments.copy()
    mids = gpd.GeoDataFrame(geometry=seg.geometry.interpolate(0.5, normalized=True), crs=seg.crs)
    attrs = gpd.sjoin_nearest(mids, roads[['ref', 'highway', 'maxspeed', 'lanes', 'geometry']])
    attrs = attrs[~attrs.index.duplicated()]
    seg['x_centro'] = mids.geometry.x.round(1)
    seg['y_centro'] = mids.geometry.y.round(1)
    seg['comprimento_m'] = seg.length.round(1)
    seg[['ref', 'tipo_via', 'velocidade_max', 'faixas']] = attrs[['ref', 'highway', 'maxspeed', 'lanes']].to_numpy()
    seg['velocidade_max'] = seg['velocidade_max'].astype(float)
    seg['faixas'] = seg['faixas'].astype(float)
    seg = seg.sort_values(['ref', 'x_centro', 'y_centro']).reset_index(drop=True)
    seg['segmento_id'] = np.arange(len(seg))
    cols = ['segmento_id', 'ref', 'tipo_via', 'velocidade_max', 'faixas', 'comprimento_m', 'x_centro', 'y_centro', 'geometry']
    return seg[cols]


# =============================================================================
# CLIMA
# =============================================================================
def load_inmet_monthly(folder: Path) -> pd.DataFrame:
    """
    Lê os CSVs horários da estação automática do INMET (Latin-1, ';', vírgula
    decimal, -9999 = ausente) e agrega por mês: soma da precipitação, média da
    temperatura e a fração de horas com dado válido.
    """
    frames = []
    for path in sorted(folder.glob('*.CSV')) + sorted(folder.glob('*.csv')):
        raw = pd.read_csv(path, sep=';', encoding='latin1', skiprows=8, decimal=',',
                          na_values=[-9999, '-9999'], usecols=[0, 2, 7])
        raw.columns = ['data', 'prec', 'temp']
        frames.append(raw)
    if not frames:
        return pd.DataFrame(columns=['ano_mes', 'prec_inmet_mm', 'temp_inmet_c', 'cobertura_inmet'])

    hourly = pd.concat(frames)
    hourly['ano_mes'] = pd.to_datetime(hourly['data'].str.replace('/', '-')).dt.to_period('M')
    monthly = hourly.groupby('ano_mes').agg(
        prec_inmet_mm=('prec', 'sum'), temp_inmet_c=('temp', 'mean'),
        horas_validas=('prec', 'count'), horas=('prec', 'size'),
    )
    monthly['cobertura_inmet'] = (monthly['horas_validas'] / monthly['horas']).round(3)
    return monthly[['prec_inmet_mm', 'temp_inmet_c', 'cobertura_inmet']].round(2).reset_index()


def download_nasa_power_monthly(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    """Precipitação (mm no mês) e temperatura média do NASA POWER num ponto."""
    url = 'https://power.larc.nasa.gov/api/temporal/monthly/point'
    params = {'parameters': 'PRECTOTCORR,T2M', 'community': 'AG', 'latitude': lat, 'longitude': lon,
              'start': start_year, 'end': end_year, 'format': 'JSON'}
    data = requests.get(url, params=params, timeout=120).json()['properties']['parameter']
    rows = []
    for key, prec_dia in data['PRECTOTCORR'].items():
        if key.endswith('13'):  # mês 13 = média anual
            continue
        periodo = pd.Period(f'{key[:4]}-{key[4:]}', freq='M')
        rows.append({'ano_mes': periodo, 'prec_nasa_mm': round(prec_dia * periodo.days_in_month, 1),
                     'temp_nasa_c': data['T2M'][key]})
    return pd.DataFrame(rows)


def combine_climate(nasa: pd.DataFrame, inmet: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    """
    Série final: INMET quando o mês tem cobertura >= `min_coverage`; caso
    contrário, NASA POWER. A coluna `fonte_clima` registra a origem de cada mês.
    """
    clima = nasa.merge(inmet, on='ano_mes', how='left')
    usa_inmet = clima['cobertura_inmet'].fillna(0) >= min_coverage
    clima['precipitacao_mm'] = np.where(usa_inmet, clima['prec_inmet_mm'], clima['prec_nasa_mm'])
    clima['temperatura_c'] = np.where(usa_inmet, clima['temp_inmet_c'], clima['temp_nasa_c'])
    clima['fonte_clima'] = np.where(usa_inmet, 'inmet_a001', 'nasa_power')
    clima = clima.sort_values('ano_mes').reset_index(drop=True)
    clima['precipitacao_mes_anterior_mm'] = clima['precipitacao_mm'].shift(1)
    clima['temperatura_mes_anterior_c'] = clima['temperatura_c'].shift(1)
    return clima


# =============================================================================
# ELEVAÇÃO (SRTM via AWS Terrain Tiles)
# =============================================================================
def _tile_index(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def download_elevation(area: gpd.GeoSeries, raw_dir: Path, out_path: Path,
                       valid_range: tuple[float, float], zoom: int = 11) -> Path:
    """
    Baixa os tiles GeoTIFF que cobrem a área, faz o mosaico e grava recortado no
    CRS dos dados. Valores fora de `valid_range` (artefatos do SRTM) viram sem dado.
    """
    import rasterio
    from rasterio.mask import mask
    from rasterio.merge import merge
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    raw_dir.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = area.to_crs(4326).total_bounds
    x0, y0 = _tile_index(lon_min, lat_max, zoom)
    x1, y1 = _tile_index(lon_max, lat_min, zoom)
    paths = []
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            path = raw_dir / f'{zoom}_{tx}_{ty}.tif'
            if not path.exists():
                url = f'https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{tx}/{ty}.tif'
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                path.write_bytes(response.content)
            paths.append(path)

    sources = [rasterio.open(p) for p in paths]
    mosaic, transform = merge(sources)
    src_crs = sources[0].crs
    for s in sources:
        s.close()

    height, width = mosaic.shape[1:]
    bounds = rasterio.transform.array_bounds(height, width, transform)
    dst_transform, dst_w, dst_h = calculate_default_transform(src_crs, CRS, width, height, *bounds)
    projected = np.zeros((1, dst_h, dst_w), dtype='float32')
    reproject(mosaic, projected, src_transform=transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=CRS, resampling=Resampling.bilinear)

    profile = {'driver': 'GTiff', 'height': dst_h, 'width': dst_w, 'count': 1, 'dtype': 'float32',
               'crs': CRS, 'transform': dst_transform, 'nodata': -9999, 'compress': 'deflate'}
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as tmp:
            tmp.write(projected)
            clipped, clip_transform = mask(tmp, [area.iloc[0]], crop=True, nodata=-9999)
    clipped[(clipped != -9999) & ((clipped < valid_range[0]) | (clipped > valid_range[1]))] = -9999
    profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=clip_transform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(clipped)
    return out_path


def sample_raster(path: Path, xs, ys, band: int = 1) -> np.ndarray:
    """Valor de uma banda do raster nas coordenadas (no CRS do raster)."""
    import rasterio

    with rasterio.open(path) as src:
        values = np.array([v[0] for v in src.sample(zip(xs, ys), indexes=band)], dtype=float)
        if src.nodata is not None:
            values[values == src.nodata] = np.nan
    return values


# =============================================================================
# USO DO SOLO (MapBiomas)
# =============================================================================
def native_vegetation_fraction(src_path: Path, out_path: Path, radius_m: float,
                               native_classes: list[int]) -> Path:
    """
    Para cada ano (banda) do MapBiomas, calcula em cada pixel a fração de
    vegetação nativa num círculo de `radius_m`. Pixels sem dado (classe 0,
    fora do recorte) não entram no denominador. Grava um GeoTIFF com uma
    banda por ano, em % inteira (0 a 100; 255 = sem dado).
    """
    import rasterio
    from scipy.signal import fftconvolve

    with rasterio.open(src_path) as src:
        classes = src.read()
        profile = src.profile
        names = src.descriptions
        pixel = src.res[0]

    r = int(round(radius_m / pixel))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = (xx ** 2 + yy ** 2 <= r ** 2).astype('float32')

    percent = np.full(classes.shape, 255, dtype='uint8')
    for b in range(classes.shape[0]):
        valid = (classes[b] != 0).astype('float32')
        native = np.isin(classes[b], native_classes).astype('float32')
        n_valid = np.rint(fftconvolve(valid, kernel, mode='same'))
        n_native = np.rint(fftconvolve(native, kernel, mode='same'))
        has_data = n_valid > 0
        percent[b][has_data] = np.rint(100 * n_native[has_data] / n_valid[has_data]).astype('uint8')

    profile.update(dtype='uint8', nodata=255, compress='deflate')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(percent)
        dst.descriptions = tuple(n.replace('classification', 'veg_nativa') for n in names)
    return out_path
