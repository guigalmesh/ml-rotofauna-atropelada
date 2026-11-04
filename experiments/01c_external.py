import geopandas as gpd
import pandas as pd
import requests

from src import config
from src.external import (
    add_segment_attributes, area_of_interest, check_crs, combine_climate,
    download_elevation, download_nasa_power_monthly, download_osm_roads,
    download_osm_water, load_inmet_monthly, native_vegetation_fraction,
    nearest_distance, points_gdf, remove_parallel_carriageways, sample_raster,
    select_monitored_roads, split_into_segments,
)

RAW = config.EXTERNAL_RAW_DIR
OUT = config.EXTERNAL_PROCESSED_DIR


def build_segments(vias, pontos):
    """Rotas monitoradas a partir dos registros e seus segmentos de ~500 m."""
    rotas = select_monitored_roads(vias, pontos, config.ROUTE_MIN_RECORDS, config.ROUTE_BUFFER_M)
    linhas = remove_parallel_carriageways(rotas, config.PARALLEL_TOL_M)
    segmentos = split_into_segments(linhas, config.SEGMENT_SIZE_M, config.SEGMENT_MIN_M, config.SEGMENT_MAX_GAP_M)
    return rotas, add_segment_attributes(segmentos, rotas)


def run_external():
    OUT.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ocorrencias = pd.read_csv(config.INTERIM_CSV)
    pontos = points_gdf(ocorrencias)
    area = area_of_interest(pontos, config.AOI_BUFFER_M)

    # =========================================================================
    # 1. MALHA VIÁRIA DO OSM E CONFIRMAÇÃO DO CRS
    # =========================================================================
    print("=" * 75)
    print("1. OPENSTREETMAP: MALHA VIÁRIA E CRS")
    print("=" * 75)
    vias_path = RAW / 'osm' / 'vias_area.gpkg'
    if vias_path.exists():
        vias = gpd.read_file(vias_path)
    else:
        vias = download_osm_roads(area, RAW / 'osm' / 'cache')
        vias.to_file(vias_path)
    print(f"Arestas da malha de carros na área: {len(vias)}")

    crs_check = check_crs(ocorrencias, vias, {
        'SIRGAS 2000 / UTM 23S': 'EPSG:31983',
        'SAD69 / UTM 23S': 'EPSG:29193',
        'WGS 84 / UTM 23S': 'EPSG:32723',
    })
    crs_check.to_csv(config.TABLES_DIR / 'verificacao_crs.csv', index=False)
    print(crs_check.to_string(index=False))

    # =========================================================================
    # 2. ROTAS MONITORADAS E SEGMENTOS DE 500 M
    # =========================================================================
    print("\n" + "=" * 75)
    print(f"2. SEGMENTOS DE {config.SEGMENT_SIZE_M} M DAS ROTAS MONITORADAS")
    print("=" * 75)
    rotas, segmentos = build_segments(vias, pontos)
    segmentos.to_file(OUT / 'segmentos_osm.gpkg')
    segmentos.drop(columns='geometry').to_csv(OUT / 'segmentos_osm.csv', index=False)

    dist = nearest_distance(pontos, segmentos)
    longe = dist > config.ASSIGN_MAX_DIST_M
    print(f"Rodovias monitoradas: {sorted(rotas['ref'].unique())}")
    print(f"Segmentos: {len(segmentos)}; {segmentos['comprimento_m'].sum() / 1000:.1f} km; comprimento "
          f"mediano {segmentos['comprimento_m'].median():.0f} m (mín {segmentos['comprimento_m'].min():.0f}, "
          f"máx {segmentos['comprimento_m'].max():.0f}); com < 250 m: {(segmentos['comprimento_m'] < 250).sum()}")
    print(f"Distância registro → segmento: mediana {dist.median():.1f} m; <= 50 m {(dist <= 50).mean() * 100:.1f}%; "
          f"<= 100 m {(dist <= 100).mean() * 100:.1f}%; máx {dist.max():.0f} m")
    print(f"Registros além de {config.ASSIGN_MAX_DIST_M} m (ficariam sem segmento): {longe.sum()}")
    print("Ausentes nos segmentos:")
    print((segmentos[['tipo_via', 'velocidade_max', 'faixas']].isna().mean() * 100).round(1).to_string())

    # Vazamento: as rotas saem dos registros. Refaz só com os anos de treino do
    # split temporal e mede o quanto a malha muda.
    treino = ocorrencias['ano'] <= config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR
    _, seg_treino = build_segments(vias, pontos[treino])
    cobertura = segmentos.geometry.intersection(seg_treino.buffer(config.PARALLEL_TOL_M).union_all()).length
    dist_teste = nearest_distance(pontos[~treino], seg_treino)
    verificacao = pd.DataFrame([{
        'segmentos_todos_anos': len(segmentos), 'km_todos_anos': round(segmentos.length.sum() / 1000, 1),
        'segmentos_so_treino': len(seg_treino), 'km_so_treino': round(seg_treino.length.sum() / 1000, 1),
        'pct_malha_coberta_pelo_treino': round(cobertura.sum() / segmentos.length.sum() * 100, 1),
        'registros_teste': int((~treino).sum()),
        'pct_teste_ate_dist_max': round((dist_teste <= config.ASSIGN_MAX_DIST_M).mean() * 100, 1),
    }])
    verificacao.to_csv(config.TABLES_DIR / 'verificacao_rotas_treino.csv', index=False)
    print("\nRotas com todos os anos × só anos de treino (<= "
          f"{config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR}):")
    print(verificacao.T.to_string(header=False))

    # =========================================================================
    # 3. CLIMA MENSAL (INMET A001, COMPLETADO PELO NASA POWER)
    # =========================================================================
    print("\n" + "=" * 75)
    print("3. CLIMA MENSAL")
    print("=" * 75)
    nasa_path = RAW / 'nasa_power' / 'nasa_power_mensal.csv'
    if nasa_path.exists():
        nasa = pd.read_csv(nasa_path)
        nasa['ano_mes'] = pd.PeriodIndex(nasa['ano_mes'], freq='M')
    else:
        nasa = download_nasa_power_monthly(*config.INMET_A001_LATLON, *config.CLIMATE_YEARS)
        nasa_path.parent.mkdir(parents=True, exist_ok=True)
        nasa.to_csv(nasa_path, index=False)
    inmet = load_inmet_monthly(RAW / 'inmet')
    clima = combine_climate(nasa, inmet, config.INMET_MIN_COVERAGE)
    inicio, fim = (pd.Period(m, 'M') for m in config.CLIMATE_WINDOW)
    clima = clima[(clima['ano_mes'] >= inicio) & (clima['ano_mes'] <= fim)]
    clima.to_csv(OUT / 'clima_mensal.csv', index=False)
    print(f"Meses na série: {len(clima)}")
    print(clima['fonte_clima'].value_counts().to_string())
    comparaveis = clima.dropna(subset=['prec_inmet_mm'])
    comparaveis = comparaveis[comparaveis['cobertura_inmet'] >= config.INMET_MIN_COVERAGE]
    if len(comparaveis) > 2:
        r = comparaveis['prec_inmet_mm'].corr(comparaveis['prec_nasa_mm'])
        print(f"Correlação INMET × NASA na precipitação ({len(comparaveis)} meses): {r:.2f}")

    # =========================================================================
    # 4. ELEVAÇÃO E HIDROGRAFIA
    # =========================================================================
    print("\n" + "=" * 75)
    print("4. ELEVAÇÃO (SRTM) E HIDROGRAFIA (OSM)")
    print("=" * 75)
    dem_path = download_elevation(area, RAW / 'srtm', OUT / 'elevacao.tif', config.ELEV_VALID_RANGE_M)
    elev = sample_raster(dem_path, ocorrencias['x'], ocorrencias['y'])
    print(f"Elevação nos registros: {pd.Series(elev).describe().round(1).to_dict()}")

    agua_path = OUT / 'hidrografia.gpkg'
    if agua_path.exists():
        agua = gpd.read_file(agua_path)
    else:
        try:
            agua = download_osm_water(area, RAW / 'osm' / 'cache')
            agua.to_file(agua_path)
        except requests.exceptions.RequestException as erro:
            agua = None
            print(f"[AVISO] Overpass indisponível, hidrografia não baixada; rode o script de novo. ({type(erro).__name__})")
    if agua is not None:
        dist_agua = nearest_distance(pontos, agua)
        print(f"Feições de água: {len(agua)}; distância dos registros à água: "
              f"mediana {dist_agua.median():.0f} m, máx {dist_agua.max():.0f} m")

    # =========================================================================
    # 5. VEGETAÇÃO NATIVA (MapBiomas)
    # =========================================================================
    print("\n" + "=" * 75)
    print(f"5. VEGETAÇÃO NATIVA NUM RAIO DE {config.VEG_RADIUS_M} M (MapBiomas)")
    print("=" * 75)
    mapbiomas_path = RAW / 'mapbiomas' / 'mapbiomas_df_2010_2015.tif'
    if not mapbiomas_path.exists():
        print(f"[AVISO] {mapbiomas_path.relative_to(config.ROOT)} não encontrado; ver data/external/FONTES.md.")
    else:
        veg_path = native_vegetation_fraction(mapbiomas_path, OUT / f'vegetacao_nativa_{config.VEG_RADIUS_M}m.tif',
                                              config.VEG_RADIUS_M, config.MAPBIOMAS_NATIVE_CLASSES)
        # Conferência: cada registro com a % do seu próprio ano
        veg = pd.Series(index=ocorrencias.index, dtype=float)
        for banda, ano in enumerate(config.MAPBIOMAS_YEARS, start=1):
            do_ano = ocorrencias['ano'] == ano
            veg[do_ano] = sample_raster(veg_path, ocorrencias.loc[do_ano, 'x'], ocorrencias.loc[do_ano, 'y'], banda)
        print(f"% de vegetação nativa nos registros: {veg.describe().round(3).to_dict()}")
        print("Mediana por UC:")
        print(veg.groupby(ocorrencias['uc']).median().round(3).to_string())

        seg_veg = [sample_raster(veg_path, segmentos['x_centro'], segmentos['y_centro'], b)
                   for b in range(1, len(config.MAPBIOMAS_YEARS) + 1)]
        variacao = pd.DataFrame(seg_veg, index=config.MAPBIOMAS_YEARS).T
        print(f"Segmentos: % média por ano {variacao.mean().round(3).to_dict()}")

    print(f"\n[OK] Saídas em {OUT.relative_to(config.ROOT)}/")


if __name__ == '__main__':
    run_external()
