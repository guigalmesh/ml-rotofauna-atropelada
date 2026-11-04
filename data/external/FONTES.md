# Fontes dos dados externos

Todas as camadas espaciais foram reprojetadas para SIRGAS 2000 / UTM 23S (EPSG:31983), o mesmo sistema das coordenadas do IBRAM. O script que baixa e processa tudo é `experiments/01c_external.py` (funções em `src/external.py`).

| Fonte | O que traz | URL | Acesso | Resolução / escala | Licença | Bruto → processado |
| --- | --- | --- | --- | --- | --- | --- |
| OpenStreetMap (Overpass API, via `osmnx`) | Malha viária de carros: tipo de via, `maxspeed`, `lanes`, `ref` | https://overpass-api.de | 2026-11-04 | Vetorial (escala de mapeamento colaborativo) | ODbL 1.0, © colaboradores do OpenStreetMap | `raw/osm/vias_area.gpkg` → `processed/segmentos_osm.{gpkg,csv}` |
| OpenStreetMap | Hidrografia: `waterway` (river, stream, canal) e `natural=water` | https://overpass-api.de | 2026-11-04 | Vetorial | ODbL 1.0 | → `processed/hidrografia.gpkg` |
| INMET, estação automática Brasília A001 | Precipitação horária e temperatura do ar, agregadas por mês | https://portal.inmet.gov.br/dadoshistoricos | 2026-11-04 (baixado à mão) | Pontual, horária (lat −15,789, lon −47,926) | Dados públicos do governo federal | `raw/inmet/*.CSV` → `processed/clima_mensal.csv` |
| NASA POWER (API mensal, comunidade AG) | Precipitação corrigida (`PRECTOTCORR`) e temperatura a 2 m (`T2M`) | https://power.larc.nasa.gov | 2026-11-04 | Grade 0,5° × 0,625° (MERRA-2), no ponto da estação A001 | Uso livre, com citação ao NASA POWER Project | `raw/nasa_power/nasa_power_mensal.csv` → `processed/clima_mensal.csv` |
| SRTM (AWS Terrain Tiles / Mapzen) | Elevação do terreno | https://registry.opendata.aws/terrain-tiles/ | 2026-11-04 | Tiles GeoTIFF zoom 11 (38 m em Web Mercator; ~37 m após reprojetar). O SRTM original tem 30 m. Valores fora de 600–1.400 m (artefatos) viram sem dado | Uso livre; SRTM é domínio público (NASA/USGS) | `raw/srtm/*.tif` → `processed/elevacao.tif` |
| MapBiomas Brasil, Coleção 9 (asset `projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1`) | Cobertura e uso da terra anual, 2010–2015 | https://brasil.mapbiomas.org (exportado pelo Google Earth Engine com `raw/mapbiomas/export_mapbiomas_df.js`) | 2026-11-04 | 30 m; recorte do DF + 2 km; EPSG:31983 | CC BY 4.0 | `raw/mapbiomas/mapbiomas_df_2010_2015.tif` → `processed/vegetacao_nativa_500m.tif` |

## Observações
- **Clima.** A série final (`precipitacao_mm`, `temperatura_c`) usa o INMET nos meses com pelo menos 90% das horas válidas e o NASA POWER nos demais; a coluna `fonte_clima` registra a origem de cada mês. Com os CSVs de 2010 a 2015, **os 60 meses de abril/2010 a março/2015 vêm do INMET**; o NASA POWER fica só como verificação (correlação 0,95 na precipitação mensal, mas cerca de 18% menos chuva no total).
- **MapBiomas.** Exportado pelo Earth Engine (o download direto do raster não está disponível na plataforma). Foi usada a **Coleção 9**, a do ID que estava no script; a plataforma mostra a Coleção 11 como a mais recente. Vegetação nativa = classes 3, 4, 5, 6, 11, 12, 13, 29, 32, 49 e 50 da legenda (no DF aparecem 3, 4, 11 e 12). O produto processado guarda, em cada pixel e para cada ano, a % de vegetação nativa num círculo de 500 m (uint8, 255 = sem dado).
- **OSM.** O cache do `osmnx` (`raw/osm/cache/`) e o `vias_area.gpkg` somam ~50 MB; se não couberem no .zip de entrega, o próprio script baixa tudo de novo.
- **Datas de acesso.** O OSM é um retrato atual da malha, não a de 2010–2015; mudanças nas rodovias no período não aparecem.
