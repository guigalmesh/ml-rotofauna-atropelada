// Exporta a cobertura e uso da terra do MapBiomas (2010–2015) recortada para o DF.
// Rodar em https://code.earthengine.google.com (conta Google + projeto não comercial).
// Resultado: um GeoTIFF multibanda (classification_2010 ... classification_2015)
// na pasta "rodofauna" do Google Drive. Baixar e salvar em data/external/raw/mapbiomas/.

// ID do asset: confira na página de downloads do MapBiomas (link "Google Earth Engine")
// e troque abaixo se a coleção for outra.
var ASSET = 'projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1';

var anos = ['2010', '2011', '2012', '2013', '2014', '2015'];
var bandas = anos.map(function (a) { return 'classification_' + a; });

// Limite do DF com margem de 2 km (pontos perto da divisa)
var df = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Brazil'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Distrito Federal'))
  .geometry()
  .buffer(2000);

var cobertura = ee.Image(ASSET).select(bandas).clip(df);

Map.centerObject(df, 9);
Map.addLayer(cobertura.select('classification_2015').randomVisualizer(), {}, 'MapBiomas 2015');
print('Bandas exportadas', cobertura.bandNames());

Export.image.toDrive({
  image: cobertura.toByte(),
  description: 'mapbiomas_df_2010_2015',
  folder: 'rodofauna',
  fileNamePrefix: 'mapbiomas_df_2010_2015',
  region: df,
  scale: 30,
  crs: 'EPSG:31983',
  maxPixels: 1e10
});
