///////////////////////////////////////////////////////////////////////////////////////////////
// Author: Ivan Zvonkov (izvonkov@umd.edu)
// Last Edited: Apr 2, 2025
// Description
//  (1) Specify ROI
//  (2) Create CropHarvest composite based on presto-v3
//  https://github.com/nasaharvest/presto-v3/tree/ff17611ef1433eff0b020f8e513c640c3959e381/src/data/earthengine
//  (3) Use Presto deployed on VertexAI to create embeddings
//  (4) Save embeddings as asset
///////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////
// 1. Specifies ROI
///////////////////////////////////////////////////////////////////////////////////////////////
var lon1 = 1.1575584013473583;
var lon2 = 1.2048954756271435;
var lat1 = 6.840427147744777;
var lat2 = 6.877467188902948;
var roi = ee.Geometry.Polygon([
    [lon1, lat1],
    [lon2, lat1],
    [lon2, lat2],
    [lon1, lat2],
    [lon1, lat1],
]);
Map.centerObject(roi, 14);

///////////////////////////////////////////////////////////////////////////////////////////////
// 2. Creates CropHarvest Composite
///////////////////////////////////////////////////////////////////////////////////////////////
var rangeStart = '2019-03-01';
var rangeEnd = '2020-03-01';

///////////////////////////////////////////////////////////////////////////////////////////////
// 2a. Sentinel-1 Data
///////////////////////////////////////////////////////////////////////////////////////////////
var S1_BANDS = ['VV', 'VH'];
var S1_all = ee
    .ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(
        ee.Date(rangeStart).advance(-31, 'days'),
        ee.Date(rangeEnd).advance(31, 'days')
    );

var S1 = S1_all.filter(
    ee.Filter.eq(
        'orbitProperties_pass',
        S1_all.first().get('orbitProperties_pass')
    )
).filter(ee.Filter.eq('instrumentMode', 'IW'));
var S1_VV = S1.filter(
    ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')
);
var S1_VH = S1.filter(
    ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')
);

function getCloseImages(middleDate, imageCollection) {
    var fromMiddleDate = imageCollection
        .map(function (img) {
            var dateDist = ee
                .Number(img.get('system:time_start'))
                .subtract(middleDate.millis())
                .abs();
            return img.set('dateDist', dateDist);
        })
        .sort({ property: 'dateDist', ascending: true });
    var fifteenDaysInMs = ee.Number(1296000000);
    var maxDiff = ee
        .Number(fromMiddleDate.first().get('dateDist'))
        .max(fifteenDaysInMs);
    return fromMiddleDate.filterMetadata(
        'dateDist',
        'not_greater_than',
        maxDiff
    );
}

function S1_img(date1, date2) {
    var startDate = ee.Date(date1);
    var daysBetween = ee.Date(date2).difference(startDate, 'days');
    var middleDate = startDate.advance(daysBetween.divide(2), 'days');
    var kept_vv = getCloseImages(middleDate, S1_VV).select('VV');
    var kept_vh = getCloseImages(middleDate, S1_VH).select('VH');
    var S1_composite = ee.Image.cat([kept_vv.median(), kept_vh.median()]);
    return S1_composite.select(S1_BANDS).add(25.0).divide(25.0); // S1 ranges from -50 to 1
}

///////////////////////////////////////////////////////////////////////////////////////////////
// 2b. Sentinel-2 Data
///////////////////////////////////////////////////////////////////////////////////////////////
var S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];
var S2 = ee
    .ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(rangeStart, rangeEnd);
var csPlus = ee
    .ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    .filterBounds(roi)
    .filterDate(rangeStart, rangeEnd);
var QA_BAND = 'cs_cdf'; // Better than cs here
var S2_cf = S2.linkCollection(csPlus, [QA_BAND]);

function S2_img(date1, date2) {
    return S2_cf.filterDate(date1, date2)
        .qualityMosaic(QA_BAND)
        .select(S2_BANDS)
        .divide(ee.Image(1e4));
}

///////////////////////////////////////////////////////////////////////////////////////////////
// 2c. ERA5 Data
///////////////////////////////////////////////////////////////////////////////////////////////
var ERA5_BANDS = ['temperature_2m', 'total_precipitation_sum'];
var ERA5 = ee
    .ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    .filterBounds(roi)
    .filterDate(rangeStart, rangeEnd);
function ERA5_img(date1, date2) {
    return ERA5.filterDate(date1, date2)
        .select(ERA5_BANDS)
        .mean()
        .add([-272.15, 0])
        .divide([35, 0.03]);
}

///////////////////////////////////////////////////////////////////////////////////////////////
// 2d. SRTM Data
///////////////////////////////////////////////////////////////////////////////////////////////
var SRTM_BANDS = ['elevation', 'slope'];
var elevation = ee.Image('USGS/SRTMGL1_003').clip(roi).select('elevation');
var slope = ee.Terrain.slope(elevation);
var SRTM_img = ee.Image.cat([elevation, slope]).toDouble().divide([2000, 50]);

function cropharvest_img(d1, d2) {
    var img = ee.Image.cat([
        S1_img(d1, d2),
        S2_img(d1, d2),
        ERA5_img(d1, d2),
        SRTM_img,
    ]);
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    return img.addBands(ndvi).clip(roi).toFloat(); // toFloat Necessary for tensor conversion
}

var latlons = ee.Image.pixelLonLat().clip(roi).select('latitude', 'longitude');
var imgs = [
    latlons,
    cropharvest_img('2019-03-01', '2019-04-01'),
    cropharvest_img('2019-04-01', '2019-05-01'),
    cropharvest_img('2019-05-01', '2019-06-01'),
    cropharvest_img('2019-06-01', '2019-07-01'),
    cropharvest_img('2019-07-01', '2019-08-01'),
    cropharvest_img('2019-08-01', '2019-09-01'),
    cropharvest_img('2019-09-01', '2019-10-01'),
    cropharvest_img('2019-10-01', '2019-11-01'),
    cropharvest_img('2019-11-01', '2019-12-01'),
    cropharvest_img('2019-12-01', '2020-01-01'),
    cropharvest_img('2020-01-01', '2020-02-01'),
    cropharvest_img('2020-02-01', '2020-03-01'),
];

var S1vis = {
    bands: ['VV', 'VH', 'VV'],
    min: [0, -0.2, 0.4],
    max: [1.0, 0.8, 1.2],
};
Map.addLayer(imgs[1], S1vis, 'S1 March');
Map.addLayer(imgs[2], S1vis, 'S1 April');
Map.addLayer(imgs[3], S1vis, 'S1 May');

var S2vis = { bands: ['B4', 'B3', 'B2'], min: 0, max: 0.25 };
Map.addLayer(imgs[1], S2vis, 'S2 March');
Map.addLayer(imgs[2], S2vis, 'S2 April');
Map.addLayer(imgs[3], S2vis, 'S2 May');

var ERA5Palette = [
    '000080',
    '0000d9',
    '4000ff',
    '8000ff',
    '0080ff',
    '00ffff',
    '00ff80',
    '80ff00',
    'daff00',
    'ffff00',
    'fff500',
    'ffda00',
    'ffb000',
    'ffa400',
    'ff4f00',
    'ff2500',
    'ff0a00',
    'ff00ff',
];
var ERA5vis = {
    bands: ['temperature_2m'],
    min: 0,
    max: 1,
    palette: ERA5Palette,
};
Map.addLayer(imgs[1], ERA5vis, 'ERA5 March');
Map.addLayer(imgs[2], ERA5vis, 'ERA5 April');
Map.addLayer(imgs[3], ERA5vis, 'ERA5 May');

Map.addLayer(imgs[1], { bands: ['slope'], min: 0, max: 1 }, 'SRTM slope');

var composite = ee.ImageCollection.fromImages(imgs).toBands();
var bands = composite.bandNames();
print(bands);

///////////////////////////////////////////////////////////////////////////////////////////////
// 3. Call Vertex AI Endpoint
///////////////////////////////////////////////////////////////////////////////////////////////
var endpoint =
    'projects/presto-deployment/locations/us-central1/endpoints/vertex-pytorch-presto-endpoint';
var vertex_model = ee.Model.fromVertexAi({
    endpoint: endpoint,
    inputTileSize: [1, 1],
    proj: ee.Projection('EPSG:4326').atScale(10),
    fixInputProj: true,
    outputTileSize: [1, 1],
    outputBands: {
        p: {
            // 'output'
            type: ee.PixelType.float(),
            dimensions: 1,
        },
    },
    payloadFormat: 'ND_ARRAYS',
});

var predictions = vertex_model.predictImage(composite).clip(roi);
print(predictions);

//Map.addLayer(predictions, {min: 0, max: 1},'predictions')

///////////////////////////////////////////////////////////////////////////////////////////////
// 4. Save embeddings
///////////////////////////////////////////////////////////////////////////////////////////////
Export.image.toAsset({
    image: predictions,
    description: 'Presto_embeddings',
    assetId: 'Togo/Presto_test_embeddings_v2025_04_10',
    region: roi,
    scale: 10,
    maxPixels: 1e12,
    crs: 'EPSG:25231',
});
