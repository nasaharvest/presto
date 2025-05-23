//------------------------------------------------------------------------------------
// Script for generating Presto embeddings using Vertex AI
// Author: Ivan Zvonkov (izvonkov@umd.edu)
//------------------------------------------------------------------------------------
// 1. Presto embedding generation parameters (set parameters according to your needs)
//------------------------------------------------------------------------------------
var roi = ee
    .FeatureCollection('FAO/GAUL/2015/level2')
    .filter(ee.Filter.eq('ADM2_NAME', 'Haho'));
var PROJ = 'EPSG:25231';

var rangeStart = ee.Date('2019-03-01');
var rangeEnd = ee.Date('2020-03-01');

var ENDPOINT =
    'projects/presto-deployment/locations/us-central1/endpoints/vertex-pytorch-presto-endpoint';
var RUN_VERTEX_AI = true; // Leave this as false to get a cost estimate first
//------------------------------------------------------------------------------------

Map.centerObject(roi, 10);
Map.addLayer(roi, {}, 'Region of Interest');
Map.setOptions('satellite');

// 2. Cost Computation
var roiAreaKM2 = roi.geometry().area().divide(1e6);
function estimate(cost) {
    return roiAreaKM2.divide(1000).multiply(cost).toInt().getInfo();
}
print('ROI Area: ' + roiAreaKM2.toInt().getInfo() + ' km2');
print(
    'Embedding Generation Estimates\nCost: $' +
        estimate(5.37) +
        '-' +
        estimate(10.14)
);
if (!RUN_VERTEX_AI)
    print(
        'If you are ready to generate embeddings,\nchange RUN_VERTEX_AI variable to true'
    );

// 3. Obtain monthly Sentinel-1 composites
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

// 4. Obtain monthly Sentinel-2 composites
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

// 5. Obtain monthly ERA5 composites
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
//var ERA5_temp = ee.Image([0,0]).rename(ERA5_BANDS).clip(roi)

// 6. Obtain SRTM Data
var SRTM_BANDS = ['elevation', 'slope'];
var elevation = ee.Image('USGS/SRTMGL1_003').clip(roi).select('elevation');
var slope = ee.Terrain.slope(elevation);
var SRTM_img = ee.Image.cat([elevation, slope]).toDouble().divide([2000, 50]);
//var SRTM_temp = ee.Image([0,0]).rename(SRTM_BANDS).clip(roi)

// 7. Combine all data into a monthly CropHarvest-style monthly composite
function cropharvest_img(d1, d2) {
    var img = ee.Image.cat([
        S1_img(d1, d2),
        S2_img(d1, d2),
        ERA5_img(d1, d2),
        SRTM_img,
    ]);
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    // toFloat Necessary for tensor conversion
    return img.addBands(ndvi).clip(roi).toFloat();
}

// 8. Create and visualize Presto input
var latlons = ee.Image.pixelLonLat().clip(roi).select('latitude', 'longitude');
var imgs = [latlons];
var numMonths = rangeEnd.difference(rangeStart, 'month').toInt().getInfo();
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

for (var i = 0; i < numMonths; i++) {
    var monthStart = rangeStart.advance(i, 'month');
    var monthEnd = monthStart.advance(1, 'month');
    var img = cropharvest_img(monthStart, monthEnd);
    imgs.push(img);

    var monthName = monthStart.format('YY/MM').getInfo();
    Map.addLayer(
        img,
        {
            bands: ['VV', 'VH', 'VV'],
            min: [0, -0.2, 0.4],
            max: [1.0, 0.8, 1.2],
        },
        monthName + ' S1',
        false
    );
    Map.addLayer(
        img,
        { bands: ['B4', 'B3', 'B2'], min: 0, max: 0.25 },
        monthName + ' S2',
        false
    );
    Map.addLayer(
        img,
        { bands: ['temperature_2m'], min: 0, max: 1, palette: ERA5Palette },
        monthName + ' ERA5',
        false
    );
}
Map.addLayer(imgs[1], { bands: ['slope'], min: 0, max: 0.3 }, 'SRTM', false);

var composite = ee.ImageCollection.fromImages(imgs).toBands();

// 9. Make predictions using Presto on Vertex AI
var vertex_model = ee.Model.fromVertexAi({
    endpoint: ENDPOINT,
    inputTileSize: [1, 1],
    proj: ee.Projection('EPSG:4326').atScale(10),
    fixInputProj: true,
    outputTileSize: [1, 1],
    outputBands: { p: { type: ee.PixelType.float(), dimensions: 1 } },
    payloadFormat: 'ND_ARRAYS',
    maxPayloadBytes: 5242880, // 5.24mb [MAX]
});

if (RUN_VERTEX_AI) {
    // Create band names for embeddingsArrayImage
    var bandNames = [];
    for (var i = 0; i < 128; i++) {
        bandNames.push('b' + i + '');
    }

    // embeddingsArrayImage is a single band image where each pixel contains an array
    var embeddingsArrayImage = vertex_model.predictImage(composite).clip(roi);
    var embeddingsMultiBandImage = embeddingsArrayImage.arrayFlatten([
        bandNames,
    ]);

    // Only smaller size embeddings can be directly viewed in GEE immediatley larger ones require the batch task
    // Map.addLayer(embeddingsMultiBandImage, {min: 0, max: 1},'embeddingsMultiBandImage')

    Export.image.toAsset({
        image: embeddingsMultiBandImage,
        description: 'Presto_embeddings',
        assetId: 'Togo/Presto_test_embeddings_v2025_04_23',
        region: roi,
        scale: 10,
        maxPixels: 1e12,
        crs: 'EPSG:25231',
    });
}
