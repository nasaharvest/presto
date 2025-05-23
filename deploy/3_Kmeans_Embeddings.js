//------------------------------------------------------------------------------------
// Script for checking the embedding salience through clutering
// Author: Ivan Zvonkov (izvonkov@umd.edu)
//------------------------------------------------------------------------------------

// 1. Load embeddings
var embeddings = ee.Image(
    'users/izvonkov/Togo/Presto_test_embeddings_v2025_05_16'
);
var roi = embeddings.geometry({ geodesics: true });
Map.centerObject(roi, 11);

// 2. Cluster embeddings and display
var training = embeddings.sample({ region: roi, scale: 10, numPixels: 10000 });
var trainedClusterer = ee.Clusterer.wekaKMeans(7).train(training);
var result = embeddings.cluster(trainedClusterer);
Map.addLayer(result.randomVisualizer(), {}, 'clusters');

// 3. Display WorldCover
var WorldCover = ee.ImageCollection('ESA/WorldCover/v200').first().clip(roi);
var vis = { bands: ['Map'] };
Map.addLayer(WorldCover, vis, 'WorldCover');
