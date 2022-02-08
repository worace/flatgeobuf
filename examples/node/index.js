/* eslint-disable no-undef */
import { geojson } from 'flatgeobuf'
import fs from 'fs';

// const expected = {
//     type: 'FeatureCollection',
//     features: [{
//         type: 'Feature',
//         geometry: {
//             type: 'Point',
//             coordinates: [0, 0]
//         }
//     }]
// }

// console.log("Input GeoJSON:")
// console.log(JSON.stringify(expected, undefined, 1))

// const flatgeobuf = geojson.serialize(expected)
// console.log(`Serialized input GeoJson into FlatGeobuf (${flatgeobuf.length} bytes)`)

// const actual = geojson.deserialize(flatgeobuf)

// console.log('FlatGeobuf deserialized back into GeoJSON:')
// console.log(JSON.stringify(actual, undefined, 1))

const file = process.argv[2];
console.error("input file: ", file);

const geoq = fs.readFileSync(file);
const bbox = {
  minX: 8.8,
  minY: 47.2,
  maxX: 9.5,
  maxY: 55.3
};
const res = geojson.deserialize(geoq, bbox);
res['features'].forEach((f) => {
  if (f['properties'] === undefined) {
    f['properties'] = {};
  }
});

const geoqDeSer = JSON.stringify(res);
// console.log('geoq deserialized:');
// console.log(geoq);
console.log(geoqDeSer);


console.log('read features: ',res['features'].length);
