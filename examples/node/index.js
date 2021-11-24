/* eslint-disable no-undef */
// eslint-disable-next-line @typescript-eslint/no-var-requires
const geojson = require("flatgeobuf/lib/cjs/geojson")
const fs = require('fs');

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

// console.log('writing to output /tmp/node.fgb');
// fs.writeFile('/tmp/node.fgb', Buffer.from(flatgeobuf),(err) => {
//   if (err) throw err;
//   console.log('The file has been saved!');
// });


// const actual = geojson.deserialize(flatgeobuf)

// console.log('FlatGeobuf deserialized back into GeoJSON:')
// console.log(JSON.stringify(actual, undefined, 1))

const file = process.argv[2];
console.error(file);

const geoq = fs.readFileSync(file);
const res = geojson.deserialize(geoq);
res['features'].forEach((f) => {
  if (f['properties'] === undefined) {
    f['properties'] = {};
  }
});

const geoqDeSer = JSON.stringify(res);
// console.log('geoq deserialized:');
// console.log(geoq);
console.log(geoqDeSer);
