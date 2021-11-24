import { expect } from 'chai';
import GeoJSONWriter from 'jsts/org/locationtech/jts/io/GeoJSONWriter.js';
import WKTReader from 'jsts/org/locationtech/jts/io/WKTReader.js';
import 'mocha';
import LocalWebServer from 'local-web-server';

import { readFileSync, writeFileSync } from 'fs';
import { TextDecoder, TextEncoder } from 'util';

import fetch from 'node-fetch';

global['fetch'] = fetch as unknown as (
    input: RequestInfo,
    init?: RequestInit
) => Promise<Response>;
global['TextDecoder'] = TextDecoder;
global['TextEncoder'] = TextEncoder;

import { arrayToStream, takeAsync } from './streams/utils.js';

import { deserialize, serialize } from './geojson.js';
import { IGeoJsonFeature } from './geojson/feature.js';
import { Rect } from './packedrtree.js';
import { IGeoJsonFeatureCollection } from './geojson/featurecollection.js';
import HeaderMeta from './HeaderMeta.js';

describe('geojson module', () => {
    describe('Writing file', () => {
        it('stuff', () => {
            const expected = {
                type: 'FeatureCollection',
                features: [
                    {
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [0, 0],
                        },
                    },
                ],
            };

            // console.log("Input GeoJSON:")
            // console.log(JSON.stringify(expected, undefined, 1))

            // const flatgeobuf = serialize(expected)
            // console.log(`Serialized input GeoJson into FlatGeobuf (${flatgeobuf.length} bytes)`)

            // console.log('writing to output /tmp/node.fgb');
            // writeFileSync('/tmp/node.fgb', Buffer.from(flatgeobuf));

            // const actual = deserialize(flatgeobuf)

            // console.log('FlatGeobuf deserialized back into GeoJSON:')
            // console.log(JSON.stringify(actual, undefined, 1))

            const testFile = '/tmp/hetero.fgb';
            const geoq = readFileSync(testFile);
            console.log(testFile);
            // console.log(geoq);

            const geoqDeSer = deserialize(geoq);
            console.log(
                'geoq deserialized num features:',
                geoqDeSer['features'].length
            );
            geoqDeSer['features'].forEach((f) => {
                console.log('geom type:', f['geometry']['type']);
            });

            console.log(JSON.stringify(geoqDeSer));
        });
    });
});
