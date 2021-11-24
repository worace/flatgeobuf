import { GeometryType } from '../flat-geobuf/geometry-type.js';
import { Geometry } from '../flat-geobuf/geometry.js';

import {
    IParsedGeometry,
    flat,
    pairFlatCoordinates,
    toGeometryType,
} from '../generic/geometry';

export interface IGeoJsonGeometry {
    type: string;
    coordinates: number[] | number[][] | number[][][] | number[][][][];
    geometries?: IGeoJsonGeometry[];
}

export function parseGeometry(geometry: IGeoJsonGeometry): IParsedGeometry {
    const cs = geometry.coordinates;
    const xy: number[] = [];
    const z: number[] = [];
    let ends: number[] | undefined;
    let parts: IParsedGeometry[] | undefined;
    const type: GeometryType = toGeometryType(geometry.type);
    // console.log('parse feature geometry:', geometry.type);
    let end = 0;
    switch (geometry.type) {
        case 'Point':
            flat(cs, xy, z);
            break;
        case 'MultiPoint':
        case 'LineString':
            flat(cs as number[][], xy, z);
            break;
        case 'MultiLineString':
        case 'Polygon': {
            const css = cs as number[][][];
            flat(css, xy, z);
            if (css.length > 1) ends = css.map((c) => (end += c.length));
            break;
        }
        case 'MultiPolygon': {
            const csss = cs as number[][][][];
            const geometries = csss.map((coordinates) => ({
                type: 'Polygon',
                coordinates,
            }));
            parts = geometries.map(parseGeometry);
            break;
        }
        case 'GeometryCollection':
            if (geometry.geometries)
                parts = geometry.geometries.map(parseGeometry);
            break;
    }
    return {
        xy,
        z: z.length > 0 ? z : undefined,
        ends,
        type,
        parts,
    } as IParsedGeometry;
}

function extractParts(xy: Float64Array, z: Float64Array, ends: Uint32Array) {
    if (!ends || ends.length === 0) return [pairFlatCoordinates(xy, z)];
    let s = 0;
    const xySlices = Array.from(ends).map((e) => xy.slice(s, (s = e << 1)));
    let zSlices: Float64Array[];
    if (z) {
        s = 0;
        zSlices = Array.from(ends).map((e) => z.slice(s, (s = e)));
    }
    return xySlices.map((xy, i) =>
        pairFlatCoordinates(xy, zSlices ? zSlices[i] : undefined)
    );
}

function toGeoJsonCoordinates(geometry: Geometry, type: GeometryType) {
    // console.log('toGeoJsonCoordinates');
    const xy = geometry.xyArray() as Float64Array;
    const z = geometry.zArray() as Float64Array;
    switch (type) {
        case GeometryType.Point: {
            const a = Array.from(xy);
            if (z) a.push(z[0]);
            return a;
        }
        case GeometryType.MultiPoint:
        case GeometryType.LineString:
            return pairFlatCoordinates(xy, z);
        case GeometryType.MultiLineString:
            return extractParts(xy, z, geometry.endsArray() as Uint32Array);
        case GeometryType.Polygon:
            return extractParts(xy, z, geometry.endsArray() as Uint32Array);
    }
}

export function fromGeometry(geometry: Geometry): IGeoJsonGeometry {
    const type = geometry.type();
    console.log('geojson/geometry.ts fromGeometry');
    console.log('For geometry type:');
    console.log(type, '(', GeometryType[type], ')');
    if (type === GeometryType.GeometryCollection) {
        const geometries = [];
        for (let i = 0; i < geometry.partsLength(); i++) {
            const part = geometry.parts(i) as Geometry;
            const partType = part.type() as GeometryType;
            geometries.push(fromGeometry(part));
        }
        return {
            type: GeometryType[type],
            geometries,
        } as IGeoJsonGeometry;
    } else if (type === GeometryType.MultiPolygon) {
        const geometries = [];
        for (let i = 0; i < geometry.partsLength(); i++)
            geometries.push(fromGeometry(geometry.parts(i) as Geometry));
        return {
            type: GeometryType[type],
            coordinates: geometries.map((g) => g.coordinates),
        } as IGeoJsonGeometry;
    }
    console.log('reading non-parted geometry');
    const coordinates = toGeoJsonCoordinates(geometry, type);
    return {
        type: GeometryType[type],
        coordinates,
    } as IGeoJsonGeometry;
}
