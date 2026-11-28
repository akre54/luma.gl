// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import {getSpatialOperationNodes} from '../utils/gpu-spatial-operation';

/** Packed views borrowed by a tangent-generation graph operation. */
export type GPUMeshTangentsProps = {
  /** Optional command node ID. */
  id?: string;
  /** One object-space position per vertex. */
  positions: GraphDataView<'float32x3'>;
  /** One object-space normal per vertex. The operation normalizes each input normal. */
  normals: GraphDataView<'float32x3'>;
  /** One UV pair per vertex for the tangent frame's texture coordinate set. */
  textureCoordinates: GraphDataView<'float32x2'>;
  /** One glTF tangent per vertex. Zero marks a zero normal or conflicting handedness. */
  tangents: GraphDataView<'float32x4'>;
  /** Omit for an unindexed triangle list. */
  topology?: {
    /** Packed triangle-list indices, with three vertex IDs per face. */
    indices: GraphDataView<'uint32'>;
    /** CSR vertex-to-face incidence offsets, one per vertex plus a sentinel. */
    offsets: GraphDataView<'uint32'>;
    /** Packed incident face IDs, with each incident face listed once per vertex. */
    faceIndices: GraphDataView<'uint32'>;
  };
};

/** Computes tangents for a GPU-resident triangle mesh without changing topology. */
export class GPUMeshTangents {
  /** Command node ID. */
  readonly id: string;
  /** Mesh inputs, topology, and destination buffer. */
  readonly props: GPUMeshTangentsProps;

  /** Creates an operation for the supplied mesh views. */
  constructor(props: GPUMeshTangentsProps) {
    this.props = props;
    this.id = props.id ?? 'gpu-mesh-tangents';
    const {positions, normals, textureCoordinates, tangents, topology} = props;
    // All vertex attributes need the same row count.
    assert(positions.length === normals.length && positions.length === textureCoordinates.length);
    assert(positions.length === tangents.length);
    // Each triangle has three corners, and indexed meshes require CSR offsets.
    assert(
      topology
        ? topology.indices.length % 3 === 0 && topology.offsets.length === positions.length + 1
        : positions.length % 3 === 0
    );
  }

  /** Builds the WebGPU compute node using the graph’s borrowed storage views. */
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
    const {positions, normals, textureCoordinates, tangents, topology} = this.props;
    return getSpatialOperationNodes(graph, {
      id: this.id,
      length: positions.length,
      inputs: {positions, normals, textureCoordinates, ...topology},
      outputs: {tangents},
      body: `let normal = safeNormal(readNormals(index));
if (all(normal == vec3f(0.0))) {
  writeTangents(index, vec4f(0.0));
  return;
}
var tangentSum = vec3f(0.0);
var handedness = 0.0;
let begin = ${topology ? 'readOffsets(index)' : 'index / 3u'};
let end = ${topology ? `min(readOffsets(index + 1u), ${topology.faceIndices.length}u)` : 'begin + 1u'};
for (var entry = begin; entry < end; entry++) {
  ${
    topology
      ? `
  let face = readFaceIndices(entry);
  if (face >= ${topology.indices.length / 3}u) { continue; }
  let first = readIndices(face * 3u);
  let second = readIndices(face * 3u + 1u);
  let third = readIndices(face * 3u + 2u);
  if (max(first, max(second, third)) >= ${positions.length}u) { continue; }
  if (index != first && index != second && index != third) { continue; }`
      : `
  let first = entry * 3u;
  let second = first + 1u;
  let third = first + 2u;`
  }
  let firstEdge = readPositions(second) - readPositions(first);
  let secondEdge = readPositions(third) - readPositions(first);
  let firstUV = readTextureCoordinates(second) - readTextureCoordinates(first);
  let secondUV = readTextureCoordinates(third) - readTextureCoordinates(first);
  let determinant = firstUV.x * secondUV.y - secondUV.x * firstUV.y;
  let orientation = dot(cross(firstEdge, secondEdge), normal);
  if (determinant == 0.0 || orientation == 0.0) { continue; }
  let uvOrientation = sign(determinant);
  let tangent = safeNormal(firstEdge * secondUV.y - secondEdge * firstUV.y);
  if (all(tangent == vec3f(0.0))) { continue; }
  let signForFace = sign(orientation) * uvOrientation;
  if (handedness != 0.0 && handedness != signForFace) {
    writeTangents(index, vec4f(0.0));
    return;
  }
  handedness = signForFace;
  tangentSum += tangent * uvOrientation;
}
let projected = tangentSum - normal * dot(normal, tangentSum);
var tangent = safeNormal(projected);
if (length(projected) <= 1e-6 * length(tangentSum)) {
  let axis = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.x) < 0.9);
  tangent = safeNormal(cross(normal, axis));
}
writeTangents(index, vec4f(tangent, select(handedness, 1.0, handedness == 0.0)));`
    });
  }
}
