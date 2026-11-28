// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {ShaderLayout} from '@luma.gl/core';
import {Kernel} from '../execution/compute/kernel';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import {
  type GPUBoundedDispatchLayout,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource
} from '../gpu-core/gpu-dispatch-utils';
import {
  getViewBinding,
  getViewElementOffset,
  validatePackedUint32View
} from '../gpu-core/graph-data-view-utils';
import type {GPUGraphAlgorithmsDegree, GPUGraphAlgorithmsDegreeViews} from './gpu-graph-degree';

const GRAPH_DEGREE_WORKGROUP_SIZE = 256;

/** Adds one exact CSR degree pass using an explicit device dispatch limit. @internal */
export function addGPUGraphAlgorithmsDegreeToGraphWithDispatchLimit<Parameters>(
  degree: GPUGraphAlgorithmsDegree,
  commandGraph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): void {
  if (degree.topology.graph.vertexCount === 0) {
    return;
  }

  const adjacency =
    degree.direction === 'incoming' && degree.topology.graph.directed
      ? degree.topology.reverse!
      : degree.topology.forward;
  const offsets = commandGraph.importGPUVector(`${degree.id}-offsets`, adjacency.offsets).data[0];
  const output = commandGraph.importGPUVector(`${degree.id}-output`, degree.output).data[0];
  addDegreeViews(commandGraph, {id: degree.id, offsets, output}, maxComputeWorkgroupsPerDimension);
}

/** Shared lowering for imported topology and caller-supplied CSR views. @internal */
export function addDegreeViews<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  {
    id = 'gpu-graph-degree',
    offsets,
    output,
    adjacencyMask,
    vertexMask,
    neighbors
  }: GPUGraphAlgorithmsDegreeViews,
  maxComputeWorkgroupsPerDimension: number
): void {
  validatePackedUint32View(offsets, 'CSR offsets');
  validatePackedUint32View(output, 'Degree output');
  const vertexCount = output.length;
  if (offsets.length !== vertexCount + 1) {
    throw new Error('CSR offsets must contain one row start per vertex and a final sentinel');
  }
  if (vertexMask && (!neighbors || vertexMask.length !== vertexCount)) {
    throw new Error('Vertex masks require one flag per vertex and a CSR neighbor view');
  }
  if (adjacencyMask && neighbors && adjacencyMask.length !== neighbors.length) {
    throw new Error('Adjacency mask and neighbor views must cover the same slots');
  }
  const maskInputs = [
    ...(adjacencyMask ? [{name: 'adjacencyMask', view: adjacencyMask}] : []),
    ...(vertexMask && neighbors
      ? [
          {name: 'vertexMask', view: vertexMask},
          {name: 'neighbors', view: neighbors}
        ]
      : [])
  ];
  for (const {name, view} of maskInputs) {
    validatePackedUint32View(view, name);
  }
  if (vertexCount === 0) {
    return;
  }
  const slotCount = adjacencyMask?.length ?? neighbors?.length;
  const inputs = [
    {name: 'output', view: output},
    ...(slotCount === 0 ? [] : [{name: 'offsets', view: offsets}, ...maskInputs])
  ].map(({name, view}, location) => ({name, view, location, writable: name === 'output'}));
  const read = (name: string, view: GraphDataView<'uint32'>, index: string) =>
    `${name}[${getViewElementOffset(view)}u + ${index}]`;
  const rowStart = read('offsets', offsets, 'index');
  const rowEnd = read('offsets', offsets, 'index + 1u');
  const destination = read('output', output, 'index');
  let count = `${destination} = ${rowEnd} - ${rowStart};`;
  if (slotCount === 0) {
    count = `${destination} = 0u;`;
  } else if (slotCount !== undefined) {
    const rowFilter = vertexMask
      ? `if (${read('vertexMask', vertexMask, 'index')} == 0u) {
      ${destination} = 0u;
      return;
    }`
      : '';
    const slotFilter = adjacencyMask
      ? `if (${read('adjacencyMask', adjacencyMask, 'slot')} == 0u) { continue; }`
      : '';
    const neighborFilter =
      vertexMask && neighbors
        ? `
      let neighbor = ${read('neighbors', neighbors, 'slot')};
      if (neighbor >= VERTEX_COUNT || ${read('vertexMask', vertexMask, 'neighbor')} == 0u) { continue; }`
        : '';
    count = /* wgsl */ `
  ${rowFilter}
  var degree = 0u;
  let end = min(${rowEnd}, ${slotCount}u);
  for (var slot = ${rowStart}; slot < end; slot++) {
    ${slotFilter}
    ${neighborFilter}
    degree += 1u;
  }
  ${destination} = degree;`;
  }
  const dispatchLayout = getGPUGraphAlgorithmsDegreeDispatchLayout(
    vertexCount,
    maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${vertexCount}u;
${inputs
  .map(
    ({name, location, writable}) =>
      `@group(0) @binding(${location}) var<storage, ${writable ? 'read_write' : 'read'}> ${name}: array<u32>;`
  )
  .join('\n')}

@compute @workgroup_size(${GRAPH_DEGREE_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, GRAPH_DEGREE_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT) { return; }
  ${count}
}`;

  const bindings: ShaderLayout['bindings'] = inputs.map(({name, location, writable}) => ({
    name,
    group: 0,
    location,
    type: writable ? 'storage' : 'read-only-storage'
  }));
  commandGraph.addComputePass({
    id,
    resources: inputs.map(({view, writable}) => ({
      buffer: view,
      usage: writable ? 'storage-write' : 'storage-read'
    })),
    compile: ({device}) => {
      const kernel = new Kernel(device, {id, source, shaderLayout: {bindings}});
      return {
        encode: ({computePass, getBuffer}) => {
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries(
              inputs.map(({name, view}) => [name, getViewBinding(view, getBuffer)])
            ),
            x: dispatchLayout.x,
            y: dispatchLayout.y,
            z: dispatchLayout.z
          });
        },
        destroy: () => kernel.destroy()
      };
    }
  });
}

/** Plans a bounded three-dimensional dispatch for exact vertex degree extraction. @internal */
export function getGPUGraphAlgorithmsDegreeDispatchLayout(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout(
    'GPUGraphAlgorithmsDegree',
    elementCount,
    GRAPH_DEGREE_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
}
