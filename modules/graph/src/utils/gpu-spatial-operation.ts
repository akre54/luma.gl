// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {getGPUVectorFormatInfo} from '../gpu-data/gpu-vector-format';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import {createChunkNode, validateChunkViews} from '../gpu-core/gpu-chunk-utils';
import {
  getViewElementOffset,
  getViewBindingRange,
  validatePackedView
} from '../gpu-core/graph-data-view-utils';
import {
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource
} from '../gpu-core/gpu-dispatch-utils';

/** Packed scalar and float-vector formats supported by spatial kernels. */
const SPATIAL_FORMATS = ['float32x2', 'float32x3', 'float32x4', 'uint32', 'float32'] as const;
/** A graph view with a format supported by the spatial WGSL accessors. */
type SpatialView = GraphDataView<(typeof SPATIAL_FORMATS)[number]>;

/** Shared packed spatial kernel plumbing. No allocation, submission, or ownership transfer. @internal */
export function getSpatialOperationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    length: number;
    inputs: Record<string, SpatialView>;
    outputs: Record<string, SpatialView>;
    atomicOutputs?: boolean;
    body: string;
  }
) {
  validateChunkViews(graph, Object.values(props.inputs), Object.values(props.outputs));
  const entries = [...Object.entries(props.inputs), ...Object.entries(props.outputs)];
  for (const [, view] of entries) validatePackedView(view, SPATIAL_FORMATS, props.id);
  if (!props.length) return [];
  const outputs = Object.values(props.outputs);
  for (let first = 0; first < outputs.length; first++) {
    for (let second = first + 1; second < outputs.length; second++) {
      if (outputs[first].buffer !== outputs[second].buffer) continue;
      const firstRange = getViewBindingRange(outputs[first]);
      const secondRange = getViewBindingRange(outputs[second]);
      // WebGPU writable bindings must remain disjoint after storage-offset alignment.
      assert(
        firstRange.offset + firstRange.size <= secondRange.offset ||
          secondRange.offset + secondRange.size <= firstRange.offset
      );
    }
  }
  const dispatch = getBoundedDispatchLayout(
    props.id,
    props.length,
    256,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const declarations = entries
    .map(([name, view], binding) => {
      const output = binding >= Object.keys(props.inputs).length;
      const atomic = output && props.atomicOutputs;
      // Atomic outputs are uint32 counters and representative IDs only.
      assert(!atomic || view.format === 'uint32');
      const scalar = view.format === 'uint32' ? 'u32' : 'f32';
      const components = getGPUVectorFormatInfo(view.format).components;
      const valueType = components === 1 ? scalar : `vec${components}f`;
      const address = `${getViewElementOffset(view)}u + row * ${components}u`;
      const title = name[0].toUpperCase() + name.slice(1);
      const elements = Array.from(
        {length: components},
        (_, component) => `${name}[offset + ${component}u]`
      );
      const read = atomic
        ? `atomicLoad(&${name}[offset])`
        : components === 1
          ? elements[0]
          : `${valueType}(${elements.join(', ')})`;
      const write = atomic
        ? `atomicStore(&${name}[offset], value);`
        : components === 1
          ? `${elements[0]} = value;`
          : elements.map((element, component) => `${element} = value[${component}];`).join(' ');
      return `@group(0) @binding(${binding}) var<storage, ${output ? 'read_write' : 'read'}> ${name}: array<${atomic ? 'atomic<u32>' : scalar}>;
fn read${title}(row: u32) -> ${valueType} { let offset = ${address}; return ${read}; }
${output ? `fn write${title}(row: u32, value: ${valueType}) { let offset = ${address}; ${write} }` : ''}
${
  atomic
    ? `fn add${title}(row: u32, value: u32) { atomicAdd(&${name}[${address}], value); }
fn min${title}(row: u32, value: u32) { atomicMin(&${name}[${address}], value); }`
    : ''
}`;
    })
    .join('\n');
  return [
    createChunkNode(graph, {
      ...props,
      dispatch,
      source: `${declarations}
fn safeNormal(value: vec3f) -> vec3f {
  let scale = max(max(abs(value.x), abs(value.y)), abs(value.z));
  if (!(scale > 0.0) || scale > 3.402823e38) { return vec3f(0.0); }
  let scaled = value / scale;
  let magnitude = length(scaled);
  if (!(magnitude > 0.0)) { return vec3f(0.0); }
  return scaled / magnitude;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getBoundedInvocationIndexSource(dispatch, 256)}
  if (index >= ${props.length}u) { return; }
  ${props.body}
}`
    })
  ];
}
