// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import {type Binding} from '@luma.gl/core';
import {Kernel} from '../execution/compute/kernel';
import type {GPUCommandGraph, GraphBufferUse, GraphDataView} from '../gpu-core/gpu-command-graph';
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
import type {
  GPUGraphAlgorithmsConnectedComponents,
  GPUGraphAlgorithmsConnectedComponentsViews
} from './gpu-graph-connected-components';

const CONNECTED_COMPONENTS_WORKGROUP_SIZE = 256;
const INVALID_COMPONENT = 0xffffffff;

type ImportedConnectedComponents = {
  id: string;
  vertexCount: number;
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  overflow?: GraphDataView<'uint32'>;
  vertexMask?: GraphDataView<'uint32'>;
  adjacencyMask?: GraphDataView<'uint32'>;
  output: GraphDataView<'uint32'>;
  converged?: GraphDataView<'uint32'>;
  maxComputeWorkgroupsPerDimension: number;
};

type ConnectedComponentsBinding = {
  view: GraphDataView<'uint32'>;
  usage: GraphBufferUse['usage'];
  atomic?: boolean;
};

type ConnectedComponentsPassProps = {
  id: string;
  source: string;
  bindings: Record<string, ConnectedComponentsBinding>;
  dispatchLayout: GPUBoundedDispatchLayout;
};

/** Adds bounded GPU weak-component hooking using an explicit dispatch limit. @internal */
export function addGPUGraphAlgorithmsConnectedComponentsToGraphWithDispatchLimit<Parameters>(
  components: GPUGraphAlgorithmsConnectedComponents,
  commandGraph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): void {
  if (components.topology.graph.vertexCount === 0 && !components.converged) {
    return;
  }

  const state: ImportedConnectedComponents = {
    id: components.id,
    vertexCount: components.topology.graph.vertexCount,
    offsets: commandGraph.importGPUVector(
      `${components.id}-offsets`,
      components.topology.forward.offsets
    ).data[0],
    neighbors: commandGraph.importGPUVector(
      `${components.id}-neighbors`,
      components.topology.forward.neighbors
    ).data[0],
    overflow: commandGraph.importGPUVector(
      `${components.id}-overflow`,
      components.topology.forward.overflow
    ).data[0],
    output: commandGraph.importGPUVector(`${components.id}-output`, components.output).data[0],
    ...(components.converged
      ? {
          converged: commandGraph.importGPUVector(
            `${components.id}-converged`,
            components.converged
          ).data[0]
        }
      : {}),
    maxComputeWorkgroupsPerDimension
  };

  addConnectedComponentPasses(commandGraph, state, components.iterations);
}

/** Validates and lowers borrowed logical CSR views. @internal */
export function addConnectedComponentViews<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  {
    id = 'gpu-graph-connected-components',
    offsets,
    neighbors,
    output,
    converged,
    vertexMask,
    adjacencyMask
  }: GPUGraphAlgorithmsConnectedComponentsViews,
  iterations: number,
  maxComputeWorkgroupsPerDimension: number
): void {
  for (const [name, view, length] of [
    ['offsets', offsets, output.length + 1],
    ['neighbors', neighbors, neighbors.length],
    ['output', output, output.length],
    ['converged', converged, 1],
    ['vertexMask', vertexMask, output.length],
    ['adjacencyMask', adjacencyMask, neighbors.length]
  ] as const) {
    if (view) {
      validatePackedUint32View(view, name);
      if (view.length !== length) throw new Error(`${name} must contain ${length} rows`);
    }
  }
  addConnectedComponentPasses(
    graph,
    {
      id,
      vertexCount: output.length,
      offsets,
      neighbors,
      output,
      converged,
      vertexMask,
      adjacencyMask,
      maxComputeWorkgroupsPerDimension
    },
    iterations
  );
}

function addConnectedComponentPasses<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedConnectedComponents,
  iterations: number
): void {
  if (state.vertexCount === 0 && !state.converged) return;
  addInitializationPass(commandGraph, state);
  if (state.vertexCount === 0 || state.neighbors.length === 0) {
    return;
  }

  for (let iteration = 0; iteration < iterations; iteration++) {
    if (state.converged) {
      addConvergenceResetPass(commandGraph, {state, iteration});
    }
    addHookingPass(commandGraph, {state, iteration});
    addPointerJumpPass(commandGraph, {state, iteration});
  }
}

/** CSR views can detect truncated adjacency directly from the final offset. */
function getOverflowExpression(state: ImportedConnectedComponents): string {
  if (state.overflow) {
    return `overflow[${getViewElementOffset(state.overflow)}u] != 0u`;
  }
  const sentinel = getViewElementOffset(state.offsets) + state.vertexCount;
  return `overflow[${sentinel}u] > ${state.neighbors.length}u`;
}

/** Initializes identity labels, fails closed on overflow, and marks empty graphs converged. */
function addInitializationPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  state: ImportedConnectedComponents
): void {
  const bindings: Record<string, ConnectedComponentsBinding> = {
    ...(state.vertexCount
      ? {output: {view: state.output, usage: 'storage-write', atomic: true}}
      : {}),
    ...(state.vertexMask && state.vertexCount
      ? {vertexMask: {view: state.vertexMask, usage: 'storage-read'}}
      : {}),
    overflow: {view: state.overflow ?? state.offsets, usage: 'storage-read'},
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-write', atomic: true}}
      : {})
  };
  const convergenceOffset = state.converged
    ? `const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;`
    : '';
  const enabled = state.vertexMask
    ? `vertexMask[${getViewElementOffset(state.vertexMask)}u + index] != 0u`
    : 'true';
  const initializeOutput = state.vertexCount
    ? `if (index < VERTEX_COUNT) {
    let component = select(${INVALID_COMPONENT}u, index, !hasOverflow && (${enabled}));
    atomicStore(&output[OUTPUT_OFFSET + index], component);
  }`
    : '';
  const convergenceInitialization = state.converged
    ? `if (index == 0u) {
    let emptyAndValid = (VERTEX_COUNT == 0u || ${state.neighbors.length}u == 0u) && !hasOverflow;
    atomicStore(&converged[CONVERGED_OFFSET], select(0u, 1u, emptyAndValid));
  }`
    : '';
  const dispatchLayout = getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
    Math.max(state.vertexCount, 1),
    state.maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
${convergenceOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${CONNECTED_COMPONENTS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, CONNECTED_COMPONENTS_WORKGROUP_SIZE)}
  let hasOverflow = ${getOverflowExpression(state)};
  ${initializeOutput}
  ${convergenceInitialization}
}`;

  addConnectedComponentsPass(commandGraph, {
    id: `${state.id}-initialize`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Resets the caller-owned convergence scalar in a dedicated globally synchronized pass. */
function addConvergenceResetPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedConnectedComponents; iteration: number}
): void {
  const {state} = props;
  const converged = state.converged!;
  const bindings: Record<string, ConnectedComponentsBinding> = {
    overflow: {view: state.overflow ?? state.offsets, usage: 'storage-read'},
    converged: {view: converged, usage: 'storage-write', atomic: true}
  };
  const dispatchLayout = getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
    1,
    state.maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const CONVERGED_OFFSET: u32 = ${getViewElementOffset(converged)}u;
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${CONNECTED_COMPONENTS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, CONNECTED_COMPONENTS_WORKGROUP_SIZE)}
  if (index != 0u) { return; }
  atomicStore(&converged[CONVERGED_OFFSET], select(1u, 0u, (${getOverflowExpression(state)})));
}`;

  addConnectedComponentsPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-reset`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Hooks both endpoints of every forward edge into one order-independent weak component. */
function addHookingPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedConnectedComponents; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, ConnectedComponentsBinding> = {
    offsets: {view: state.offsets, usage: 'storage-read'},
    neighbors: {view: state.neighbors, usage: 'storage-read'},
    ...(state.adjacencyMask
      ? {adjacencyMask: {view: state.adjacencyMask, usage: 'storage-read'}}
      : {}),
    output: {view: state.output, usage: 'storage-read-write', atomic: true},
    overflow: {view: state.overflow ?? state.offsets, usage: 'storage-read'},
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-read-write', atomic: true}}
      : {})
  };
  const convergenceOffset = state.converged
    ? `const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;`
    : '';
  const markChanged = state.converged
    ? `if (previous > lower) { atomicStore(&converged[CONVERGED_OFFSET], 0u); }`
    : '';
  const dispatchLayout = getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
    state.vertexCount,
    state.maxComputeWorkgroupsPerDimension
  );
  const slotFilter = state.adjacencyMask
    ? `if (adjacencyMask[${getViewElementOffset(state.adjacencyMask)}u + slot] == 0u) { continue; }`
    : '';
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const CAPACITY: u32 = ${state.neighbors.length}u;
const OFFSETS_OFFSET: u32 = ${getViewElementOffset(state.offsets)}u;
const NEIGHBORS_OFFSET: u32 = ${getViewElementOffset(state.neighbors)}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
${convergenceOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${CONNECTED_COMPONENTS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, CONNECTED_COMPONENTS_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || (${getOverflowExpression(state)})) { return; }
  let first = min(offsets[OFFSETS_OFFSET + index], CAPACITY);
  let last = min(offsets[OFFSETS_OFFSET + index + 1u], CAPACITY);
  for (var slot = first; slot < last; slot++) {
    ${slotFilter}
    let neighbor = neighbors[NEIGHBORS_OFFSET + slot];
    if (neighbor >= VERTEX_COUNT) { continue; }
    let sourceComponent = atomicLoad(&output[OUTPUT_OFFSET + index]);
    let neighborComponent = atomicLoad(&output[OUTPUT_OFFSET + neighbor]);
    if (sourceComponent >= VERTEX_COUNT || neighborComponent >= VERTEX_COUNT) { continue; }
    let lower = min(sourceComponent, neighborComponent);
    let higher = max(sourceComponent, neighborComponent);
    if (higher == lower) { continue; }
    let previous = atomicMin(&output[OUTPUT_OFFSET + higher], lower);
    ${markChanged}
  }
}`;

  addConnectedComponentsPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-hook`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Shortcuts the monotone parent forest in a pass synchronized after all edge hooks. */
function addPointerJumpPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {state: ImportedConnectedComponents; iteration: number}
): void {
  const {state} = props;
  const bindings: Record<string, ConnectedComponentsBinding> = {
    output: {view: state.output, usage: 'storage-read-write', atomic: true},
    overflow: {view: state.overflow ?? state.offsets, usage: 'storage-read'},
    ...(state.converged
      ? {converged: {view: state.converged, usage: 'storage-read-write', atomic: true}}
      : {})
  };
  const convergenceOffset = state.converged
    ? `const CONVERGED_OFFSET: u32 = ${getViewElementOffset(state.converged)}u;`
    : '';
  const markChanged = state.converged
    ? `if (previous > parentComponent) {
    atomicStore(&converged[CONVERGED_OFFSET], 0u);
  }`
    : '';
  const dispatchLayout = getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
    state.vertexCount,
    state.maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${state.vertexCount}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(state.output)}u;
${convergenceOffset}
${getBindingDeclarations(bindings)}

@compute @workgroup_size(${CONNECTED_COMPONENTS_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, CONNECTED_COMPONENTS_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT || (${getOverflowExpression(state)})) { return; }
  let component = atomicLoad(&output[OUTPUT_OFFSET + index]);
  if (component >= VERTEX_COUNT) { return; }
  let parentComponent = atomicLoad(&output[OUTPUT_OFFSET + component]);
  if (parentComponent >= component) { return; }
  let previous = atomicMin(&output[OUTPUT_OFFSET + index], parentComponent);
  ${markChanged}
}`;

  addConnectedComponentsPass(commandGraph, {
    id: `${state.id}-iteration-${props.iteration}-jump`,
    source,
    bindings,
    dispatchLayout
  });
}

/** Declares every storage binding in its exact generated shader-layout order. */
function getBindingDeclarations(bindings: Record<string, ConnectedComponentsBinding>): string {
  return Object.entries(bindings)
    .map(([name, binding], location) => {
      const access = binding.usage === 'storage-read' ? 'read' : 'read_write';
      const element = binding.atomic ? 'atomic<u32>' : 'u32';
      return `@group(0) @binding(${location}) var<storage, ${access}> ${name}: array<${element}>;`;
    })
    .join('\n');
}

/** Compiles one bounded storage-buffer pass without hidden GPU allocation or submission. */
function addConnectedComponentsPass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: ConnectedComponentsPassProps
): void {
  commandGraph.addComputePass({
    id: props.id,
    resources: Object.values(props.bindings).map(({view, usage}) => ({buffer: view, usage})),
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id: props.id,
        source: props.source,
        shaderLayout: {
          bindings: Object.entries(props.bindings).map(([name, binding], location) => ({
            name,
            type: binding.usage === 'storage-read' ? 'read-only-storage' : 'storage',
            group: 0,
            location
          }))
        }
      });

      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {};
          for (const [name, binding] of Object.entries(props.bindings)) {
            bindings[name] = getViewBinding(binding.view, getBuffer);
          }
          const kernelBindings = bindings;
          kernel.dispatch(computePass, {
            bindings: kernelBindings,
            x: props.dispatchLayout.x,
            y: props.dispatchLayout.y,
            z: props.dispatchLayout.z
          });
        },
        destroy: () => kernel.destroy()
      };
    }
  });
}

/** Plans a bounded three-dimensional weak-component vertex or status dispatch. @internal */
export function getGPUGraphAlgorithmsConnectedComponentsDispatchLayout(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout(
    'GPUGraphAlgorithmsConnectedComponents',
    elementCount,
    CONNECTED_COMPONENTS_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
}
