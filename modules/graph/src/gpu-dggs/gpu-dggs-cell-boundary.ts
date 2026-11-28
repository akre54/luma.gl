// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {dggs} from '@luma.gl/shaders';
import {Kernel} from '../execution/compute/kernel';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import {
  createTransientView,
  getViewBinding,
  getViewElementOffset,
  validatePackedView
} from '../gpu-core/graph-data-view-utils';
import {
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource
} from '../gpu-core/gpu-dispatch-utils';
import {GPUScan} from '../gpu-core/gpu-scan';
import type {DGGSCellFamily, DGGSCellWordOrder} from './gpu-dggs-cell-projection';

/** Boundary dispatch workgroup width. */
const WORKGROUP_SIZE = 64;

/** Longitude/latitude in degrees, optionally stored with zero split low components. */
export type DGGSCellBoundaryFormat = 'float32x2' | 'float32x4';

/** Families supported by the resident boundary operation. */
export type DGGSCellBoundaryFamily = DGGSCellFamily | 's2' | 'quadkey' | 'geohash';

/** Caller-owned views for one physical chunk of packed cell IDs. */
export type GPUDGGSCellBoundaryProps = {
  /** Unique graph pass identifier. */
  id?: string;
  /** Cell index family. */
  family: DGGSCellBoundaryFamily;
  /** Packed split-uint64 keys. */
  cells: GraphDataView<'uint32x2'>;
  /** Fixed-size slots of closed paths, each with getDGGSCellBoundaryCapacity(family) vertices. */
  vertices: GraphDataView<DGGSCellBoundaryFormat>;
  /** Optional actual vertex count including closure. Zero denotes invalid or unsupported input. */
  counts?: GraphDataView<'uint32'>;
  /** Word order in cells. Defaults to little-endian. */
  wordOrder?: DGGSCellWordOrder;
};

/** Returns the number of vertex slots per row, including the closing point. */
export function getDGGSCellBoundaryCapacity(family: DGGSCellBoundaryFamily): number {
  switch (family) {
    case 'h3':
      return 11;
    case 'a5':
      return 6;
    case 's2':
    case 'quadkey':
    case 'geohash':
      return 5;
    default:
      throw new Error(`Unsupported DGGS boundary family ${family}`);
  }
}

/**
 * Expands a resident DGGS chunk into fixed-size boundary slots.
 * Invalid rows are zero-filled. Valid short paths repeat their first point through the end of
 * the slot. H3 supports hexagons, pentagons, and face crossings. A5 supports triangles and pentagons.
 * The caller owns all views and controls graph compilation and submission.
 */
export class GPUDGGSCellBoundary {
  /** Graph pass identifier. */
  readonly id: string;

  /** Checks packed layouts and source-aligned output lengths. */
  constructor(private readonly props: GPUDGGSCellBoundaryProps) {
    const {family, cells, vertices, counts} = props;
    this.id = props.id ?? `gpu-${family}-cell-boundary`;
    validatePackedView(cells, ['uint32x2'], `${this.id} cells`);
    validatePackedView(vertices, ['float32x2', 'float32x4'], `${this.id} vertices`);
    if (counts) validatePackedView(counts, ['uint32'], `${this.id} counts`);
    if (
      vertices.length !== cells.length * getDGGSCellBoundaryCapacity(family) ||
      (counts && counts.length !== cells.length)
    ) {
      throw new Error(`${this.id} output lengths must match the cell count and boundary capacity`);
    }
    if (
      vertices.buffer === cells.buffer ||
      counts?.buffer === cells.buffer ||
      counts?.buffer === vertices.buffer
    ) {
      throw new Error(`${this.id} cells, vertices, and counts must use separate buffers`);
    }
    this.props = {...props};
  }

  /** Registers one pass without uploading, submitting, or reading back any data. */
  addToGraph<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    addBoundaryPass(graph, this.id, this.props);
  }
}

/** Views used by the count and compact write stages. */
export type GPUH3CellBoundaryOffsetsProps = {
  /** Prefix for the count and scan passes. */
  id?: string;
  /** Packed resident cell IDs. */
  cells: GraphDataView<'uint32x2'>;
  /** Row count plus one offsets in vertex units. The last entry is the total vertex count. */
  offsets: GraphDataView<'uint32'>;
  /** Word order in cells. Defaults to little-endian. */
  wordOrder?: DGGSCellWordOrder;
};

/** Compact write inputs with offsets generated from the same cell snapshot. */
export type GPUH3CompactCellBoundaryProps = GPUH3CellBoundaryOffsetsProps & {
  /** Output with space for the terminal offset's vertex count. */
  vertices: GraphDataView<DGGSCellBoundaryFormat>;
};

/** Counts H3 boundary vertices and scans them into GPU-resident row offsets. */
export class GPUH3CellBoundaryOffsets {
  /** Captures the input and output views. */
  constructor(private readonly props: GPUH3CellBoundaryOffsetsProps) {
    validateH3CompactViews(props);
    this.props = {...props};
  }

  /** Registers a count pass followed by an exclusive scan, including for empty chunks. */
  addToGraph<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    const {cells, offsets, wordOrder, id = 'gpu-h3-boundary-offsets'} = this.props;
    if (cells.buffer.graph !== graph || offsets.buffer.graph !== graph) {
      throw new Error('H3 views must belong to the target graph');
    }
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', cells.length + 1);
    addBoundaryPass(graph, `${id}-count`, {family: 'h3', cells, counts, wordOrder});
    graph.add(
      new GPUScan({id: `${id}-scan`, input: counts, output: offsets}).getCommandNodes(graph)
    );
  }
}

/** Writes one closed ring per valid cell into consecutive vertex ranges without padding. */
export class GPUH3CompactCellBoundary {
  /** Captures packed views. The cell contents must match those used to generate offsets. */
  constructor(private readonly props: GPUH3CompactCellBoundaryProps) {
    validateH3CompactViews(props);
    const {cells, offsets, vertices} = props;
    validatePackedView(vertices, ['float32x2', 'float32x4'], 'H3 compact vertices');
    if (vertices.buffer === cells.buffer || vertices.buffer === offsets.buffer) {
      throw new Error('H3 compact vertices must use separate storage');
    }
    this.props = {...props};
  }

  /** Registers the write pass. Offsets and output can be reused on the same GPU. */
  addToGraph<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    addBoundaryPass(graph, this.props.id ?? 'gpu-h3-compact-boundary', {
      ...this.props,
      family: 'h3'
    });
  }
}

/** Checks the shared offset layout and prevents prefix sum overflow. */
function validateH3CompactViews({cells, offsets}: GPUH3CellBoundaryOffsetsProps): void {
  validatePackedView(cells, ['uint32x2'], 'H3 compact cells');
  validatePackedView(offsets, ['uint32'], 'H3 boundary offsets');
  if (
    offsets.length !== cells.length + 1 ||
    cells.length * getDGGSCellBoundaryCapacity('h3') > 0xffffffff
  ) {
    throw new Error('H3 boundary offsets require cells.length + 1 entries within uint32 capacity');
  }
  if (cells.buffer === offsets.buffer) {
    throw new Error('H3 cells and offsets must use separate storage');
  }
}

/** Internal kernel views for padded, count-only, and compact passes. */
type BoundaryPassProps = Omit<GPUDGGSCellBoundaryProps, 'id' | 'vertices'> & {
  vertices?: GraphDataView<DGGSCellBoundaryFormat>;
  offsets?: GraphDataView<'uint32'>;
};

/** Registers a shader pass with exactly the bindings used in that mode. */
function addBoundaryPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: BoundaryPassProps
): void {
  const {cells, vertices, counts, offsets, family} = props;
  const views = [
    ...(cells.length ? [{name: 'cells', view: cells, readOnly: true}] : []),
    ...(vertices ? [{name: 'vertices', view: vertices, readOnly: false}] : []),
    ...(counts ? [{name: 'counts', view: counts, readOnly: false}] : []),
    ...(offsets ? [{name: 'offsets', view: offsets, readOnly: true}] : [])
  ];
  for (const {view} of views) {
    if (view.buffer.graph !== graph) throw new Error(`${id} views must belong to the target graph`);
  }
  if (vertices && (!cells.length || !vertices.length)) return;
  const layout = getBoundedDispatchLayout(
    id,
    Math.max(cells.length, 1),
    WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const source = getBoundaryShaderSource(props, layout);
  graph.addComputePass({
    id,
    workload: {
      operation: `${family.toUpperCase()}CellBoundary`,
      commandCount: 1,
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      maximumInvocationCount: layout.x * layout.y * layout.z * WORKGROUP_SIZE,
      readByteLength: cells.length * cells.rowByteLength + (offsets ? offsets.length * 4 : 0),
      writeByteLength:
        (vertices ? vertices.length * vertices.rowByteLength : 0) + (counts ? counts.length * 4 : 0)
    },
    resources: views.map(({view, readOnly}) => ({
      buffer: view,
      usage: readOnly ? 'storage-read' : 'storage-write'
    })),
    compile: ({device}) => {
      const kernel = new Kernel(device, {
        id,
        source,
        shaderLayout: {
          bindings: views.map(({name, readOnly}, location) => ({
            name,
            location,
            group: 0,
            type: readOnly ? 'read-only-storage' : 'storage'
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            ...layout,
            bindings: Object.fromEntries(
              views.map(({name, view}) => [name, getViewBinding(view, getBuffer)])
            )
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}

/** Builds a bounded kernel over the supplied packed views. */
function getBoundaryShaderSource(
  {family, cells, vertices, counts, offsets, wordOrder}: BoundaryPassProps,
  dispatchLayout: {x: number; y: number; z: number}
): string {
  if (!cells.length && counts) {
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> counts: array<u32>;
@compute @workgroup_size(1)
fn main() { counts[${getViewElementOffset(counts)}u] = 0u; }`;
  }
  const capacity = getDGGSCellBoundaryCapacity(family);
  const split = vertices?.format === 'float32x4';
  const countIndex = counts ? getViewElementOffset(counts) : 0;
  const storeCount = (corners: string) =>
    counts ? `counts[${countIndex}u + index] = select(0u, ${corners} + 1u, ${corners} != 0u);` : '';
  const startPath = (corners: string) =>
    offsets
      ? `let start = offsets[${getViewElementOffset(offsets)}u + index];
    let end = offsets[${getViewElementOffset(offsets)}u + index + 1u];
    let slotCount = ${corners} + 1u;
    if (${corners} == 0u || end < start || end - start != slotCount || end > ${vertices?.length}u) { return; }`
      : `let start = index * ${capacity}u; let slotCount = ${capacity}u;`;
  const expandedWrite = vertices
    ? `${startPath('expanded.count')}
    for (var slot = 0u; slot < slotCount; slot += 1u) {
      writeVertex(start + slot, expanded.points[select(0u, slot, slot < expanded.count)]);
    }`
    : '';
  const {prepare, getVertex} = getBoundaryFamilyShader(
    family,
    `${storeCount('expanded.count')}
    ${expandedWrite}`
  );
  return /* wgsl */ `${dggs.source}
@group(0) @binding(0) var<storage, read> cells: array<u32>;
${vertices ? '@group(0) @binding(1) var<storage, read_write> vertices: array<f32>;' : ''}
${counts ? `@group(0) @binding(${vertices ? 2 : 1}) var<storage, read_write> counts: array<u32>;` : ''}
${offsets ? '@group(0) @binding(2) var<storage, read> offsets: array<u32>;' : ''}

${
  vertices
    ? `fn writeVertex(vertexIndex: u32, point: vec2f) {
  let offset = ${getViewElementOffset(vertices)}u + vertexIndex * ${split ? 4 : 2}u;
  vertices[offset] = point.x;
  vertices[offset + 1u] = point.y;
  ${split ? 'vertices[offset + 2u] = 0.0; vertices[offset + 3u] = 0.0;' : ''}
}`
    : ''
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getBoundedInvocationIndexSource(dispatchLayout, WORKGROUP_SIZE)}
  ${vertices ? '' : `if (index == 0u) { counts[${countIndex + cells.length}u] = 0u; }`}
  if (index >= ${cells.length}u) { return; }
  let offset = ${getViewElementOffset(cells)}u + index * 2u;
  let words = vec2u(cells[offset], cells[offset + 1u]);
  let cell = ${wordOrder === 'high-low' ? 'words' : 'dggs_u64_from_little_endian_words(words)'};
  ${prepare}
  ${storeCount('cornerCount')}
  ${
    vertices
      ? `${startPath('cornerCount')}
  var first = vec2f(0.0);
  if (cornerCount != 0u) {
    let cornerIndex = 0u;
    first = ${getVertex};
  }
  writeVertex(start, first);
  for (var cornerIndex = 1u; cornerIndex < slotCount; cornerIndex += 1u) {
    var point = first;
    if (cornerIndex < cornerCount) { point = ${getVertex}; }
    writeVertex(start + cornerIndex, point);
  }`
      : ''
  }
}`;
}

/** Selects the decoder and corner projection used by the shared output loop. */
function getBoundaryFamilyShader(
  family: DGGSCellBoundaryFamily,
  expandedBody: string
): {
  prepare: string;
  getVertex: string;
} {
  switch (family) {
    case 'h3':
      return {
        prepare: `let boundary = dggs_h3_prepare_boundary(cell);
  if (boundary.valid == 0u) {
    let expanded = dggs_h3_get_cross_face_boundary(cell);
    ${expandedBody}
    return;
  }
  let cornerCount = 6u;`,
        getVertex: 'dggs_h3_project_boundary_vertex(boundary, cornerIndex)'
      };
    case 'a5':
      return {
        prepare:
          'let boundary = dggs_a5_deserialize(cell);\n  let cornerCount = select(0u, dggs_a5_get_shape_vertex_count(boundary), boundary.valid != 0u && !dggs_u64_is_zero(cell));',
        getVertex: 'dggs_a5_project_boundary_vertex(cell, boundary, cornerIndex)'
      };
    case 's2':
      return {
        prepare: `let isValid = dggs_s2_is_valid_cell_id(cell);
  let cornerCount = select(0u, 4u, isValid);
  let safeCell = select(vec2u(0x10000000u, 0u), cell, isValid);
  let face = dggs_s2_get_face(safeCell);
  let level = dggs_s2_get_level(safeCell);
  let ij = dggs_s2_get_ij(safeCell);`,
        getVertex: 'dggs_s2_project_boundary_vertex(face, level, ij, cornerIndex)'
      };
    case 'quadkey':
    case 'geohash': {
      const geohash = family === 'geohash';
      const lengthOffset = geohash ? 60 : 58;
      const lengthBits = geohash ? 4 : 6;
      const bitsPerDigit = geohash ? 5 : 2;
      const maximumLength = geohash ? 12 : 29;
      return {
        prepare: `let length = dggs_u64_extract_bits(cell, ${lengthOffset}u, ${lengthBits}u);
  let payload = dggs_u64_set_bits(cell, ${lengthOffset}u, ${lengthBits}u, 0u);
  let isValid = length > 0u && length <= ${maximumLength}u &&
    dggs_u64_is_zero(dggs_u64_shift_right(payload, length * ${bitsPerDigit}u));
  let cornerCount = select(0u, 4u, isValid);
  let bounds = dggs_${family}_get_bounds(cell);`,
        getVertex: 'dggs_bounds_get_boundary_point(bounds, cornerIndex)'
      };
    }
  }
}
