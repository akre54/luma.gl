// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {dggs} from '@luma.gl/shaders';
import {Kernel} from '../execution/compute/kernel';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import {
  getViewBinding,
  getViewElementOffset,
  validatePackedView
} from '../gpu-core/graph-data-view-utils';
import {
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource
} from '../gpu-core/gpu-dispatch-utils';
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
    const {cells, vertices, counts, family} = this.props;
    for (const view of [cells, vertices, counts]) {
      if (view && view.buffer.graph !== graph) {
        throw new Error(`${this.id} views must belong to the target graph`);
      }
    }
    if (!cells.length) return;
    const dispatchLayout = getBoundedDispatchLayout(
      this.id,
      cells.length,
      WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const source = getBoundaryShaderSource(this.props, dispatchLayout);
    graph.addComputePass({
      id: this.id,
      workload: {
        operation: `${family.toUpperCase()}CellBoundary`,
        commandCount: 1,
        maximumWorkgroupCount: dispatchLayout.x * dispatchLayout.y * dispatchLayout.z,
        maximumInvocationCount:
          dispatchLayout.x * dispatchLayout.y * dispatchLayout.z * WORKGROUP_SIZE,
        readByteLength: cells.length * cells.rowByteLength,
        writeByteLength:
          vertices.length * vertices.rowByteLength +
          (counts ? counts.length * counts.rowByteLength : 0)
      },
      resources: [
        {buffer: cells, usage: 'storage-read'},
        {buffer: vertices, usage: 'storage-write'},
        ...(counts ? [{buffer: counts, usage: 'storage-write'} as const] : [])
      ],
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          id: this.id,
          source,
          shaderLayout: {
            bindings: [
              {name: 'cells', type: 'read-only-storage', group: 0, location: 0},
              {name: 'vertices', type: 'storage', group: 0, location: 1},
              ...(counts ? [{name: 'counts', type: 'storage', group: 0, location: 2} as const] : [])
            ]
          }
        });
        return {
          encode: ({computePass, getBuffer}) =>
            kernel.dispatch(computePass, {
              ...dispatchLayout,
              bindings: {
                cells: getViewBinding(cells, getBuffer),
                vertices: getViewBinding(vertices, getBuffer),
                ...(counts ? {counts: getViewBinding(counts, getBuffer)} : {})
              }
            }),
          destroy: () => kernel.destroy()
        };
      }
    });
  }
}

/** Builds a bounded kernel over the supplied packed views. */
function getBoundaryShaderSource(
  {family, cells, vertices, counts, wordOrder}: GPUDGGSCellBoundaryProps,
  dispatchLayout: {x: number; y: number; z: number}
): string {
  const capacity = getDGGSCellBoundaryCapacity(family);
  const split = vertices.format === 'float32x4';
  const {prepare, getVertex} = getBoundaryFamilyShader(
    family,
    counts
      ? `counts[${getViewElementOffset(counts)}u + index] = select(0u, expanded.count + 1u, expanded.count != 0u);`
      : ''
  );
  return /* wgsl */ `${dggs.source}
@group(0) @binding(0) var<storage, read> cells: array<u32>;
@group(0) @binding(1) var<storage, read_write> vertices: array<f32>;
${counts ? '@group(0) @binding(2) var<storage, read_write> counts: array<u32>;' : ''}

fn writeVertex(vertexIndex: u32, point: vec2f) {
  let offset = ${getViewElementOffset(vertices)}u + vertexIndex * ${split ? 4 : 2}u;
  vertices[offset] = point.x;
  vertices[offset + 1u] = point.y;
  ${split ? 'vertices[offset + 2u] = 0.0; vertices[offset + 3u] = 0.0;' : ''}
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getBoundedInvocationIndexSource(dispatchLayout, WORKGROUP_SIZE)}
  if (index >= ${cells.length}u) { return; }
  let offset = ${getViewElementOffset(cells)}u + index * 2u;
  let words = vec2u(cells[offset], cells[offset + 1u]);
  let cell = ${wordOrder === 'high-low' ? 'words' : 'dggs_u64_from_little_endian_words(words)'};
  ${prepare}
  ${counts ? `counts[${getViewElementOffset(counts)}u + index] = select(0u, cornerCount + 1u, cornerCount != 0u);` : ''}
  let start = index * ${capacity}u;
  var first = vec2f(0.0);
  if (cornerCount != 0u) {
    let cornerIndex = 0u;
    first = ${getVertex};
  }
  writeVertex(start, first);
  for (var cornerIndex = 1u; cornerIndex < ${capacity}u; cornerIndex += 1u) {
    var point = first;
    if (cornerIndex < cornerCount) { point = ${getVertex}; }
    writeVertex(start + cornerIndex, point);
  }
}`;
}

/** Selects the decoder and corner projection used by the shared output loop. */
function getBoundaryFamilyShader(
  family: DGGSCellBoundaryFamily,
  countStore: string
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
    ${countStore}
    let start = index * ${getDGGSCellBoundaryCapacity(family)}u;
    for (var slot = 0u; slot < ${getDGGSCellBoundaryCapacity(family)}u; slot += 1u) {
      writeVertex(start + slot, expanded.points[select(0u, slot, slot < expanded.count)]);
    }
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
