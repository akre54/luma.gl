// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RAY_TRACING_SCENE_TYPES} from './ray-tracing-scene-shaders';

/** Limits the amount of serial GPU work in one surface-area build. */
export const SURFACE_AREA_TLAS_MAXIMUM_PRIMITIVES = 1024;

/** Uint32 words per node and per final header record. */
export const SURFACE_AREA_TLAS_WORDS_PER_NODE = 4;

/** Shared input and persistent hierarchy for the top-level build and refit passes. */
const SURFACE_AREA_TLAS_BINDINGS = /* wgsl */ `
const INVALID_NODE = 0xffffffffu;
const INVALID_BOUND = 3.402823466e+38;

struct Bounds {
  minimum: vec3<f32>,
  maximum: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> primitiveMinima: array<f32>;
@group(0) @binding(1) var<storage, read> primitiveMaxima: array<f32>;
@group(0) @binding(2) var<storage, read_write> nodeMinima: array<f32>;
@group(0) @binding(3) var<storage, read_write> nodeMaxima: array<f32>;
// Node: left child or primitive, right child or INVALID_NODE, depth, reserved.
// Final record: active node count, deepest level, reserved, reserved.
@group(0) @binding(4) var<storage, read_write> topology: array<vec4<u32>>;

fn emptyBounds() -> Bounds {
  return Bounds(vec3<f32>(INVALID_BOUND), vec3<f32>(-INVALID_BOUND));
}

fn unionBounds(first: Bounds, second: Bounds) -> Bounds {
  return Bounds(min(first.minimum, second.minimum), max(first.maximum, second.maximum));
}

fn getPrimitiveBounds(primitive: u32) -> Bounds {
  let start = primitive * 3u;
  let bounds = Bounds(
    vec3<f32>(primitiveMinima[start], primitiveMinima[start + 1u], primitiveMinima[start + 2u]),
    vec3<f32>(primitiveMaxima[start], primitiveMaxima[start + 1u], primitiveMaxima[start + 2u])
  );
  if (!all(bounds.minimum <= bounds.maximum) ||
      !all(abs(bounds.minimum) <= vec3<f32>(INVALID_BOUND)) ||
      !all(abs(bounds.maximum) <= vec3<f32>(INVALID_BOUND))) {
    return emptyBounds();
  }
  return bounds;
}

fn storeNodeBounds(node: u32, bounds: Bounds) {
  let start = node * 3u;
  for (var axis = 0u; axis < 3u; axis++) {
    nodeMinima[start + axis] = bounds.minimum[axis];
    nodeMaxima[start + axis] = bounds.maximum[axis];
  }
}
`;

/** Builds a binned surface-area tree with explicit child links and one primitive per leaf. */
export const RAY_TRACING_SURFACE_AREA_BUILD_SHADER = /* wgsl */ `
${RAY_TRACING_SCENE_TYPES}
${SURFACE_AREA_TLAS_BINDINGS}

const BIN_COUNT = 8u;
// Keep the depth below the traversal stack capacity. Splits must leave enough
// levels for even the larger child to reach one primitive per leaf.
const MAXIMUM_DEPTH = 16u;
struct Bin {
  bounds: Bounds,
  count: u32,
};

@group(0) @binding(5) var<uniform> uniforms: RayTracingUniforms;
@group(0) @binding(6) var<storage, read_write> primitiveOrder: array<u32>;

fn emptyBin() -> Bin {
  return Bin(emptyBounds(), 0u);
}

fn combineBins(first: Bin, second: Bin) -> Bin {
  return Bin(unionBounds(first.bounds, second.bounds), first.count + second.count);
}

fn getCentroid(bounds: Bounds) -> vec3<f32> {
  return bounds.minimum * 0.5 + bounds.maximum * 0.5;
}

fn getBin(coordinate: f32, minimum: f32, extent: f32) -> u32 {
  return min(u32(clamp((coordinate - minimum) / extent, 0.0, 1.0) * f32(BIN_COUNT)), BIN_COUNT - 1u);
}

fn getHalfSurfaceArea(bounds: Bounds) -> f32 {
  let extent = max(bounds.maximum - bounds.minimum, vec3<f32>(0.0));
  return extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
}

@compute @workgroup_size(1)
fn main() {
  let primitiveCount = uniforms.dimensions.z;
  let headerIndex = arrayLength(&topology) - 1u;
  if (primitiveCount == 0u) {
    storeNodeBounds(0u, emptyBounds());
    topology[headerIndex] = vec4<u32>(0u);
    return;
  }
  for (var primitive = 0u; primitive < primitiveCount; primitive++) {
    primitiveOrder[primitive] = primitive;
  }

  // Pending nodes temporarily store a range into primitiveOrder in x/y.
  // Breadth-first allocation keeps the two children adjacent.
  topology[0] = vec4<u32>(0u, primitiveCount, 0u, 0u);
  var nodeCount = 1u;
  var deepestLevel = 0u;
  for (var node = 0u; node < nodeCount; node++) {
    let range = topology[node];
    let start = range.x;
    let count = range.y;
    let depth = range.z;
    deepestLevel = max(depth, deepestLevel);
    if (count == 1u) {
      let primitive = primitiveOrder[start];
      topology[node] = vec4<u32>(primitive, INVALID_NODE, depth, 0u);
      storeNodeBounds(node, getPrimitiveBounds(primitive));
      continue;
    }

    var bounds = emptyBounds();
    var centroidBounds = emptyBounds();
    for (var offset = start; offset < start + count; offset++) {
      let primitiveBounds = getPrimitiveBounds(primitiveOrder[offset]);
      let centroid = getCentroid(primitiveBounds);
      bounds = unionBounds(bounds, primitiveBounds);
      centroidBounds = unionBounds(centroidBounds, Bounds(centroid, centroid));
    }
    storeNodeBounds(node, bounds);

    let childLimit = 1u << (MAXIMUM_DEPTH - depth - 1u);
    let centroidExtent = centroidBounds.maximum - centroidBounds.minimum;
    var bestCost = INVALID_BOUND;
    var bestAxis = 3u;
    var bestBoundary = 0u;
    for (var axis = 0u; axis < 3u; axis++) {
      if (!(centroidExtent[axis] > 0.0 && centroidExtent[axis] <= INVALID_BOUND)) { continue; }
      var bins: array<Bin, BIN_COUNT>;
      for (var bin = 0u; bin < BIN_COUNT; bin++) { bins[bin] = emptyBin(); }
      for (var offset = start; offset < start + count; offset++) {
        let primitiveBounds = getPrimitiveBounds(primitiveOrder[offset]);
        let bin = getBin(getCentroid(primitiveBounds)[axis], centroidBounds.minimum[axis],
          centroidExtent[axis]);
        bins[bin] = combineBins(bins[bin], Bin(primitiveBounds, 1u));
      }

      var suffix: array<Bin, BIN_COUNT>;
      var right = emptyBin();
      for (var end = BIN_COUNT; end > 0u; end--) {
        right = combineBins(bins[end - 1u], right);
        suffix[end - 1u] = right;
      }
      var left = emptyBin();
      for (var boundary = 1u; boundary < BIN_COUNT; boundary++) {
        left = combineBins(left, bins[boundary - 1u]);
        right = suffix[boundary];
        if (left.count == 0u || right.count == 0u ||
            left.count > childLimit || right.count > childLimit) { continue; }
        let cost = getHalfSurfaceArea(left.bounds) * f32(left.count) +
          getHalfSurfaceArea(right.bounds) * f32(right.count);
        if (cost < bestCost) {
          bestCost = cost;
          bestAxis = axis;
          bestBoundary = boundary;
        }
      }
    }

    // Degenerate centroid ranges still split evenly, in the current order.
    var split = start + count / 2u;
    if (bestAxis < 3u) {
      split = start;
      for (var offset = start; offset < start + count; offset++) {
        let primitive = primitiveOrder[offset];
        let bin = getBin(getCentroid(getPrimitiveBounds(primitive))[bestAxis],
          centroidBounds.minimum[bestAxis], centroidExtent[bestAxis]);
        if (bin < bestBoundary) {
          primitiveOrder[offset] = primitiveOrder[split];
          primitiveOrder[split] = primitive;
          split++;
        }
      }
    }
    topology[node] = vec4<u32>(nodeCount, nodeCount + 1u, depth, 0u);
    topology[nodeCount] = vec4<u32>(start, split - start, depth + 1u, 0u);
    topology[nodeCount + 1u] = vec4<u32>(split, start + count - split, depth + 1u, 0u);
    nodeCount += 2u;
  }
  topology[headerIndex] = vec4<u32>(nodeCount, deepestLevel, 0u, 0u);
}
`;

/** Reloads leaf bounds and refits the retained surface-area tree, one depth at a time. */
export const RAY_TRACING_SURFACE_AREA_REFIT_SHADER = /* wgsl */ `
${SURFACE_AREA_TLAS_BINDINGS.replace('var<storage, read_write> topology', 'var<storage, read> topology')}

const REFIT_WORKGROUP_SIZE = 128u;
var<workgroup> header: vec2<u32>;

fn getNodeBounds(node: u32) -> Bounds {
  let start = node * 3u;
  return Bounds(
    vec3<f32>(nodeMinima[start], nodeMinima[start + 1u], nodeMinima[start + 2u]),
    vec3<f32>(nodeMaxima[start], nodeMaxima[start + 1u], nodeMaxima[start + 2u])
  );
}

@compute @workgroup_size(REFIT_WORKGROUP_SIZE)
fn main(@builtin(local_invocation_index) lane: u32) {
  if (lane == 0u) {
    header = topology[arrayLength(&topology) - 1u].xy;
  }
  let tree = workgroupUniformLoad(&header);
  // One workgroup owns the entire tree. Each barrier makes child storage
  // writes visible before any lane processes the next parent level.
  for (var depth = i32(tree.y); depth >= 0; depth--) {
    for (var node = lane; node < tree.x; node += REFIT_WORKGROUP_SIZE) {
      let links = topology[node];
      if (links.z != u32(depth)) { continue; }
      var bounds: Bounds;
      if (links.y == INVALID_NODE) {
        bounds = getPrimitiveBounds(links.x);
      } else {
        bounds = unionBounds(getNodeBounds(links.x), getNodeBounds(links.y));
      }
      storeNodeBounds(node, bounds);
    }
    storageBarrier();
  }
}
`;
