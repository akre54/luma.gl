import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';
import {DGGSCellProjectionBenchmark} from '@site/src/components/docs/dggs-cell-projection-benchmark';

# GPU DGGS

<ExperimentalDocsTabs active="gpu-dggs" />

Discrete global grid indexes are compact, stable identifiers. Rendering and spatial analysis need
coordinates. The experimental `@luma.gl/graph/gpu-dggs` module connects those two representations
without pulling a large cell column back to JavaScript.

| Input | Shared GPU primitive | Output |
| --- | --- | --- |
| Packed `uint32x2` cell IDs | validation → topology decode → spherical projection | `float32x2` longitude/latitude or `float32x3` unit vectors |

The same command-graph operation powers the family-specific [`gpu-h3`](./gpu-h3) and
[`gpu-a5`](./gpu-a5) modules. Use those typed adapters when the grid is known in advance. Use
`GPUDGGSCellProjection` when a planner or application selects the grid at runtime.

## Live benchmark

This benchmark runs locally on the reader's WebGPU adapter. It validates every result before and
after timing and reports portable completion-fence latency separately from compute-only timestamp
queries when the adapter exposes them.

<DGGSCellProjectionBenchmark />

Benchmark input is already resident when timing begins. Upload, shader compilation, and validation
readback are deliberately excluded. That makes the result representative of a projection pass
inside a larger GPU pipeline, not a claim about end-to-end file ingestion.

## One contract, three entry points

| Entry point | Primary class | Best fit |
| --- | --- | --- |
| `@luma.gl/graph/gpu-dggs` | `GPUDGGSCellProjection` | Runtime-selected H3/A5 workflows and reusable planners |
| `@luma.gl/graph/gpu-h3` | `GPUH3CellProjection` | H3-specific pipelines and the clearest static types |
| `@luma.gl/graph/gpu-a5` | `GPUA5CellProjection` | A5-specific pipelines and the clearest static types |

All three classes implement `GPUCommandGraphContributor`. They add one bounded compute pass; they
do not submit commands, allocate hidden result buffers, or read output back.

```ts
import {GPUDGGSCellProjection} from '@luma.gl/graph/gpu-dggs';

new GPUDGGSCellProjection({
  family: selectedGrid, // 'h3' | 'a5'
  cells,
  output: centers,
  validity,
  projection: 'unit-vector'
}).addToGraph(graph);
```

## GPU data contract

`cells` is a packed `GraphDataView<'uint32x2'>`. The default `wordOrder: 'little-endian'` expects
the low 32-bit word first and matches a native `BigUint64Array` on the browser platforms WebGPU
targets. Set `wordOrder: 'high-low'` for canonical high-word-first storage.

| Property | Format | Meaning |
| --- | --- | --- |
| `cells` | `uint32x2` | One split 64-bit cell index per row |
| `output`, `projection: 'lnglat'` | `float32x2` | Longitude then latitude, in degrees |
| `output`, `projection: 'unit-vector'` | `float32x3` | Normalized Earth-centered Cartesian vector |
| `validity` | optional `uint32` | `1` for a valid cell, `0` for an invalid index |

Invalid rows write zero coordinates. All views must have the same row count, belong to the target
graph, use packed layouts, and occupy separate buffers. Three-dimensional bounded dispatch removes
the device's one-dimensional workgroup-count limit for very large arrays.

:::tip
**Choose unit vectors for GPU-resident geometry.**

Unit vectors compose directly with globe rendering, dot-product distance filters, hemisphere
tests, and normal generation. They also avoid the inverse trigonometry required to materialize
longitude and latitude. Choose geographic output when downstream code genuinely needs degrees.
:::

## Reusable pipeline patterns

The primitive is source-aligned: output row `i` always describes cell row `i`. That makes it easy
to compose with existing graph operations.

| Goal | Suggested graph |
| --- | --- |
| Render cell centers | DGGS projection → model vertex input |
| Cull invalid data | projection with validity → `GPUCompaction` |
| Aggregate spatial metrics | unit vectors → expression/filter → reduction |
| Publish coordinates to the CPU | projection → explicit readback ring |
| Switch grids at runtime | planner chooses `family` → `GPUDGGSCellProjection` |

The fixed H3 icosahedron basis is currently a 960-byte module-scope WGSL constant. A storage-buffer
table would add a binding and upload without reducing the working set. The shared family-neutral
class deliberately leaves that implementation choice behind the API boundary, so a profiled
adapter-specific table path could be added later without changing application graphs.

## Resident cell boundaries

Use `GPUDGGSCellBoundary` to generate closed cell paths from packed resident keys.
Use `GPUH3CellBoundary` or `GPUA5CellBoundary` from the family subpath when the
family is known at compile time. The boundary family type includes S2, quadkey, and geohash independently of the center projection family type.

```ts
import {GPUDGGSCellBoundary, getDGGSCellBoundaryCapacity} from '@luma.gl/graph/gpu-dggs';

const family = 'a5';
const capacity = getDGGSCellBoundaryCapacity(family);
const cells = graph.createDataView(inputHandle, {format: 'uint32x2', length: cellCount});
const vertices = graph.createDataView(outputHandle, {
  format: 'float32x2',
  length: cellCount * capacity
});
const counts = graph.createDataView(countHandle, {format: 'uint32', length: cellCount});

new GPUDGGSCellBoundary({family, cells, vertices, counts}).addToGraph(graph);
```

Register one operation per physical source chunk. The input, vertices, and optional
counts use separate caller-owned storage buffers. The caller compiles, encodes, and
submits the graph. Vertex row `i * capacity` starts the path for input row `i`.
Coordinates are longitude/latitude in degrees. Choose `float32x4` for the
`[longitude, latitude, 0, 0]` layout used by existing split-position consumers.
This layout retains Float32 coordinate precision.

The optional count includes the closing point. A count of zero identifies an invalid or
unsupported key, whose slot is zero-filled. Valid short paths repeat their first vertex
through the end of the slot. Consumers can read the counts on the GPU or use the fixed
layout directly when they can handle degenerate edges. The Arrow adapter wraps these
same fixed slots in its existing list representation for all five families.

| Family | Slot capacity | Count and coverage |
| --- | ---: | --- |
| H3 | 11 | Closed hexagons and pentagons, including vertices at icosahedral face crossings. Regular hexagons have count seven. Class III pentagons can use all eleven slots. |
| A5 | 6 | Four for triangles and six for pentagons. Longitude follows the cell center across the antimeridian. |
| S2 | 5 | Four spherical corners plus closure. Edge tessellation and seam clipping belong to downstream consumers. |
| Quadkey | 5 | Four tile bounds corners plus closure. The high six bits store the digit count (1–29), and the remaining bits hold right-aligned base-4 digits. |
| Geohash | 5 | Four box bounds corners plus closure. The high four bits store the character count (1–12), and the remaining bits hold right-aligned base-32 digits. |

The generic contributor accepts all five families. For packed quadkeys and geohashes, unused
payload bits must be zero. S2 uses the native 64-bit CellId.
H3 slots increased from seven to eleven. Allocate with
`getDGGSCellBoundaryCapacity('h3')` and use the per-row count for the true path length.
The Arrow adapter uses eleven positions per H3 row and pads shorter rings with their
first vertex. Icosahedron face crossings are part of the H3 ring. Antimeridian
splitting for a specific map viewport remains a consumer operation.

## Benchmark API

The live page uses an isolated optional entry point, keeping measurement code out of application
bundles:

```ts
import {runGPUDGGSCellProjectionBenchmark} from '@luma.gl/graph/gpu-dggs-benchmarks';

const report = await runGPUDGGSCellProjectionBenchmark(device, {
  family: 'h3',
  cells: packedCellWords,
  referenceValues: precomputedUnitVectorCenters,
  projection: 'unit-vector'
});
```

The runner accepts low-word/high-word `Uint32Array` input plus row-matched CPU or precomputed
reference centers. It rejects invalid cells, compares every output component with that oracle,
range-checks geographic centers, verifies vector normalization, and returns distributions rather
than a single best-case sample. Use `referenceTolerance` to override the default `0.03°` geographic
or `0.002` unit-vector component tolerance.
