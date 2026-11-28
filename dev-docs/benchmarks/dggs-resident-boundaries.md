# GPU-resident DGGS boundaries

The existing Arrow helper, `convertDggsCellKeysToGPUPaths`, expands packed resident
keys on WebGPU. It emits a closed path in each input row and preserves physical
chunks. The center operations in `gpu-h3` and `gpu-a5` are separate graph
contributors. They emit longitude/latitude pairs or unit vectors.

| Cell family | Existing boundary output | Scope |
| --- | --- | --- |
| H3 | Up to ten vertices plus closure | Includes pentagons and face-edge intersections. Invalid keys emit zero paths. |
| A5 | Five corners plus closure | Uses the A5 polygon decoder, with reflected cell fixtures in the shader tests. |
| S2 | Four corners plus closure | Cell corners on the sphere. Map seam clipping and edge tessellation are consumer work. |
| Quadkey | Four corners plus closure | Web Mercator tile bounds expressed as longitude and latitude. |
| Geohash | Four corners plus closure | Longitude and latitude box. |

The [Arrow boundary parity test](../../modules/graph/test/arrow/dggs-gpu-polygons.spec.ts)
covers all five families. Its fixtures verify regular cells, H3 face crossings and pentagons, and invalid keys.

## H3 capture

Chromium WebGPU on Apple hardware, using a disk around San Francisco at H3 resolution
9. The workload cycles through the same local cells to reach the requested batch size.
Both paths emit closed Float32 paths. Timings are wall time through a queue fence,
with allocation and submission included. The resident input is prepared ahead of the
timed GPU path. The CPU path generates polygons from ready cell IDs and uploads the
Float32 coordinates. Six measured iterations follow one warm-up for each size.
A readback before timing checked for unsupported rows and verified the location.

| Capture | Cells | Resident GPU boundary median | CPU boundary plus upload median |
| --- | ---: | ---: | ---: |
| Initial | 16,384 | 58.3 ms | 39.4 ms |
| Initial | 131,072 | 331.6 ms | 320.6 ms |
| Repeat | 16,384 | 93.2 ms | 40.6 ms |
| Repeat | 131,072 | 639.1 ms | 274.0 ms |

The current helper was slower in both captures, with near parity in the initial
131K run. The repeat showed substantial variation on a shared development machine.
These results establish that the resident path works for the sample. A dedicated run
is needed for a stable throughput comparison. The capture covers boundary creation.
Draws, picking, and other vendors remain unmeasured.

```sh
LUMA_TEST_BROWSER_BENCHMARKS=true yarn exec vitest run --project headless --reporter=verbose h3-boundary-resident-benchmark
```

## H3 decode once per cell

The extraction kernel now decodes and validates a single-face H3 boundary once per
cell, projects six corners and copies the first projected point to close the path.
The same Apple WebGPU adapter and benchmark provided a fresh baseline and two
optimized runs. Each entry is a median in milliseconds.

| Cells | Original GPU | Optimized GPU | Optimized repeat | CPU plus upload, optimized runs |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 | 51.2 | 13.2 | 12.9 | 29.6 / 29.4 |
| 131,072 | 325.3 | 22.5 | 21.5 | 228.3 / 238.7 |

This is a 3.9–4.0× improvement for 16K cells and a 14.5–15.1× improvement
for 131K cells. The CPU baseline also varied across runs. This capture measures
the existing supported single-face, resolution-9 batch and ends at a queue fence.
The boundary parity suite checks regular and unsupported inputs in both output
layouts, including a Class III cell.

## Shared graph operation capture

After moving Arrow H3 extraction onto the shared graph operation, the same Apple
WebGPU workload took 3.6 ms for 16,384 rows and 12.7 ms for 131,072 rows in one
capture. The benchmark includes the Arrow adapter, graph compilation, allocation,
and GPU completion. This capture was separate from the decode-once runs above.

## Follow-up

The [resident boundary contributor](../../docs/api-reference/experimental/gpu-dggs.md#resident-cell-boundaries)
borrows packed keys and writes fixed-capacity vertex slots with optional counts. The Arrow
helper uses it for H3, A5, S2, quadkey, and geohash. The H3 and A5 subpaths also expose
typed entry points.

The shared operation now includes pentagon and icosahedral face-crossing rings.
It writes into eleven slots per H3 row, including closure, with repeated first
vertices as padding. The captures above used the previous seven-slot layout.
Seam handling and high-resolution precision remain separate qualification work.
[H3 boundary algorithm](https://h3geo.org/docs/core-library/cellToBoundaryDesc/)
· [Upstream vertex limit](https://github.com/uber/h3/blob/master/src/h3lib/include/h3api.h.in)
