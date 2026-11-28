# Closed-Loop Lift at 120 Hz

One motor drives a fixed-base four-bar linkage. Its platform lifts a freely rotating box, presses the box into an overhead stop, then backs away. Both solvers resolve the same four `ScenegraphJoint` attachments and use the same drive, mass, friction, geometry and timestep. Ordered uses 64 maximal-coordinate iterations. Reduced uses three tree coordinates, a two-row planar closure block, and a four-point platform contact patch. The surrounding rigid-body world uses eight parallel iterations.

The reduced solver reads the platform reaction by summing its vertical contact impulses on the load over one step. Ordered reports the pose and closure metrics. Its measured contact reaction is omitted from this benchmark.

## Reproduction

Start the example with `yarn workspace luma.gl-examples-experimental-gpu-rigid-body-world start --port 5344`. Open `four-bar.html` and use **Run lift**. The Settings panel controls mass, offset and torque and reports closure error and tilt. The GPU Graph and timeline show the simulation passes.

From the repository root run:

```sh
node scripts/physics/benchmark-four-bar.mjs http://localhost:5344/
node scripts/physics/benchmark-four-bar.mjs http://localhost:5344/ --ordered
```

Each lane takes 560 fixed steps. It excludes shader compilation, the first 20 steps, rendering and diagnostic reads. Each timing includes encoding, submitting and waiting for GPU completion. The script samples quality every 20 steps. Both lanes must stay below 3 mm closure error and 1° tilt, raise the platform at least 25 cm into the press, reach the ceiling within 15 mm, return within 3 cm vertically and retain the load within 6 cm horizontally. The GPU tests check force above the payload weight during the press and within 2 N of the weight after release, with both 3 kg and 5 kg payloads. They also check that a rank-deficient closure stops before accepting a step.

## Apple GPU results

Chromium WebGPU Core on Apple Metal 3. Each lane has 540 synchronized timing samples. Run the same commands with `--heavy` for the 5 kg workload.

| Load | Solver | Median step | p95 step | Max sampled closure | Max sampled tilt |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 kg | Reduced | 2.5 ms | 4.7 ms | 0.00146 mm | 0.000034° |
| 3 kg | Ordered | 8.6 ms | 11.8 ms | 2.53 mm | 0.350° |
| 5 kg | Reduced | 2.8 ms | 5.5 ms | 0.00114 mm | 0.000032° |
| 5 kg | Ordered | 8.9 ms | 12.3 ms | 2.91 mm | 0.215° |

All four lanes passed the same pose and motion gates. Both reduced runs met the 8.33 ms budget at p95. In the 3 kg run, the sampled platform reaction was 218.31 N during the press and 29.56 N after release. The 5 kg run measured 202.91 N and 49.73 N, respectively. The reports reflect one device and one run per lane. [Raw snapshots](four-bar-lift-apple.json)
