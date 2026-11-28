# Two-Legged Landing Rig at 120 Hz

A 4 kg chassis drops 30 cm onto split decks with a 15 cm height difference. The upper foot strikes first. After the first foot touches, a commanded right strut extension brings the chassis level. The 3 kg and 5 kg payloads are rigidly carried 16 cm off center. Both lanes use the same resolved scenegraph attachments, box shapes, mass properties, drive gains, and friction. The ordered reference uses 64 iterations. The reduced lane couples both box feet and the chassis to the indexed world contacts and runs eight iterations for the surrounding world. The levelling command is a timed extension for the authored step height.

Start Vite with `yarn workspace luma.gl-examples-experimental-gpu-rigid-body-world start --host 127.0.0.1 --port 5346`, then open `landing-rig.html`. Choose **Drop rig**. The panel shows independent foot reactions in the collapsed view. Settings exposes drop height, payload mass, and lateral offset, along with chassis diagnostics and access to the GPU Graph.

```sh
node scripts/physics/benchmark-landing-rig.mjs http://127.0.0.1:5346/
node scripts/physics/benchmark-landing-rig.mjs http://127.0.0.1:5346/ --ordered
node scripts/physics/benchmark-landing-rig.mjs http://127.0.0.1:5346/ --heavy
node scripts/physics/benchmark-landing-rig.mjs http://127.0.0.1:5346/ --ordered --heavy
```

The benchmark runs 360 deterministic steps, with 352 timed samples. Timings include actuator updates, encoding, submission, and GPU completion. Timing excludes the first eight steps, compilation, rendering, and diagnostic reads. The script samples geometry every four steps and computes penetration from the lowest corner of each oriented foot box relative to its deck. Lateral displacement is measured from the initial foot centers to the settled centers. Recovery is the first sampled time after the transient from which tilt stays below 2° and chassis speed stays below 0.035 m/s. Both lanes gate penetration, recovery, final height and tilt, and foot displacement.

The browser regressions check two-foot support force against the total weight at both payloads. A separate zero-gravity fixture exercises a revolute drive on a rotating, translating chassis, checks total linear and angular momentum before and after the motor move, then verifies reset.

## Apple GPU results

Chromium WebGPU Core on Apple Metal 3. All eight runs passed their physical gates. The table shows the repeat timings plus the initial p95 from the first series on the shared machine. Pose outcomes agreed across both series.

| Payload | Solver | Repeat p50 / p95 | Initial p95 | Peak sampled penetration | Final foot displacement | Recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3 kg | Reduced | 4.7 / 6.8 ms | 11.6 ms | 8.6 mm | 6.6 mm | 1.23 s |
| 3 kg | Ordered | 5.6 / 7.0 ms | 25.2 ms | 11.0 mm | 41.0 mm | 1.33 s |
| 5 kg | Reduced | 3.8 / 6.2 ms | 12.7 ms | 8.6 mm | 2.3 mm | 1.30 s |
| 5 kg | Ordered | 5.7 / 6.9 ms | 11.6 ms | 15.1 mm | 22.3 mm | 1.73 s |

Both solvers met the 8.33 ms p95 budget in the repeat series. Both exceeded it in the initial series. The reduced solver showed less foot displacement at both masses and, after recovery, measured a combined support force of 81.42 N with 3 kg and 101.04 N with 5 kg. The ordered data covers pose and timing. [Lane summaries and representative step samples](landing-rig-apple.json)
