// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../playwright/get-playwright-launch-options.mjs';

const url = new URL(process.argv[2] ?? 'http://127.0.0.1:5346/');
if (!['localhost', '127.0.0.1'].includes(url.hostname)) throw new Error('Expected local lab');
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
try {
  const page = await browser.newPage();
  await page.route(url.href, route =>
    route.fulfill({body: '<title>Landing rig benchmark</title>', contentType: 'text/html'})
  );
  page.on('console', m => {
    if (m.type() === 'error') console.error(m.text());
  });
  await page.goto(url.href);
  const report = await page.evaluate(
    async ({root, ordered, payloadMass}) => {
      const [{luma}, {webgpuAdapter}, {GPUCommandGraph}, {makeLandingRig}] =
        await Promise.all([
          import(root + '/modules/core/src/index.ts'),
          import(root + '/modules/core/src/webgpu/index.ts'),
          import(root + '/modules/graph/src/gpu-core/gpu-command-graph.ts'),
          import('/landing-rig.ts')
        ]);
      const device = await luma.createDevice({adapters: [webgpuAdapter], type: 'webgpu'});
      const trial = makeLandingRig({ordered, payloadMass});
      const definition = new GPUCommandGraph(device);
      trial.addToGraph(definition);
      const graph = await definition.compileAsync();
      try {
        const measurements = [],
          snapshots = [];
        for (let step = 0; step < 360; step++) {
          const start = performance.now();
          if (step % 4 === 0) trial.command(step);
          const encoder = device.createCommandEncoder();
          graph.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          const fence = device.createFence();
          try {
            await fence.signaled;
          } finally {
            fence.destroy();
          }
          if (step >= 8) measurements.push(performance.now() - start);
          if (step % 4 === 0 || step === 359)
            snapshots.push({step, ...(await trial.read())});
        }
        measurements.sort((first, second) => first - second);
        const settled = snapshots.at(-1);
        const maximumTilt = Math.max(...snapshots.map(snapshot => snapshot.tilt));
        const maximumPenetration = Math.max(...snapshots.flatMap(snapshot => snapshot.penetrations));
        const lastUnsettled = snapshots.findLast(snapshot =>
          snapshot.step >= 36 && (snapshot.tilt > Math.PI / 90 || Math.hypot(...snapshot.velocity) > 0.035)
        );
        const recoverySeconds = lastUnsettled ? (lastUnsettled.step + 4) / 120 : null;
        const slip = Math.max(...settled.feet.map((foot, index) =>
          Math.hypot(foot[0] - snapshots[0].feet[index][0], foot[1] - snapshots[0].feet[index][1])
        ));
        const qualified =
          snapshots.every(snapshot => snapshot.complete) &&
          maximumPenetration < 0.045 &&
          recoverySeconds !== null && recoverySeconds < 2.8 &&
          settled.tilt < Math.PI / 45 &&
          Math.abs(settled.position[2] - 0.84) < 0.07 &&
          Math.hypot(...settled.velocity) < 0.035 &&
          slip < 0.12;
        return {
          device: device.info,
          ordered,
          payloadMass,
          qualified,
          maximumPenetration,
          recoverySeconds,
          slip,
          maximumTilt,
          snapshots,
          samples: measurements.length,
          p50: measurements[Math.floor(measurements.length * 0.5)],
          p95: measurements[Math.floor(measurements.length * 0.95)]
        };
      } finally {
        graph.destroy();
        trial.destroy();
        device.destroy();
      }
    },
    {
      root: '/@fs' + process.cwd(),
      ordered: process.argv.includes('--ordered'),
      payloadMass: process.argv.includes('--heavy') ? 5 : 3
    }
  );
  console.log(JSON.stringify(report, null, 2));
  if (!report.qualified) process.exitCode = 2;
} finally {
  await browser.close();
}
