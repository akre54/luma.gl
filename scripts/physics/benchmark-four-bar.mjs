// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../playwright/get-playwright-launch-options.mjs';

const url = new URL(process.argv[2] ?? 'http://127.0.0.1:5344/');
if (!['localhost', '127.0.0.1'].includes(url.hostname)) throw new Error('Expected local lab');
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
try {
  const page = await browser.newPage();
  await page.route(url.href, route =>
    route.fulfill({body: '<title>Four-bar benchmark</title>', contentType: 'text/html'})
  );
  page.on('console', m => {
    if (m.type() === 'error') console.error(m.text());
  });
  await page.goto(url.href);
  const report = await page.evaluate(
    async ({root, ordered, payloadMass}) => {
      const [{luma}, {webgpuAdapter}, {GPUCommandGraph}, {makeFourBar, fourBarTarget}] =
        await Promise.all([
          import(root + '/modules/core/src/index.ts'),
          import(root + '/modules/core/src/webgpu/index.ts'),
          import(root + '/modules/graph/src/gpu-core/gpu-command-graph.ts'),
          import('/four-bar.ts')
        ]);
      const device = await luma.createDevice({adapters: [webgpuAdapter], type: 'webgpu'});
      const trial = makeFourBar({ordered, payloadMass});
      const definition = new GPUCommandGraph(device);
      trial.addToGraph(definition);
      const graph = await definition.compileAsync();
      try {
        const measurements = [],
          snapshots = [];
        for (let step = 0; step < 560; step++) {
          const start = performance.now();
          if (step % 4 === 0) trial.command(fourBarTarget(step));
          const encoder = device.createCommandEncoder();
          graph.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          const fence = device.createFence();
          try {
            await fence.signaled;
          } finally {
            fence.destroy();
          }
          if (step >= 20) measurements.push(performance.now() - start);
          if ((step + 1) % 20 === 0)
            snapshots.push({step, ...(await trial.read())});
        }
        measurements.sort((first, second) => first - second);
        const initial = snapshots[0];
        const press = snapshots.find(snapshot => snapshot.step === 299);
        const released = snapshots.at(-1);
        const maximumClosure = Math.max(...snapshots.map(snapshot => snapshot.closure));
        const maximumTilt = Math.max(...snapshots.map(snapshot => snapshot.tilt));
        const qualified =
          snapshots.every(snapshot => snapshot.complete) &&
          maximumClosure < 0.003 &&
          maximumTilt < Math.PI / 180 &&
          Math.abs(press.payload[2] - 1.23) < 0.015 &&
          press.platform[2] - initial.platform[2] > 0.25 &&
          Math.abs(released.platform[2] - initial.platform[2]) < 0.03 &&
          Math.hypot(released.payload[0] - initial.payload[0], released.payload[1] - initial.payload[1]) < 0.06;
        return {
          device: device.info,
          ordered,
          payloadMass,
          qualified,
          maximumClosure,
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
