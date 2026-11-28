// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '../../src/gpu-core';
import {GPUArticulationTree} from '../../src/gpu-physics';
import {
  makeFourBar,
  fourBarTarget
} from '../../../../examples/experimental/gpu-rigid-body-world/four-bar';

it.for([
  3, 5
])('closes the linkage while pressing and releasing an offset %s kg box', async (payloadMass, context) => {
  const device = await getWebGPUTestDevice('core');
  if (!device) return context.skip('WebGPU unavailable');
  const trial = makeFourBar({payloadMass});
  const definition = new GPUCommandGraph(device);
  trial.addToGraph(definition);
  const graph = await definition.compileAsync();
  try {
    /** Dispatch four fixed steps per command update and sample the settled phase. */
    async function advance(start: number, end: number) {
      for (let step = start; step < end; step += 4) {
        trial.command(fourBarTarget(step));
        const encoder = device.createCommandEncoder();
        for (let substep = 0; substep < 4; substep++)
          graph.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
      }
      return trial.read();
    }
    const initial = await advance(0, 20);
    const press = await advance(20, 300);
    const released = await advance(300, 560);
    for (const state of [initial, press, released]) {
      expect(state.complete).toBe(true);
      expect(state.closure).toBeLessThan(0.002);
      expect(state.tilt).toBeLessThan(Math.PI / 180);
    }
    expect(press.payload[2]).toBeGreaterThan(1.21);
    expect(press.payload[2]).toBeLessThan(1.24);
    expect(press.support!).toBeGreaterThan(payloadMass * 9.81 + 30);
    expect(press.contacts).toBeGreaterThan(0);
    expect(Math.abs(released.support! - payloadMass * 9.81)).toBeLessThan(2);
    expect(Math.abs(released.platform[2] - initial.platform[2])).toBeLessThan(0.03);
  } finally {
    graph.destroy();
    trial.destroy();
  }
});

it('rejects a rank-deficient loop before publishing an accepted step', async context => {
  const device = await getWebGPUTestDevice('core');
  if (!device) return context.skip('WebGPU unavailable');
  const tree = new GPUArticulationTree({
    links: [
      {
        id: 'first',
        parent: -1,
        parentAnchor: [0, 0, 0],
        childAnchor: [0, 0, 1],
        mass: 1,
        inertia: [1, 1, 1],
        axis: [0, 1, 0]
      },
      {
        id: 'second',
        parent: -1,
        parentAnchor: [0, 0, 0],
        childAnchor: [0, 0, 1],
        mass: 1,
        inertia: [1, 1, 1],
        axis: [0, 1, 0]
      }
    ],
    loop: {
      first: {link: 'first', anchor: [0, 0, 0]},
      second: {link: 'second', anchor: [0, 0, 0]},
      normal: [0, 1, 0]
    }
  });
  const definition = new GPUCommandGraph(device);
  tree.addToGraph(definition);
  const graph = await definition.compileAsync();
  try {
    const encoder = device.createCommandEncoder();
    graph.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    expect(await tree.readSimulationDiagnostics()).toMatchObject({complete: false, steps: 0});
  } finally {
    graph.destroy();
    tree.destroy();
  }
});
