// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '../../src/gpu-core';
import {GPUArticulationTree} from '../../src/gpu-physics';
import {makeLandingRig} from '../../../../examples/experimental/gpu-rigid-body-world/landing-rig';

/** Masses and spherical inertias of the joint and chassis in the free-flight fixture. */
const MOMENTUM_FIXTURE = [
  {mass: 1, inertia: 0.19},
  {mass: 3, inertia: 0.7}
];

/** Compute world linear momentum and angular momentum about the total COM. */
function measureMomentum(poses: Awaited<ReturnType<GPUArticulationTree['readState']>>) {
  const center = [0, 0, 0];
  const linear = [0, 0, 0];
  const angular = [0, 0, 0];
  poses.forEach((pose, index) => {
    for (let component = 0; component < 3; component++) {
      center[component] += (pose.position[component] * MOMENTUM_FIXTURE[index].mass) / 4;
      linear[component] += pose.velocity[component] * MOMENTUM_FIXTURE[index].mass;
    }
  });
  poses.forEach((pose, index) => {
    const {mass, inertia} = MOMENTUM_FIXTURE[index];
    const [offsetX, offsetY, offsetZ] = pose.position.map(
      (position, axis) => position - center[axis]
    );
    const [velocityX, velocityY, velocityZ] = pose.velocity;
    const orbital = [
      offsetY * velocityZ - offsetZ * velocityY,
      offsetZ * velocityX - offsetX * velocityZ,
      offsetX * velocityY - offsetY * velocityX
    ];
    for (let component = 0; component < 3; component++) {
      angular[component] += mass * orbital[component] + inertia * pose.angularVelocity[component];
    }
  });
  return {linear, angular};
}

it('conserves free-flight momentum under internal motor torque and resets the floating chassis', async context => {
  const device = await getWebGPUTestDevice('core');
  if (!device) return context.skip('WebGPU unavailable');
  const tree = new GPUArticulationTree({
    gravity: [0, 0, 0],
    timeStep: 1 / 120,
    links: [
      {
        id: 'arm',
        parent: -1,
        parentAnchor: [0.4, 0.13, 0],
        childAnchor: [-0.35, 0, 0],
        mass: 1,
        inertia: [0.19, 0.19, 0.19],
        axis: [0, 1, 0]
      }
    ],
    floatingBase: {
      id: 'chassis',
      position: [0, 0, 0],
      mass: 3,
      inertia: [0.7, 0.7, 0.7],
      velocity: [0.15, -0.12, 0.05],
      angularVelocity: [0.12, -0.18, 0.1]
    }
  });
  const definition = new GPUCommandGraph(device);
  tree.addToGraph(definition);
  const graph = await definition.compileAsync();
  try {
    /** Step the same compiled graph without per-step host readback. */
    function advance(steps: number) {
      const encoder = device.createCommandEncoder();
      for (let step = 0; step < steps; step++) graph.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
    }
    advance(1);
    const initialPose = await tree.readState();
    const initial = measureMomentum(initialPose);
    tree.setDrive('arm', {angle: 0.8, stiffness: 22, damping: 5, maximumTorque: 7});
    advance(119);
    const finalPose = await tree.readState();
    const final = measureMomentum(finalPose);
    expect(Math.abs(finalPose[1].orientation[1] - initialPose[1].orientation[1])).toBeGreaterThan(
      0.05
    );
    expect(
      Math.hypot(...final.linear.map((value, axis) => value - initial.linear[axis]))
    ).toBeLessThan(0.025);
    expect(
      Math.hypot(...final.angular.map((value, axis) => value - initial.angular[axis]))
    ).toBeLessThan(0.025);
    expect(await tree.readSimulationDiagnostics()).toMatchObject({complete: true, steps: 120});
    tree.reset();
    advance(1);
    expect(await tree.readState()).toEqual(initialPose);
  } finally {
    graph.destroy();
    tree.destroy();
  }
});

it.for([
  3, 5
])('lands an offset %s kg payload and levels the floating chassis', async (payloadMass, context) => {
  const device = await getWebGPUTestDevice('core');
  if (!device) return context.skip('WebGPU unavailable');
  const trial = makeLandingRig({payloadMass});
  const definition = new GPUCommandGraph(device);
  trial.addToGraph(definition);
  const graph = await definition.compileAsync();
  try {
    /** Advance four fixed steps between servo updates. */
    async function advance(start: number, end: number) {
      for (let step = start; step < end; step += 4) {
        trial.command(step);
        const encoder = device.createCommandEncoder();
        for (let substep = 0; substep < 4; substep++)
          graph.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
      }
      return trial.read();
    }
    const initial = await advance(0, 4);
    const landing = await advance(4, 56);
    const settled = await advance(56, 360);
    expect(landing.tilt).toBeGreaterThan(0.05);
    expect(settled.complete).toBe(true);
    expect(settled.tilt).toBeLessThan(Math.PI / 180);
    expect(settled.position[2]).toBeCloseTo(0.84, 1);
    expect(Math.abs(settled.feet[0][0] - initial.feet[0][0])).toBeLessThan(0.06);
    expect(Math.abs(settled.feet[1][0] - initial.feet[1][0])).toBeLessThan(0.06);
    expect(settled.loads![0]).toBeGreaterThan(20);
    expect(settled.loads![1]).toBeGreaterThan(20);
    expect(
      Math.abs(settled.loads![0] + settled.loads![1] - (trial.totalBaseMass + 1.3) * 9.81)
    ).toBeLessThan(2);
  } finally {
    graph.destroy();
    trial.destroy();
  }
});
