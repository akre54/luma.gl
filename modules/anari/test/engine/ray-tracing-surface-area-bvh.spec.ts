// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RayTracingSceneRenderer, type RayTracingSceneRenderOptions} from '@luma.gl/anari';
import {Buffer} from '@luma.gl/core';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {Geometry} from '@luma.gl/engine';
import {Matrix4} from '@math.gl/core/matrix';
import {expect, it, vi} from 'vitest';

it('traces and picks the same placements with a surface-area build and retained refit', async () => {
  if (!webgpuAdapter.isSupported()) return;
  const device = await webgpuAdapter.create({featureLevel: 'core'});
  const renderer = new RayTracingSceneRenderer(device);
  const framebuffer = device.createFramebuffer({
    width: 32,
    height: 32,
    colorAttachments: ['rgba8unorm']
  });
  const reportError = vi.spyOn(device, 'reportError');
  try {
    /** An uneven leaf population and a mixture of overlapping and separated bounds. */
    const placements = [-2.9, -1.8, -0.8, 0, 0.9, 2, 3.5];
    /** The same scene feeds both builders. Sampling and adaptation are fixed. */
    const scene: RayTracingSceneRenderOptions = {
      id: 'tlas',
      framebuffer,
      width: 32,
      height: 32,
      camera: {
        position: [0, 0, 5],
        projectionMatrix: new Matrix4().perspective({
          fovy: Math.PI / 3,
          aspect: 1,
          near: 0.1,
          far: 20
        }),
        viewMatrix: new Matrix4().lookAt({eye: [0, 0, 5], center: [0, 0, 0], up: [0, 1, 0]})
      },
      primitives: {sphere: {type: 'sphere', radius: 0.46}},
      surfaces: [
        {
          id: 'sphere',
          geometry: new Geometry({
            attributes: {POSITION: {size: 3, value: new Float32Array([0, 0, 0])}}
          }),
          material: {id: 'matte', uniforms: {baseColorFactor: [0.5, 0.25, 0.1, 1]}},
          transforms: placements.map(x => new Matrix4().translate([x, 0, 0]))
        }
      ],
      lights: [{type: 'directional', direction: [0, -1, -1], color: [1, 1, 1], intensity: 2}],
      resolutionScale: 1,
      adaptiveResolution: false,
      temporalReprojection: false,
      progressive: false,
      samplingFrameIndex: 0
    };
    /** Move the central sphere far enough to disocclude the center ray. */
    const movedSurface = {
      ...scene.surfaces[0],
      transforms: placements.map(x => new Matrix4().translate([x, x === 0 ? 1.7 : 0, 0]))
    };

    /** Reads the center identifier from the independent full-resolution channel shader. */
    async function readCenterIdentifier(id: string): Promise<number> {
      const channels = renderer.captureChannels(id)!;
      const region = {x: 16, y: 16, width: 1, height: 1};
      const buffer = device.createBuffer({
        byteLength: channels.identifierTexture.computeMemoryLayout(region).byteLength,
        usage: Buffer.COPY_DST | Buffer.MAP_READ
      });
      try {
        channels.identifierTexture.readBuffer(region, buffer);
        const bytes = await buffer.readAsync();
        return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true);
      } finally {
        buffer.destroy();
        channels.destroy();
      }
    }

    /** Capture color and picking at the initial and refitted poses. */
    async function capture(tlasBuilder: 'morton' | 'surfaceArea') {
      const id = tlasBuilder;
      const initial = renderer.render({...scene, id, tlasBuilder});
      device.submit();
      const first = await renderer.readRadiance(id);
      const initialPick = await renderer.pick(id, 16, 16);
      expect(await readCenterIdentifier(id)).toBe(4);
      const moved = renderer.render({...scene, id, tlasBuilder, surfaces: [movedSurface]});
      device.submit();
      const second = await renderer.readRadiance(id);
      const movedPick = await renderer.pick(id, 16, 16);
      expect(await readCenterIdentifier(id)).toBe(0);
      expect(initial.rayTracing?.graph?.acceleration).toBeDefined();
      expect(moved.rayTracing?.graph?.refit).toBeDefined();
      expect(moved.rayTracing?.graph?.acceleration).toBeUndefined();
      expect(initialPick?.placementId).toBe('sphere:3');
      expect(movedPick).toBeNull();
      expect(first?.data).not.toEqual(second?.data);
      return {first: first?.data, second: second?.data};
    }
    expect(await capture('surfaceArea')).toEqual(await capture('morton'));
    expect(reportError).not.toHaveBeenCalled();
  } finally {
    renderer.destroy();
    framebuffer.destroy();
    reportError.mockRestore();
    device.destroy();
  }
});
