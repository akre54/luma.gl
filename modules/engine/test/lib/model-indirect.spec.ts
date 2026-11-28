// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer, Texture, type Device, type Framebuffer} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGLTestDevice, getWebGPUTestDevice} from '@luma.gl/core/test-utils';

const PIXEL_COUNT = 4;

/** One instance per pixel column: instance `i` covers pixel `i` of a 4x1 target. */
const INSTANCE_PER_PIXEL_WGSL = /* WGSL */ `
@vertex fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> @builtin(position) vec4<f32> {
  // Two triangles covering one pixel column: corners (0,0) (1,0) (0,1) (1,1).
  let corner = vec2<f32>(f32(vertexIndex & 1u), f32((vertexIndex >> 1u) & 1u));
  let x = (f32(instanceIndex) + corner.x) / ${PIXEL_COUNT}.0 * 2.0 - 1.0;
  let y = corner.y * 2.0 - 1.0;
  return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment fn fragmentMain() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
`;

it('Model#drawIndirect consumes a culled instance count in the same command graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const {framebuffer, texture} = createTarget(device);
  const distances = device.createBuffer({
    data: new Float32Array([-1, 0.5, 2, 0.25]),
    usage: Buffer.STORAGE
  });
  const record = device.createBuffer({
    data: new Uint32Array([4, 0, 0, 0]),
    usage: Buffer.STORAGE | Buffer.INDIRECT
  });
  const cull = new Kernel(device, {
    source: `
      @group(0) @binding(0) var<storage, read> distances: array<f32>;
      @group(0) @binding(1) var<storage, read_write> record: array<u32>;
      @compute @workgroup_size(1) fn main() {
        var count = 0u;
        for (var i = 0u; i < arrayLength(&distances); i++) {
          if (distances[i] >= 0.0 && distances[i] <= 1.0) { count += 1u; }
        }
        record[1] = count;
      }`,
    shaderLayout: {
      bindings: [
        {name: 'distances', type: 'read-only-storage', group: 0, location: 0},
        {name: 'record', type: 'storage', group: 0, location: 1}
      ]
    }
  });
  const model = new Model(device, {
    source: INSTANCE_PER_PIXEL_WGSL,
    topology: 'triangle-strip',
    vertexCount: 4,
    isInstanced: true,
    instanceCount: 0
  });
  const graph = new GPUCommandGraph(device);
  const input = graph.importBuffer(
    {id: 'distances', byteLength: distances.byteLength, usage: distances.usage},
    distances
  );
  const command = graph.importBuffer(
    {id: 'command', byteLength: record.byteLength, usage: record.usage},
    record
  );
  graph.addComputePass({
    id: 'cull',
    resources: [
      {buffer: input, usage: 'storage-read'},
      {buffer: command, usage: 'storage-write'}
    ],
    compile: () => ({
      encode: ({computePass, getBuffer}) => {
        cull.dispatch(computePass, {
          bindings: {distances: getBuffer(input), record: getBuffer(command)},
          x: 1
        });
      }
    })
  });
  graph.addRenderPass({
    id: 'draw-visible',
    resources: [{buffer: command, usage: 'indirect'}],
    compile: () => ({
      getRenderPassProps: () => ({framebuffer, clearColor: [0, 0, 0, 0]}),
      encode: ({renderPass, getBuffer}) => {
        expect(model.drawIndirect(renderPass, getBuffer(command))).toBe(true);
      }
    })
  });
  const compiled = graph.compile();
  try {
    compiled.encode(device.commandEncoder, {parameters: undefined});
    device.submit();
    expect(await readRedChannel(texture)).toEqual([255, 255, 0, 0]);
    const pass = device.beginRenderPass({framebuffer, clearColor: [0, 0, 0, 0]});
    expect(model.draw(pass)).toBe(true);
    pass.end();
    device.submit();
    expect(await readRedChannel(texture)).toEqual([0, 0, 0, 0]);
  } finally {
    compiled.destroy();
    cull.destroy();
    model.destroy();
    distances.destroy();
    record.destroy();
    framebuffer.destroy();
    texture.destroy();
  }
});

it('Model#drawIndirect reads indexed records at an offset', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;

  const {framebuffer, texture} = createTarget(device);
  const indexBuffer = device.createBuffer({
    usage: Buffer.INDEX,
    data: new Uint16Array([0, 1, 2, 2, 1, 3])
  });
  // Record at a non-zero offset inside a larger buffer; the instanceCount word is pre-seeded.
  const drawRecords = device.createBuffer({
    usage: Buffer.INDIRECT | Buffer.COPY_DST | Buffer.COPY_SRC,
    data: new Uint32Array([0, 0, 0, 0, 99, 6, 3, 0, 0, 0])
  });
  const model = new Model(device, {
    id: 'indexed-indirect-instance-count',
    source: INSTANCE_PER_PIXEL_WGSL,
    topology: 'triangle-list',
    indexBuffer,
    isInstanced: true
  });

  const renderPass = device.beginRenderPass({framebuffer, clearColor: [0, 0, 0, 0]});
  model.drawIndirect(renderPass, drawRecords, 20);
  renderPass.end();
  device.submit();

  expect(await readRedChannel(texture), 'indexed indirect draw used the GPU count').toEqual([
    255, 255, 255, 0
  ]);
  const words = await readWords(drawRecords);
  expect(
    words.slice(5),
    'indexCount, instanceCount, firstIndex, baseVertex, firstInstance'
  ).toEqual([6, 3, 0, 0, 0]);
  expect(words[4], 'words outside the record are untouched').toBe(99);

  model.destroy();
  drawRecords.destroy();
  indexBuffer.destroy();
  framebuffer.destroy();
  texture.destroy();
});

it('Model#drawIndirect throws on WebGL', async () => {
  const device = await getWebGLTestDevice();
  if (!device) return;

  const model = new Model(device, {
    vs: `#version 300 es
void main() { gl_Position = vec4(0.0); }`,
    fs: `#version 300 es
precision highp float;
out vec4 fragColor;
void main() { fragColor = vec4(1.0); }`,
    vertexCount: 3
  });
  const buffer = device.createBuffer({byteLength: 16});

  const {framebuffer, texture} = createTarget(device);
  const pass = device.beginRenderPass({framebuffer});
  expect(() => model.drawIndirect(pass, buffer)).toThrow();
  pass.end();
  framebuffer.destroy();
  texture.destroy();

  model.destroy();
  buffer.destroy();
});

it('Model#drawIndirect rejects a buffer without INDIRECT usage', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;

  const model = new Model(device, {
    source: INSTANCE_PER_PIXEL_WGSL,
    topology: 'triangle-strip',
    vertexCount: 4
  });
  const buffer = device.createBuffer({byteLength: 16, usage: Buffer.COPY_DST});

  const {framebuffer, texture} = createTarget(device);
  const pass = device.beginRenderPass({framebuffer});
  expect(() => model.drawIndirect(pass, buffer)).toThrow();
  pass.end();
  framebuffer.destroy();
  texture.destroy();

  model.destroy();
  buffer.destroy();
});

/** One instanced point per `positions` row, so each vector chunk is a separate batch. */
const INSTANCE_POSITIONS_WGSL = /* WGSL */ `
@vertex fn vertexMain(@location(0) positions: vec2<f32>) -> @builtin(position) vec4<f32> {
  return vec4<f32>(positions, 0.0, 1.0);
}

@fragment fn fragmentMain() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
`;

it('Model#drawIndirect rejects draws that span multiple vector batches', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;

  const {framebuffer, texture} = createTarget(device);
  const makePositions = (chunkLengths: number[]) =>
    new GPUVector<'float32x2'>({
      type: 'data',
      name: 'positions',
      format: 'float32x2',
      data: chunkLengths.map(
        length =>
          new GPUData({
            buffer: device.createBuffer({byteLength: length * 8, usage: Buffer.VERTEX}),
            format: 'float32x2',
            length,
            ownsBuffer: true
          })
      ),
      ownsData: true
    });
  const twoBatches = makePositions([2, 3]);
  const drawRecord = device.createBuffer({
    byteLength: 16,
    usage: Buffer.INDIRECT | Buffer.COPY_DST
  });
  const model = new Model(device, {
    id: 'indirect-multi-batch',
    source: INSTANCE_POSITIONS_WGSL,
    vertexEntryPoint: 'vertexMain',
    fragmentEntryPoint: 'fragmentMain',
    topology: 'point-list',
    bufferLayout: [{name: 'positions', format: 'float32x2', stepMode: 'instance'}],
    attributeVectors: {positions: twoBatches},
    isInstanced: true,
    vertexCount: 1
  });
  const renderPass = device.beginRenderPass({framebuffer});
  try {
    expect(() => model.drawIndirect(renderPass, drawRecord), 'bound multi-batch vectors').toThrow();
    expect(model.draw(renderPass), 'direct draws still split batches').toBe(true);
  } finally {
    renderPass.end();
    model.destroy();
    twoBatches.destroy();
    drawRecord.destroy();
    framebuffer.destroy();
    texture.destroy();
  }
});

function createTarget(device: Device): {framebuffer: Framebuffer; texture: Texture} {
  const texture = device.createTexture({
    width: PIXEL_COUNT,
    height: 1,
    format: 'rgba8unorm',
    usage: Texture.RENDER_ATTACHMENT | Texture.COPY_SRC
  });
  const framebuffer = device.createFramebuffer({
    width: PIXEL_COUNT,
    height: 1,
    colorAttachments: [texture]
  });
  return {framebuffer, texture};
}

async function readRedChannel(texture: Texture): Promise<number[]> {
  const layout = texture.computeMemoryLayout({width: PIXEL_COUNT, height: 1});
  const buffer = texture.device.createBuffer({
    byteLength: layout.byteLength,
    usage: Buffer.COPY_DST | Buffer.MAP_READ
  });
  try {
    texture.readBuffer({width: PIXEL_COUNT, height: 1}, buffer);
    const bytes = await buffer.readAsync(0, layout.byteLength);
    return Array.from({length: PIXEL_COUNT}, (_, pixel) => bytes[pixel * 4]);
  } finally {
    buffer.destroy();
  }
}

async function readWords(buffer: Buffer): Promise<number[]> {
  const bytes = await buffer.readAsync();
  return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
}
