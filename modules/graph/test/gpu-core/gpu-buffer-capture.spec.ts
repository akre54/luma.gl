// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, GPUBufferCapture} from '@luma.gl/graph/gpu-core';

test.each([4, 8])('capture preserves intermediate rows with byte stride %s', async byteStride => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const snapshot = device.createBuffer({
    data: new Uint32Array(8).fill(99),
    usage: Buffer.COPY_DST | Buffer.COPY_SRC
  });
  const graph = new GPUCommandGraph(device);
  const source = graph.createDataView(
    graph.createTransientBuffer({
      id: 'source',
      byteLength: 32,
      usage: Buffer.COPY_SRC | Buffer.COPY_DST
    }),
    {format: 'uint32', length: 3, byteOffset: 4, byteStride}
  );
  const output = graph.createDataView(
    graph.importBuffer(
      {id: 'snapshot', byteLength: snapshot.byteLength, usage: snapshot.usage},
      snapshot
    ),
    {format: 'uint32', length: 3, byteOffset: 8, byteStride}
  );
  const staging = device.createBuffer({
    data: new Uint32Array([0, 2, 0, 4, 0, 6, 0, 0]),
    usage: Buffer.COPY_SRC | Buffer.COPY_DST
  });
  graph.addCopyPass({
    id: 'initialize',
    resources: [{buffer: source, usage: 'copy-destination'}],
    compile: () => ({
      encode: ({commandEncoder, getBuffer}) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: staging,
          destinationBuffer: getBuffer(source),
          size: 32
        })
    })
  });
  graph.add(new GPUBufferCapture({id: 'capture', source, output}));
  const zeros = device.createBuffer({data: new Uint32Array(8), usage: Buffer.COPY_SRC});
  graph.addCopyPass({
    id: 'overwrite',
    resources: [{buffer: source, usage: 'copy-destination'}],
    compile: () => ({
      encode: ({commandEncoder, getBuffer}) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: zeros,
          destinationBuffer: getBuffer(source),
          size: 32
        })
    })
  });
  const compiled = graph.compile();
  try {
    expect(compiled.preflight.nodes.find(node => node.id === 'capture')).toMatchObject({
      commandCount: byteStride === 4 ? 1 : 3,
      readByteLength: 12,
      writeByteLength: 12
    });
    for (const multiplier of [1, 3]) {
      const values = new Uint32Array(8);
      for (let row = 0; row < 3; row++)
        values[1 + (row * byteStride) / 4] = (row + 1) * 2 * multiplier;
      staging.write(values);
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      const bytes = await snapshot.readAsync();
      const expected = new Array<number>(8).fill(99);
      for (let row = 0; row < 3; row++)
        expected[2 + (row * byteStride) / 4] = (row + 1) * 2 * multiplier;
      expect(Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, 8))).toEqual(expected);
    }
  } finally {
    compiled.destroy();
    staging.destroy();
    zeros.destroy();
    snapshot.destroy();
  }
});
