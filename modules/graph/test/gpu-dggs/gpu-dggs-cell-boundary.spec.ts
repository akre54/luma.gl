// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUA5CellBoundary} from '@luma.gl/graph/gpu-a5';
import {GPUH3CellBoundary} from '@luma.gl/graph/gpu-h3';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {getDGGSCellBoundaryCapacity} from '@luma.gl/graph/gpu-dggs';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {cellToBoundary as getA5Boundary} from 'a5-js';
import {cellToBoundary as getH3Boundary} from 'h3-js';
import {expect, it} from 'vitest';

/** Includes a supported hexagon and an unsupported pentagon. */
const H3_KEYS = [0x89283080c83ffffn, 0x81083ffffffffffn];
/** Includes an A5 triangle, a pentagon, and the reserved zero key. */
const A5_KEYS = [0x0500000000000000n, 0x1a38000000000000n, 0n];

it('decodes native resident H3 and A5 views with sliced buffers and per-cell counts', async context => {
  const device = await getWebGPUTestDevice();
  if (!device) return context.skip();
  for (const family of ['h3', 'a5'] as const) {
    const h3 = family === 'h3';
    const keys = h3 ? H3_KEYS : A5_KEYS;
    const capacity = getDGGSCellBoundaryCapacity(family);
    const format = h3 ? 'float32x2' : 'float32x4';
    const components = h3 ? 2 : 4;
    // Use the geographic tolerances from the existing Arrow boundary parity suite.
    const tolerance = h3 ? 0.03 : 0.01;
    const wordOrder = h3 ? 'little-endian' : 'high-low';
    const Boundary = h3 ? GPUH3CellBoundary : GPUA5CellBoundary;
    const words = new Uint32Array(1 + keys.length * 2);
    keys.forEach((key, index) => {
      const lower = Number(key & 0xffffffffn);
      const upper = Number(key >> 32n);
      words.set(h3 ? [lower, upper] : [upper, lower], 1 + index * 2);
    });
    const cellsBuffer = device.createBuffer({data: words, usage: Buffer.STORAGE});
    const verticesBuffer = device.createBuffer({
      data: new Float32Array(1 + keys.length * capacity * components).fill(-1),
      usage: Buffer.STORAGE | Buffer.COPY_SRC
    });
    const countsBuffer = device.createBuffer({
      data: new Uint32Array(1 + keys.length).fill(100),
      usage: Buffer.STORAGE | Buffer.COPY_SRC
    });
    const graph = new GPUCommandGraph<void>(device);
    const cellsHandle = graph.importBuffer(
      {id: 'cells', byteLength: cellsBuffer.byteLength, usage: cellsBuffer.usage},
      cellsBuffer
    );
    const verticesHandle = graph.importBuffer(
      {id: 'vertices', byteLength: verticesBuffer.byteLength, usage: verticesBuffer.usage},
      verticesBuffer
    );
    const countsHandle = graph.importBuffer(
      {id: 'counts', byteLength: countsBuffer.byteLength, usage: countsBuffer.usage},
      countsBuffer
    );
    const cells = graph.createDataView(cellsHandle, {
      format: 'uint32x2',
      length: keys.length,
      byteOffset: 4
    });
    const vertices = graph.createDataView(verticesHandle, {
      format,
      length: keys.length * capacity,
      byteOffset: 4
    });
    const counts = graph.createDataView(countsHandle, {
      format: 'uint32',
      length: keys.length,
      byteOffset: 4
    });
    const shorter = graph.createDataView(verticesHandle, {format, length: capacity, byteOffset: 4});
    expect(() => new Boundary({cells, vertices: shorter})).toThrow(/output lengths/);
    new Boundary({cells, vertices, counts, wordOrder}).addToGraph(graph);
    const compiled = graph.compile();
    try {
      const commandEncoder = device.createCommandEncoder();
      compiled.encode(commandEncoder, {parameters: undefined});
      device.submit(commandEncoder.finish());
      const [vertexBytes, countBytes] = await Promise.all([
        verticesBuffer.readAsync(),
        countsBuffer.readAsync()
      ]);
      const values = new Float32Array(
        vertexBytes.buffer,
        vertexBytes.byteOffset,
        vertexBytes.byteLength / 4
      );
      const lengths = new Uint32Array(
        countBytes.buffer,
        countBytes.byteOffset,
        countBytes.byteLength / 4
      );
      expect(values[0]).toBe(-1);
      expect([...lengths]).toEqual(h3 ? [100, 7, 0] : [100, 4, 6, 0]);
      for (let rowIndex = 0; rowIndex < keys.length; rowIndex++) {
        const key = keys[rowIndex];
        const expected =
          rowIndex === keys.length - 1
            ? []
            : h3
              ? getH3Boundary(key.toString(16), true)
              : getA5Boundary(key, {segments: 1, closedRing: true});
        for (let vertexIndex = 0; vertexIndex < capacity; vertexIndex++) {
          const [longitude, latitude] = expected[vertexIndex] ?? expected[0] ?? [0, 0];
          const offset = 1 + (rowIndex * capacity + vertexIndex) * components;
          if (expected.length === 0) {
            expect([...values.slice(offset, offset + components)]).toEqual(
              new Array(components).fill(0)
            );
          } else {
            expect(Math.abs(values[offset] - longitude)).toBeLessThanOrEqual(tolerance);
            expect(Math.abs(values[offset + 1] - latitude)).toBeLessThanOrEqual(tolerance);
            if (vertexIndex >= expected.length - 1) {
              const firstOffset = 1 + rowIndex * capacity * components;
              expect([...values.slice(offset, offset + components)]).toEqual([
                ...values.slice(firstOffset, firstOffset + components)
              ]);
            }
          }
          if (!h3) expect([...values.slice(offset + 2, offset + 4)]).toEqual([0, 0]);
        }
      }
    } finally {
      compiled.destroy();
      expect(cellsBuffer.destroyed).toBe(false);
      cellsBuffer.destroy();
      verticesBuffer.destroy();
      countsBuffer.destroy();
    }
  }
});
