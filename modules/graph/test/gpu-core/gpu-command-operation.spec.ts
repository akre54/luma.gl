// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGraph, GPUCPUOperation, GPUCommandOperation, gpuGraph} from '@luma.gl/graph';
import {GPUScan} from '@luma.gl/graph/gpu-core';

test('named command ports compose CPU inputs, GPU primitives, and CPU consumers across runs', async ({
  skip
}) => {
  const device = await getWebGPUTestDevice();
  if (!device) return skip();
  const graph = new GPUGraph();
  graph.addInput('length', gpuGraph.scalar('uint32'));
  graph.add(
    new GPUCPUOperation({
      id: 'values',
      inputs: {length: 'length'},
      outputs: {values: gpuGraph.vector('uint32')},
      execute: ({inputs}) => {
        const length = Number(inputs.length);
        const firstLength = Math.min(length, 2);
        return {
          values: {
            chunks: [
              {values: new Uint32Array(firstLength).fill(2)},
              {values: new Uint32Array(0)},
              {values: new Uint32Array(length - firstLength).fill(2)}
            ]
          }
        };
      }
    })
  );
  graph.add(
    new GPUCommandOperation({
      id: 'prefix',
      ports: {constructor: 'values', output: 'prefixes'},
      contract: {
        outputs: {prefixes: gpuGraph.vector('uint32')},
        values: [
          {value: 'values', access: 'read'},
          {value: 'prefixes', access: 'write', shape: {like: 'values'}}
        ]
      },
      createNodes: ({resolveVector}) =>
        new GPUScan({
          id: 'prefix-kernel',
          input: resolveVector<'uint32'>('constructor'),
          output: resolveVector<'uint32'>('output')
        })
    })
  );
  graph.add(
    new GPUCPUOperation({
      id: 'consume',
      inputs: {values: 'prefixes'},
      outputs: {result: gpuGraph.vector('uint32')},
      execute: ({inputs}) => {
        const values = inputs.values;
        if (typeof values === 'number') throw new Error('Expected vector');
        return {result: values.pack()};
      }
    })
  );
  const compiled = graph.compile(device, {outputs: ['result']});
  try {
    for (const length of [3, 5, 0]) {
      const outputs = await compiled.run({inputs: {length}});
      try {
        const actual: number[] = [];
        for (const chunk of outputs.result.data) {
          if (!chunk.length) continue;
          const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
          actual.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
        }
        expect(actual).toEqual(Array.from({length}, (_, index) => index * 2));
      } finally {
        outputs.result.destroy();
      }
    }
  } finally {
    compiled.destroy();
  }
});
