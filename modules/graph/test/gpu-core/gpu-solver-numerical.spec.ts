// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGraph, GPUCPUOperation, gpuGraph} from '@luma.gl/graph';
import {CPUData, CPUVector, GPUData} from '@luma.gl/graph/gpu-data';
import {
  createGPUConjugateGradientGraph,
  GPUGraphCompiler,
  GPUJacobiPCG,
  type GPUOperationLowering
} from '@luma.gl/graph/gpu-core';
import {BatchConformanceFixture} from './batch-conformance-utils';

for (const initial of [
  [0, 0],
  [0.5, 1],
  [1, 2]
]) {
  test(`Jacobi PCG solves chunked SPD with initial solution ${initial}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const fixture = new BatchConformanceFixture(device);
    // Duplicate entries on the first diagonal must sum, just as CSR SpMV does.
    const rowOffsets = fixture.column('offsets', 'uint32', [0, 3, 5], [1, 0, 2]);
    const columnIndices = fixture.column('columns', 'uint32', [0, 0, 1, 0, 1], [2, 3]);
    const values = fixture.column('values', 'float32', [2, 2, 1, 1, 3], [1, 4]);
    const rhs = fixture.column('rhs', 'float32', [6, 7], [1, 1]);
    const solution = fixture.column('solution', 'float32', initial, [0, 2]);
    const solver = new GPUJacobiPCG({
      rowOffsets,
      columnIndices,
      values,
      rhs,
      solution,
      columns: 2,
      iterations: 5,
      toleranceSquared: 1e-10
    });
    fixture.graph.add(solver);
    const result = solver.getResult();
    const initialNorm = fixture.capture('initial-norm', result.initialResidualSquared.view);
    const finalNorm = fixture.capture('final-norm', result.residualSquared.view);
    const breakdown = fixture.capture('breakdown', result.breakdown.view);
    const compiled = fixture.graph.compile();
    try {
      for (let run = 0; run < 2; run++) {
        const encoder = device.createCommandEncoder();
        compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        const actual = await fixture.read(solution);
        expect(actual[0]).toBeCloseTo(1, 5);
        expect(actual[1]).toBeCloseTo(2, 5);
        const expectedInitial =
          run === 0
            ? (6 - 4 * initial[0] - initial[1]) ** 2 + (7 - initial[0] - 3 * initial[1]) ** 2
            : 0;
        expect((await fixture.read(initialNorm))[0]).toBeCloseTo(expectedInitial, 5);
        expect((await fixture.read(finalNorm))[0]).toBeLessThanOrEqual(1e-10);
        expect(await fixture.read(breakdown)).toEqual([0]);
      }
    } finally {
      compiled.destroy();
      expect(fixture.buffers.every(buffer => !buffer.destroyed)).toBe(true);
      fixture.destroy();
    }
  });
}

for (const diagonal of [0, -1]) {
  test(`Jacobi PCG reports unusable diagonal ${diagonal} without poisoning solution`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const fixture = new BatchConformanceFixture(device);
    const rowOffsets = fixture.column('offsets', 'uint32', [0, 1], [2]);
    const columnIndices = fixture.column('columns', 'uint32', [0], [1]);
    const values = fixture.column('values', 'float32', [diagonal], [1]);
    const rhs = fixture.column('rhs', 'float32', [1], [1]);
    const solution = fixture.column('solution', 'float32', [0], [1]);
    const solver = new GPUJacobiPCG({
      rowOffsets,
      columnIndices,
      values,
      rhs,
      solution,
      columns: 1,
      iterations: 3
    });
    fixture.graph.add(solver);
    const breakdown = fixture.capture('breakdown', solver.getResult().breakdown.view);
    const compiled = fixture.graph.compile();
    try {
      const encoder = device.createCommandEncoder();
      compiled.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      expect(await fixture.read(solution)).toEqual([0]);
      expect(await fixture.read(breakdown)).toEqual([1]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

for (const initial of [
  [0, 0],
  [0.5, 1],
  [1, 2]
]) {
  test(`semantic CG solves independently chunked bindings from ${initial}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const buffers: Buffer[] = [];
    const chunk = (values: number[], format: 'float32' | 'uint32') => {
      const data = format === 'float32' ? new Float32Array(values) : new Uint32Array(values);
      const buffer = device.createBuffer({
        data,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      return new GPUData({buffer, format, length: values.length});
    };
    const solver = createGPUConjugateGradientGraph({
      size: 2,
      nonZeros: 4,
      maxIterations: 5,
      toleranceSquared: 1e-10
    });
    const solution = [chunk(initial.slice(0, 1), 'float32'), chunk(initial.slice(1), 'float32')];
    const compilation = new GPUGraphCompiler(device).compile(solver.program, {
      vectors: {
        [solver.rowOffsets.id]: [chunk([0], 'uint32'), chunk([2, 4], 'uint32')],
        [solver.columnIndices.id]: [chunk([0, 1, 0], 'uint32'), chunk([1], 'uint32')],
        [solver.values.id]: [chunk([4, 1], 'float32'), chunk([1, 3], 'float32')],
        [solver.rhs.id]: chunk([6, 7], 'float32'),
        [solver.solution.id]: solution
      }
    });
    const compiled = compilation.graph.compile();
    try {
      for (let run = 0; run < 2; run++) {
        const encoder = device.createCommandEncoder();
        compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        const actual = await Promise.all(
          solution.map(async data => {
            const bytes = await data.buffer.readAsync();
            return new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0];
          })
        );
        expect(actual[0]).toBeCloseTo(1, 5);
        expect(actual[1]).toBeCloseTo(2, 5);
      }
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  });
}

test('semantic CG reports zero curvature and preserves finite state', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const buffers: Buffer[] = [];
  const data = (values: number[], format: 'float32' | 'uint32') => {
    const buffer = device.createBuffer({
      data: format === 'float32' ? new Float32Array(values) : new Uint32Array(values),
      usage: Buffer.STORAGE | Buffer.COPY_SRC
    });
    buffers.push(buffer);
    return new GPUData({buffer, format, length: values.length});
  };
  const solver = createGPUConjugateGradientGraph({
    size: 1,
    nonZeros: 1,
    maxIterations: 3,
    toleranceSquared: 1e-10
  });
  const solution = data([0], 'float32');
  const compilation = new GPUGraphCompiler(device).compile(solver.program, {
    vectors: {
      [solver.rowOffsets.id]: data([0, 1], 'uint32'),
      [solver.columnIndices.id]: data([0], 'uint32'),
      [solver.values.id]: data([0], 'float32'),
      [solver.rhs.id]: data([1], 'float32'),
      [solver.solution.id]: solution
    }
  });
  const diagnostics = device.createBuffer({
    byteLength: 8,
    usage: Buffer.COPY_SRC | Buffer.COPY_DST
  });
  buffers.push(diagnostics);
  const target = compilation.graph.importBuffer(
    {id: 'diagnostics', byteLength: 8, usage: diagnostics.usage},
    diagnostics
  );
  for (const [index, scalar] of [solver.residualSquared, solver.breakdown].entries()) {
    const view = compilation.scalars.get(scalar.id)!.view;
    compilation.graph.addCopyPass({
      id: `diagnostic-${index}`,
      resources: [
        {buffer: view, usage: 'copy-source'},
        {buffer: target, usage: 'copy-destination'}
      ],
      compile: () => ({
        encode: ({commandEncoder, getBuffer}) =>
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: getBuffer(view),
            sourceOffset: view.byteOffset,
            destinationBuffer: getBuffer(target),
            destinationOffset: index * 4,
            size: 4
          })
      })
    });
  }
  const compiled = compilation.graph.compile();
  try {
    const encoder = device.createCommandEncoder();
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    const bytes = await diagnostics.readAsync();
    const readback = new DataView(bytes.buffer, bytes.byteOffset);
    expect(readback.getFloat32(0, true)).toBe(1);
    expect(readback.getUint32(4, true)).toBe(1);
    const solutionBytes = await solution.buffer.readAsync();
    expect(new Float32Array(solutionBytes.buffer, solutionBytes.byteOffset, 1)[0]).toBe(0);
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

test('named Jacobi-PCG ports compose with a CPU consumer across repeated GPUGraph runs', async ({
  skip
}) => {
  const device = await getWebGPUTestDevice();
  if (!device) return skip();
  const graph = new GPUGraph();
  const ports = {
    rowOffsets: 'uint32',
    columnIndices: 'uint32',
    values: 'float32',
    rhs: 'float32',
    initialSolution: 'float32'
  } as const;
  for (const [name, format] of Object.entries(ports)) {
    graph.addInput(name, gpuGraph.vector(format));
  }
  graph.add({
    id: 'solve',
    type: 'jacobi-pcg',
    contract: {
      outputs: {solution: gpuGraph.vector('float32')},
      values: [
        ...Object.keys(ports).map(value => ({value, access: 'read' as const})),
        {value: 'solution', access: 'write', shape: {like: 'rhs'}}
      ]
    }
  });
  const lowering: GPUOperationLowering = {
    type: 'jacobi-pcg',
    lower: (operation, context) => {
      const initialSolution = context.resolveVector<'float32'>('initialSolution').data[0];
      const solution = context.resolveVector<'float32'>('solution').data[0];
      context.graph.addCopyPass({
        id: operation.id + '-initialize',
        resources: [
          {buffer: initialSolution, usage: 'copy-source'},
          {buffer: solution, usage: 'copy-destination'}
        ],
        compile: () => ({
          encode: ({commandEncoder, getBuffer}) =>
            commandEncoder.copyBufferToBuffer({
              sourceBuffer: getBuffer(initialSolution),
              sourceOffset: initialSolution.byteOffset,
              destinationBuffer: getBuffer(solution),
              destinationOffset: solution.byteOffset,
              size: solution.length * Float32Array.BYTES_PER_ELEMENT
            })
        })
      });
      const rhs = context.resolveVector<'float32'>('rhs');
      context.graph.add(
        new GPUJacobiPCG({
          id: operation.id,
          rowOffsets: context.resolveVector<'uint32'>('rowOffsets'),
          columnIndices: context.resolveVector<'uint32'>('columnIndices'),
          values: context.resolveVector<'float32'>('values'),
          rhs,
          solution: context.resolveVector<'float32'>('solution'),
          columns: rhs.length,
          iterations: 5
        })
      );
    }
  };
  graph.add(
    new GPUCPUOperation({
      id: 'summarize',
      inputs: {solution: 'solution'},
      outputs: {sum: gpuGraph.vector('float32')},
      execute: ({inputs}) => {
        const solution = inputs.solution;
        if (typeof solution === 'number') throw new Error('Expected a solution vector');
        const values = solution.pack();
        expect(values[0] * 2).toBeCloseTo(values[1], 5);
        return {sum: new Float32Array([values[0] + values[1]])};
      }
    })
  );
  const compiled = graph.compile<{sum: CPUVector}>(device, {
    lowerings: [lowering],
    outputs: ['sum'],
    outputLocations: {sum: 'cpu'}
  });
  function input<Format extends 'float32' | 'uint32'>(
    format: Format,
    buffer: Float32Array | Uint32Array
  ) {
    return new CPUVector({format, data: [new CPUData({format, buffer, length: buffer.length})]});
  }
  try {
    for (const factor of [1, 2]) {
      const result = await compiled.run({
        inputs: {
          rowOffsets: input('uint32', new Uint32Array([0, 2, 4])),
          columnIndices: input('uint32', new Uint32Array([0, 1, 0, 1])),
          values: input('float32', new Float32Array([4, 1, 1, 3])),
          rhs: input('float32', new Float32Array([6 * factor, 7 * factor])),
          initialSolution: input('float32', new Float32Array([0, 0]))
        }
      });
      const {buffer, byteOffset} = result.sum.data[0];
      expect(new Float32Array(buffer, byteOffset, 1)[0]).toBeCloseTo(3 * factor, 5);
    }
  } finally {
    compiled.destroy();
  }
});
