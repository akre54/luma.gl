// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import type {GPUCommandGraph, GraphDataView} from './gpu-command-graph';
import {createGPUCopyCommandNode, type GPUCommandNode} from './gpu-command-node';

export type GPUBufferCaptureProps<Format extends GPUVectorFormat = GPUVectorFormat> = {
  id: string;
  source: GraphDataView<Format>;
  /** Caller-owned snapshot storage, with the same row format and length. */
  output: GraphDataView<Format>;
};

/** Explicit GPU snapshot at this point in graph order. Each encode replaces the snapshot. */
export class GPUBufferCapture<Format extends GPUVectorFormat = GPUVectorFormat> {
  readonly props: GPUBufferCaptureProps<Format>;

  constructor(props: GPUBufferCaptureProps<Format>) {
    const {source, output} = props;
    // Capture preserves logical rows and requires independent storage.
    assert(source.format === output.format && source.length === output.length);
    assert(source.rowByteLength === output.rowByteLength && source.buffer !== output.buffer);
    assert(!output.buffer.transient);
    assert(
      [
        source.byteOffset,
        output.byteOffset,
        source.byteStride,
        output.byteStride,
        source.rowByteLength
      ].every(value => value % 4 === 0)
    );
    this.props = Object.freeze({...props});
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, source, output} = this.props;
    assert(source.buffer.graph === graph && output.buffer.graph === graph);
    if (!source.length) return [];
    const packed =
      source.byteStride === source.rowByteLength && output.byteStride === output.rowByteLength;
    return [
      createGPUCopyCommandNode({
        id,
        description: 'Capture GPU buffer rows into caller-owned snapshot storage',
        resources: [
          {buffer: source, usage: 'copy-source'},
          {buffer: output, usage: 'copy-destination'}
        ],
        workload: {
          operation: 'GPUBufferCapture',
          commandCount: packed ? 1 : source.length,
          readByteLength: source.length * source.rowByteLength,
          writeByteLength: source.length * source.rowByteLength
        },
        compile: () => ({
          encode: ({commandEncoder, getBuffer}) => {
            for (let row = 0; row < source.length; row += packed ? source.length : 1) {
              commandEncoder.copyBufferToBuffer({
                sourceBuffer: getBuffer(source),
                sourceOffset: source.byteOffset + row * source.byteStride,
                destinationBuffer: getBuffer(output),
                destinationOffset: output.byteOffset + row * output.byteStride,
                size: source.rowByteLength * (packed ? source.length : 1)
              });
            }
          }
        })
      })
    ];
  }
}
