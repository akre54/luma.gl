// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type TextureFormat} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GraphVectorView,
  GraphTextureHandle,
  GraphTextureView
} from './gpu-command-graph';
import type {GPUGraphVectorFormat} from './gpu-graph-value';
import type {GPUScalar} from './gpu-scalar';
import type {GPUOperation} from './gpu-operation';
import type {GPUOperationContract} from './gpu-operation-contract';
import type {GPUOperationLoweringContext} from './gpu-graph-lowering';
import type {GPUNode} from './gpu-command-node';

export type GPUCommandOperationContext<Parameters = void> = {
  graph: GPUCommandGraph<Parameters>;
  resolveScalar<T extends 'float32' | 'uint32' | 'sint32'>(port: string): GPUScalar<T>;
  resolveVector<T extends GPUGraphVectorFormat>(port: string): GraphVectorView<T>;
  resolveTexture<Format extends TextureFormat>(port: string): GraphTextureHandle<Format>;
  resolveTextureView<Format extends TextureFormat>(port: string): GraphTextureView<Format>;
};

export type GPUCommandOperationProps<Parameters = void> = {
  id: string;
  /** Port names map to logical graph value names declared by the contract. */
  ports: Readonly<Record<string, string>>;
  contract: GPUOperationContract;
  /** Called for each lowering with this execution's resolved resources. */
  createNodes(context: GPUCommandOperationContext<Parameters>): GPUNode<Parameters>;
};

/** Adapts command-node producers to the logical value contracts of GPUGraph. */
export class GPUCommandOperation<Parameters = void> implements GPUOperation {
  readonly type = 'command-operation';
  readonly id: string;
  readonly ports: Readonly<Record<string, string>>;
  readonly contract: GPUOperationContract;
  readonly createNodes: GPUCommandOperationProps<Parameters>['createNodes'];

  constructor(props: GPUCommandOperationProps<Parameters>) {
    this.id = props.id;
    this.ports = Object.freeze({...props.ports});
    this.contract = props.contract;
    this.createNodes = props.createNodes;
    const values = new Set(
      props.contract.values.map(use => (typeof use.value === 'string' ? use.value : use.value.id))
    );
    // Every named port must participate in planning and lifetime tracking.
    for (const value of Object.values(this.ports)) assert(values.has(value));
  }

  /** @internal Called by the compiler with logical names resolved through this operation's ports. */
  lower(context: GPUOperationLoweringContext<Parameters>): void {
    const getValue = (port: string): string => {
      // Prototype names are valid only when explicitly declared as ports.
      assert(Object.hasOwn(this.ports, port));
      return this.ports[port];
    };
    context.graph.add(
      this.createNodes({
        graph: context.graph,
        resolveScalar: port => context.resolveScalar(getValue(port)),
        resolveVector: port => context.resolveVector(getValue(port)),
        resolveTexture: port => context.resolveTexture(getValue(port)),
        resolveTextureView: port => context.resolveTextureView(getValue(port))
      })
    );
  }
}
