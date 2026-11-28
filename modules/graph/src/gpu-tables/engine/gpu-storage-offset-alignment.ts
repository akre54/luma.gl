// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type Device} from '@luma.gl/core';

export function assertStorageBufferOffsetAligned(
  device: Device,
  byteOffset: number,
  numRows: number
): void {
  const alignment = device.limits.minStorageBufferOffsetAlignment;
  // WebGPU requires aligned storage offsets. Moving the offset would change what the shader reads.
  assert(numRows === 0 || !alignment || byteOffset % alignment === 0);
}
