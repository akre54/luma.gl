// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUDGGSCellBoundary,
  type GPUDGGSCellBoundaryProps
} from '../gpu-dggs/gpu-dggs-cell-boundary';

/** Caller-owned views for a resident A5 boundary pass. */
export type GPUA5CellBoundaryProps = Omit<GPUDGGSCellBoundaryProps, 'family'>;

/** Expands packed A5 cells with the shared DGGS boundary contract. */
export class GPUA5CellBoundary extends GPUDGGSCellBoundary {
  /** Selects the A5 boundary kernel. */
  constructor(props: GPUA5CellBoundaryProps) {
    super({...props, family: 'a5'});
  }
}
