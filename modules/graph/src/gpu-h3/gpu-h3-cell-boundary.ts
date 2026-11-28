// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUDGGSCellBoundary,
  type GPUDGGSCellBoundaryProps
} from '../gpu-dggs/gpu-dggs-cell-boundary';

/** Caller-owned views for a resident H3 boundary pass. */
export type GPUH3CellBoundaryProps = Omit<GPUDGGSCellBoundaryProps, 'family'>;

/** Expands packed H3 cells with the shared DGGS boundary contract. */
export class GPUH3CellBoundary extends GPUDGGSCellBoundary {
  /** Selects the H3 boundary kernel. */
  constructor(props: GPUH3CellBoundaryProps) {
    super({...props, family: 'h3'});
  }
}
