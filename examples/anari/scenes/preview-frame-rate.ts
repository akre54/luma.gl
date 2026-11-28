// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Measures rendered preview frames over wall-clock time, including skipped animation ticks. */
export class PreviewFrameRate {
  /** Timestamp in milliseconds at the start of the current measurement window. */
  private windowStart?: number;
  /** Frames rendered in the current measurement window. */
  private renderedFrames = 0;

  /** Creates a counter that writes FPS to the optional preview element. */
  constructor(private readonly indicator: HTMLElement | null) {}

  /**
   * Advances the measurement window with the current time in milliseconds.
   * Call every animation tick, including ticks that skip rendering.
   */
  tick(time: number): void {
    this.windowStart ??= time;
    const elapsed = time - this.windowStart;
    if (elapsed < 500) return;

    const framesPerSecond = (this.renderedFrames * 1000) / elapsed;
    if (this.indicator) {
      this.indicator.textContent = framesPerSecond.toFixed(framesPerSecond < 10 ? 1 : 0);
    }
    this.windowStart = time;
    this.renderedFrames = 0;
  }

  /** Counts one rendered preview frame. */
  recordFrame(): void {
    this.renderedFrames++;
  }
}
