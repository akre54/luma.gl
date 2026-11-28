// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {luma} from '@luma.gl/core';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {
  CubeGeometry,
  GroupNode,
  type ScenegraphNode,
  type ScenegraphTransformBinding
} from '@luma.gl/engine';
import {SceneMeshNode, SceneRenderer, type SceneMaterial} from '@luma.gl/anari';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Matrix4} from '@math.gl/core/matrix';
import {PerspectiveCamera} from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GPUGraphDiagnostics} from '../../gpu-graph-diagnostics';
import {createRigidBodyPanel} from './info-box';
import {makeLandingRig, LANDING_STEP} from './landing-rig';

/** Presentation surface shared by the renderer and orbit controls. */
const canvas = document.querySelector<HTMLCanvasElement>('canvas')!;
/** Drop the rig or reset its starting state. */
const cycle = document.querySelector<HTMLButtonElement>('#cycle')!;
/** Live upward reactions on the two feet in newtons. */
const reaction = document.querySelector<HTMLOutputElement>('#reaction')!;
/** Live chassis attitude and height metrics. */
const reading = document.querySelector<HTMLOutputElement>('#reading')!;
/** Shared GPU graph inspector and timeline. */
const diagnostics = new GPUGraphDiagnostics({});
/** Workcell controls and diagnostics panel. */
const panel = createRigidBodyPanel(diagnostics, 'Two-Legged Landing Rig', 'Floating base · 120 Hz');
/** Z-up orbit camera around the linkage. */
const camera = new PerspectiveCamera(38, 1, 0.1, 40);
camera.up.set(0, 0, 1);
camera.position.set(2.8, -4.4, 2.25);
/** Orbit interaction on the scene canvas. */
const orbit = new OrbitControls(camera, canvas);
orbit.target.set(0, 0, 0.53);
orbit.update();
/** Convert the camera projection into WebGPU clip depth. */
const depth = new Matrix4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1]);
/** Unit cube shared by the decks, chassis and struts. */
const cube = new CubeGeometry();

/** Make a metallic-roughness finish for a workcell component. */
function makeFinish(
  id: string,
  color: [number, number, number, number],
  metallic: number
): SceneMaterial {
  return {id, uniforms: {baseColorFactor: color, metallicRoughnessValues: [metallic, 0.39]}};
}

/** Add PBR meshes to the authored workcell and bind published transforms. */
function populateScene(trial: ReturnType<typeof makeLandingRig>) {
  const steel = makeFinish('steel', [0.52, 0.6, 0.66, 1], 0.57);
  const enamel = makeFinish('enamel', [0.07, 0.25, 0.46, 1], 0.28);
  const brass = makeFinish('brass', [0.71, 0.41, 0.14, 1], 0.48);
  const base = makeFinish('base', [0.14, 0.17, 0.2, 1], 0.18);
  const statics = new GroupNode({id: 'workcell'});
  trial.scene.add(statics);
  const linkBindings = new Map<ScenegraphNode, ScenegraphTransformBinding>();
  /** Create one PBR object, optionally driven by the corresponding simulated body. */
  function addBox(
    group: GroupNode,
    id: string,
    size: readonly number[],
    material: SceneMaterial,
    offset: readonly number[],
    bodyIndex?: number
  ) {
    const matrix = new Matrix4().translate(offset).scale(size);
    const mesh = new SceneMeshNode({id, geometry: cube, material, matrix});
    group.add(mesh);
    if (bodyIndex !== undefined)
      linkBindings.set(mesh, {matrixIndex: bodyIndex, localMatrix: matrix});
  }
  for (const [index, deck] of trial.decks.entries()) {
    if (deck.shape?.type === 'box')
      addBox(statics, 'deck-' + index, deck.shape.halfExtents, base, deck.position);
  }
  const chassis = trial.nodes[2];
  addBox(
    chassis,
    'chassis-shell',
    trial.chassisShape.halfExtents,
    enamel,
    [-trial.baseOffset, 0, 0],
    2
  );
  addBox(
    chassis,
    'payload',
    [0.14, 0.17, 0.105],
    brass,
    [trial.offset - trial.baseOffset, 0, 0.215],
    2
  );
  for (const [index, lateral] of [-0.43, 0.43].entries()) {
    addBox(
      chassis,
      'strut-sleeve-' + index,
      [0.071, 0.081, 0.2],
      enamel,
      [lateral - trial.baseOffset, 0, -0.25],
      2
    );
    addBox(trial.nodes[index], 'shaft-' + index, [0.043, 0.043, 0.26], steel, [0, 0, 0.32], index);
    addBox(
      trial.nodes[index],
      'foot-' + index,
      trial.footShape.halfExtents,
      steel,
      [0, 0, 0],
      index
    );
  }
  trial.scene.transformSource = {
    getFrame: () => ({
      matrices: trial.tree!.poseMatrices!,
      matrixCount: trial.nodes.length,
      revision: 0,
      bindings: linkBindings
    })
  };
}

/** Display an error in the lab. */
function showError(error: unknown) {
  document.querySelector<HTMLOutputElement>('#error')!.value = String(error);
}

/** Start the renderer and a fence-paced fixed-step simulation. */
async function run() {
  const device = await luma.createDevice({
    adapters: [webgpuAdapter],
    type: 'webgpu',
    createCanvasContext: {canvas, autoResize: true}
  });
  const renderer = new SceneRenderer(device);
  let reset = false;
  /** Bind a slider that rebuilds the workcell at its next frame boundary. */
  function bindSlider(id: string, unit: string) {
    const slider = document.querySelector<HTMLInputElement>('#' + id)!;
    const output = document.querySelector<HTMLOutputElement>('#' + id + '-value')!;
    slider.oninput = () => {
      output.value = slider.valueAsNumber + ' ' + unit;
    };
    slider.onchange = () => {
      reset = true;
      cycle.disabled = true;
    };
    return slider;
  }
  const mass = bindSlider('mass', 'kg');
  const offset = bindSlider('offset', 'm');
  const drop = bindSlider('drop', 'm');
  /** Compile a workcell with the selected load and motor. */
  async function createTrial() {
    const trial = makeLandingRig({
      publish: true,
      payloadMass: mass.valueAsNumber,
      offset: offset.valueAsNumber,
      drop: drop.valueAsNumber
    });
    const definition = new GPUCommandGraph(device);
    trial.addToGraph(definition);
    const graph = await definition.compileAsync();
    const encoder = device.createCommandEncoder();
    trial.command(0);
    diagnostics.encode(graph, encoder, {parameters: undefined});
    device.submit(encoder.finish());
    populateScene(trial);
    return {trial, graph};
  }
  let simulation = await createTrial();
  let stage: 'ready' | 'running' | 'done' = 'ready';
  let step = 0;
  let accumulator = 0;
  let previous = performance.now();
  let lastReading = 0;
  let pending: Promise<void> | undefined;
  cycle.disabled = false;
  cycle.onclick = () => {
    if (stage === 'ready') {
      stage = 'running';
      cycle.textContent = 'Reset rig';
    } else {
      reset = true;
      cycle.disabled = true;
    }
  };
  /** Key light and shadow direction for the workcell. */
  const spot = {
    type: 'spot',
    position: [-2, -3, 6],
    direction: [2, 3, -6],
    color: [1, 0.96, 0.88],
    intensity: 9,
    innerConeAngle: 0.45,
    outerConeAngle: 0.9
  } as const;
  /** Advance the simulation, present the published scene and sample diagnostics. */
  async function frame(now: number) {
    if (reset) {
      reset = false;
      await pending;
      const replacement = await createTrial();
      simulation.graph.destroy();
      simulation.trial.destroy();
      simulation = replacement;
      diagnostics.clear();
      stage = 'ready';
      step = 0;
      accumulator = 0;
      previous = now;
      cycle.textContent = 'Drop rig';
      cycle.disabled = false;
    }
    accumulator = Math.min(4 * LANDING_STEP, accumulator + (now - previous) / 1000);
    previous = now;
    while (accumulator >= LANDING_STEP && stage === 'running') {
      if (step % 4 === 0) simulation.trial.command(step);
      const encoder = device.createCommandEncoder();
      diagnostics.encode(simulation.graph, encoder, {parameters: undefined});
      device.submit(encoder.finish());
      accumulator -= LANDING_STEP;
      if (++step === 360) stage = 'done';
    }
    camera.aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    renderer.render({
      scene: simulation.trial.scene,
      camera: {
        viewMatrix: new Matrix4(camera.matrixWorldInverse.elements),
        projectionMatrix: new Matrix4(depth).multiplyRight(camera.projectionMatrix.elements),
        near: camera.near,
        far: camera.far,
        clipDepth: 'zero-to-one',
        position: [camera.position.x, camera.position.y, camera.position.z]
      },
      lights: [{type: 'ambient', intensity: 0.7, color: [0.89, 0.94, 1]}, spot],
      background: [0.095, 0.12, 0.15, 1],
      spotShadows: {
        spotLights: [
          {
            position: spot.position,
            direction: spot.direction,
            outerConeAngle: spot.outerConeAngle,
            range: 14,
            normalBias: 0.004,
            sourceRadius: 0.035
          }
        ]
      }
    });
    device.submit();
    panel.update();
    if (!pending && now - lastReading > 250) {
      lastReading = now;
      pending = simulation.trial
        .read()
        .then(metrics => {
          const [left, right] = metrics.loads ?? [0, 0];
          reaction.value = metrics.complete
            ? `L ${left.toFixed(1)} N · R ${right.toFixed(1)} N`
            : 'Solve stopped';
          reading.value =
            ((metrics.tilt * 180) / Math.PI).toFixed(2) +
            '° tilt · ' +
            metrics.position[2].toFixed(2) +
            ' m chassis height';
        })
        .catch(showError)
        .finally(() => {
          pending = undefined;
        });
    }
    const fence = device.createFence();
    try {
      await fence.signaled;
    } finally {
      fence.destroy();
    }
    requestAnimationFrame(tick);
  }
  /** Schedule a frame and surface failures in the panel. */
  const tick = (now: number) => void frame(now).catch(showError);
  requestAnimationFrame(tick);
}

void run().catch(showError);
