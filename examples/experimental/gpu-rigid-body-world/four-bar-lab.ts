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
import {makeFourBar, fourBarTarget, FOUR_BAR_STEP} from './four-bar';

/** Presentation surface shared by the renderer and orbit controls. */
const canvas = document.querySelector<HTMLCanvasElement>('canvas')!;
/** Run the lift cycle or reset its starting state. */
const cycle = document.querySelector<HTMLButtonElement>('#cycle')!;
/** Live upward reaction on the load in newtons. */
const reaction = document.querySelector<HTMLOutputElement>('#reaction')!;
/** Live loop closure and attitude metrics. */
const reading = document.querySelector<HTMLOutputElement>('#reading')!;
/** Shared GPU graph inspector and timeline. */
const diagnostics = new GPUGraphDiagnostics({});
/** Workcell controls and diagnostics panel. */
const panel = createRigidBodyPanel(diagnostics, 'Closed-Loop Lift', 'Four-bar mechanism · 120 Hz');
/** Z-up orbit camera around the linkage. */
const camera = new PerspectiveCamera(38, 1, 0.1, 40);
camera.up.set(0, 0, 1);
camera.position.set(3.1, -4.3, 2.6);
/** Orbit interaction on the scene canvas. */
const orbit = new OrbitControls(camera, canvas);
orbit.target.set(0.25, 0, 0.75);
orbit.update();
/** Convert the camera projection into WebGPU clip depth. */
const depth = new Matrix4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 1]);
/** Unit cube shared by the frame, linkage and payload. */
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
function populateScene(trial: ReturnType<typeof makeFourBar>) {
  const steel = makeFinish('steel', [0.52, 0.6, 0.66, 1], 0.57);
  const enamel = makeFinish('enamel', [0.07, 0.25, 0.46, 1], 0.28);
  const brass = makeFinish('brass', [0.71, 0.41, 0.14, 1], 0.48);
  const base = makeFinish('base', [0.14, 0.17, 0.2, 1], 0.18);
  const statics = new GroupNode({id: 'workcell'});
  trial.scene.add(statics);
  const linkBindings = new Map<ScenegraphNode, ScenegraphTransformBinding>();
  const worldBindings = new Map<ScenegraphNode, ScenegraphTransformBinding>();
  /** Create a PBR cuboid with an optional local transform. */
  function addBox(
    group: GroupNode,
    id: string,
    size: readonly number[],
    material: SceneMaterial,
    transform = new Matrix4()
  ) {
    const matrix = transform.scale(size);
    const mesh = new SceneMeshNode({id, geometry: cube, material, matrix});
    group.add(mesh);
    return {mesh, matrix};
  }
  for (const [index, body] of trial.linkBodies.entries()) {
    const shape = body.shape ?? body.colliders?.[0].shape;
    if (shape?.type !== 'box') continue;
    const transform =
      index === 1 ? new Matrix4() : new Matrix4().fromQuaternion(trial.armOrientation);
    const {mesh, matrix} = addBox(
      trial.nodes[index],
      'link-mesh-' + index,
      shape.halfExtents,
      index === 1 ? enamel : steel,
      transform
    );
    linkBindings.set(mesh, {matrixIndex: index, localMatrix: matrix});
  }
  for (const [index, body] of trial.bodies.entries()) {
    if (body.shape?.type !== 'box') continue;
    const {mesh, matrix} = addBox(
      statics,
      'body-mesh-' + index,
      body.shape.halfExtents,
      index === 1 ? brass : base
    );
    worldBindings.set(mesh, {matrixIndex: index, localMatrix: matrix});
  }
  for (const pivotX of [-0.65, 0.65]) {
    addBox(
      statics,
      'pedestal-' + pivotX,
      [0.09, 0.105, 0.075],
      base,
      new Matrix4().translate([pivotX, 0, 0.105])
    );
  }
  const stop = trial.bodies[2].position;
  addBox(
    statics,
    'stop-post',
    [0.06, 0.06, stop[2] * 0.5],
    base,
    new Matrix4().translate([stop[0], 0.62, stop[2] * 0.5])
  );
  addBox(
    statics,
    'stop-bracket',
    [0.06, 0.23, 0.045],
    base,
    new Matrix4().translate([stop[0], 0.41, stop[2]])
  );
  trial.scene.transformSource = {
    getFrame: () => ({
      matrices: trial.tree!.poseMatrices!,
      matrixCount: trial.nodes.length,
      revision: 0,
      bindings: linkBindings
    })
  };
  statics.transformSource = {
    getFrame: () => ({
      matrices: trial.world.poseMatrices!,
      matrixCount: trial.bodies.length,
      revision: 0,
      bindings: worldBindings
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
  const torque = bindSlider('torque', 'N·m');
  /** Compile a workcell with the selected load and motor. */
  async function createTrial() {
    const trial = makeFourBar({
      publish: true,
      payloadMass: mass.valueAsNumber,
      offset: offset.valueAsNumber,
      maximumTorque: torque.valueAsNumber
    });
    const definition = new GPUCommandGraph(device);
    trial.addToGraph(definition);
    const graph = await definition.compileAsync();
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
      cycle.textContent = 'Reset lift';
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
      cycle.textContent = 'Run lift';
      cycle.disabled = false;
    }
    accumulator = Math.min(4 * FOUR_BAR_STEP, accumulator + (now - previous) / 1000);
    previous = now;
    while (accumulator >= FOUR_BAR_STEP) {
      if (step % 4 === 0) simulation.trial.command(fourBarTarget(step));
      const encoder = device.createCommandEncoder();
      diagnostics.encode(simulation.graph, encoder, {parameters: undefined});
      device.submit(encoder.finish());
      accumulator -= FOUR_BAR_STEP;
      if (stage === 'running' && ++step === 560) stage = 'done';
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
          reaction.value = metrics.complete
            ? (metrics.support ?? 0).toFixed(1) + ' N'
            : 'Solve stopped';
          reading.value =
            (metrics.closure * 1000).toFixed(2) +
            ' mm closure · ' +
            ((metrics.tilt * 180) / Math.PI).toFixed(2) +
            '° tilt · ' +
            metrics.contacts +
            ' contact rows';
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
