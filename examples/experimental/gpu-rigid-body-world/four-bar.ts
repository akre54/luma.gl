// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GroupNode, JointController, type ScenegraphJoint} from '@luma.gl/engine';
import {makeRigidBodyJointsFromScenegraph} from '@luma.gl/engine/physics';
import {Matrix4} from '@math.gl/core/matrix';
import {
  GPUArticulationTree,
  GPURigidBodyWorld,
  getBodyInertia,
  type GPUArticulationLink,
  type GPURigidBody
} from '@luma.gl/graph/gpu-physics';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';

/** World and attachment coordinates in meters. */
type Vector = [number, number, number];
/** Fixed simulation interval shared by both solvers. */
export const FOUR_BAR_STEP = 1 / 120;
/** Pivot spacing and arm dimensions for the authored parallel linkage. */
const PIVOT_SPACING = 1.3;
/** Center-to-center hinge spacing along each arm. */
const ARM_LENGTH = 1.05;
/** Initial lean of each arm from vertical in radians. */
const START_ANGLE = 0.86;
/** Platform collider half extents. */
const PLATFORM_SIZE: Vector = [0.79, 0.27, 0.045];
/** Motor spring stiffness in newton-meters per radian. */
const MOTOR_STIFFNESS = 700;
/** Motor damping in newton-meter seconds per radian. */
const MOTOR_DAMPING = 48;
/** Shared ID of the powered hinge. */
const DRIVE = 'left-pivot';

/** Create a solid box with the workcell's coefficient of friction. */
function box(position: Vector, halfExtents: Vector, mass: number): GPURigidBody {
  return {position, shape: {type: 'box', halfExtents}, mass, friction: 0.95, restitution: 0};
}

/** Allocate one scene and resolve the same four hinge attachments for both solvers. */
export function makeFourBar({
  ordered = false,
  publish = false,
  payloadMass = 3,
  offset = 0.3,
  maximumTorque = 90
} = {}) {
  const scene = new GroupNode({id: 'four-bar'});
  const reach = [
    ARM_LENGTH * Math.sin(START_ANGLE),
    0,
    ARM_LENGTH * Math.cos(START_ANGLE)
  ] as const;
  const leftBase: Vector = [-PIVOT_SPACING / 2, 0, 0.13];
  const rightBase: Vector = [PIVOT_SPACING / 2, 0, 0.13];
  const leftMiddle = leftBase.map((value, index) => value + reach[index] / 2) as Vector;
  const rightMiddle = rightBase.map((value, index) => value + reach[index] / 2) as Vector;
  const platformCenter: Vector = [reach[0], 0, leftBase[2] + reach[2]];
  const platform = box(platformCenter, PLATFORM_SIZE, 1.8);
  const armShape = {type: 'box' as const, halfExtents: [0.035, 0.055, ARM_LENGTH / 2] as Vector};
  const armOrientation: [number, number, number, number] = [
    0,
    Math.sin(START_ANGLE / 2),
    0,
    Math.cos(START_ANGLE / 2)
  ];
  const linkBodies: GPURigidBody[] = [
    {position: leftMiddle, mass: 0.85, colliders: [{shape: armShape, orientation: armOrientation}]},
    platform,
    {position: rightMiddle, mass: 0.85, colliders: [{shape: armShape, orientation: armOrientation}]}
  ];
  const bodies: GPURigidBody[] = [
    box([0, 0, 0], [1.65, 0.7, 0.05], 0),
    box(
      [platformCenter[0] + offset, 0, platformCenter[2] + 0.135],
      [0.16, 0.18, 0.09],
      payloadMass
    ),
    // Overhead stop above the load. The lift pushes into it before backing away.
    box([offset + 0.32, 0, 1.4], [0.32, 0.26, 0.08], 0)
  ];
  const nodes = linkBodies.map((body, index) => {
    const node = new GroupNode({
      id: ['left-arm', 'platform', 'right-arm'][index],
      matrix: new Matrix4().translate(body.position!)
    });
    scene.add(node);
    return node;
  });
  /** Add an attachment node at a world point under its owning body or the scene. */
  function attach(id: string, owner: number, point: Vector) {
    const parent = nodes[owner] ?? scene;
    const position =
      owner < 0 ? point : point.map((value, index) => value - linkBodies[owner].position![index]);
    const node = new GroupNode({id, matrix: new Matrix4().translate(position)});
    parent.add(node);
    return node;
  }
  const connections = [
    [-1, 0],
    [0, 1],
    [-1, 2],
    [2, 1]
  ] as const;
  const anchors: Vector[] = [
    leftBase,
    [platformCenter[0] + leftBase[0], 0, platformCenter[2]],
    rightBase,
    [platformCenter[0] + rightBase[0], 0, platformCenter[2]]
  ];
  const joints: ScenegraphJoint[] = connections.map(([first, second], index) => {
    const id = [DRIVE, 'left-top', 'right-pivot', 'right-top-closure'][index];
    return {
      id,
      first: attach(id + '-first', first, anchors[index]),
      second: attach(id + '-second', second, anchors[index]),
      collideConnected: false,
      limits: [
        {axes: [0, 1, 2], minimum: 0, maximum: 0},
        {axes: [0, 2], angular: true, minimum: 0, maximum: 0}
      ],
      drives:
        index === 0 && maximumTorque > 0
          ? [
              {
                axis: 1,
                angular: true,
                positionTarget: 0,
                stiffness: MOTOR_STIFFNESS,
                damping: MOTOR_DAMPING,
                maximumForce: maximumTorque
              }
            ]
          : []
    };
  });
  const resolved = new JointController(joints).resolve([scene], new Set(nodes));
  const links = resolved.slice(0, 3).map(
    ({joint, first, second}, index): GPUArticulationLink => ({
      id: joint.id!,
      parent: first.body ? nodes.indexOf(first.body as GroupNode) : -1,
      parentAnchor: first.position,
      childAnchor: second.position,
      axis: [0, 1, 0],
      mass: linkBodies[index].mass!,
      inertia: getBodyInertia(
        linkBodies[index].mass!,
        index === 1 ? {type: 'box', halfExtents: PLATFORM_SIZE} : armShape
      )
    })
  );
  const closure = resolved[3];
  const linkStart = bodies.length;
  const bodyIndices = new Map<GroupNode, number>(
    nodes.map((node, index) => [node, linkStart + index])
  );
  if (ordered) bodies.push(...linkBodies);
  const world = new GPURigidBodyWorld({
    bodies,
    joints: ordered
      ? resolved.flatMap(joint => makeRigidBodyJointsFromScenegraph(joint, bodyIndices, 0))
      : [],
    timeStep: FOUR_BAR_STEP,
    gravity: [0, 0, -9.81],
    solver: ordered ? 'ordered' : 'parallel',
    iterations: ordered ? 64 : 8,
    publishPoses: publish
  });
  const tree = ordered
    ? undefined
    : new GPUArticulationTree({
        links,
        timeStep: FOUR_BAR_STEP,
        publishMatrices: publish,
        loop: {
          first: {link: links[2].id, anchor: closure.first.position},
          second: {link: links[1].id, anchor: closure.second.position},
          normal: [0, 1, 0]
        }
      });
  return {
    scene,
    nodes,
    bodies,
    linkBodies,
    armOrientation,
    world,
    tree,
    /** Add one world step with the platform's four-point box face enabled. */
    addToGraph(graph: GPUCommandGraph) {
      world.addToGraph(
        graph,
        tree
          ? {
              tree,
              colliders: [
                {
                  link: links[1].id,
                  shape: {type: 'box', halfExtents: PLATFORM_SIZE},
                  faceManifold: true
                }
              ],
              friction: 0.95
            }
          : undefined
      );
    },
    /** Update the shared hinge servo in radians. */
    command(angle: number) {
      if (tree)
        tree.setDrive(DRIVE, {
          angle,
          stiffness: MOTOR_STIFFNESS,
          damping: MOTOR_DAMPING,
          maximumTorque
        });
      else if (maximumTorque > 0) world.setAngularActuator(DRIVE, 1, {angle, maximumTorque});
    },
    /** Sample closure and platform attitude directly from the simulated body poses. */
    async read() {
      const [states, worldStatus, linkStates, contact] = await Promise.all([
        world.readState(),
        world.readSimulationDiagnostics(),
        tree?.readState(),
        tree?.readSimulationDiagnostics()
      ]);
      const poses = linkStates ?? states.slice(linkStart);
      /** Transform an attachment point by its simulated body pose. */
      function anchor(index: number, point: Vector) {
        const body = poses[index];
        return new Matrix4()
          .translate(body.position)
          .multiplyRight(new Matrix4().fromQuaternion(body.orientation))
          .transformAsPoint(point);
      }
      const first = anchor(2, closure.first.position);
      const second = anchor(1, closure.second.position);
      const rotation = poses[1].orientation;
      let support = 0;
      contact?.contactPairs?.forEach((pair, index) => {
        if (pair.body === 1) support += contact.contactImpulses![index][2] / FOUR_BAR_STEP;
      });
      return {
        complete: worldStatus.complete && (contact?.complete ?? true),
        closure: Math.hypot(first[0] - second[0], first[1] - second[1], first[2] - second[2]),
        tilt: 2 * Math.atan2(Math.hypot(...rotation.slice(0, 3)), Math.abs(rotation[3])),
        platform: poses[1].position,
        payload: states[1].position,
        support: contact ? support : null,
        contacts: contact?.activeContacts
      };
    },
    /** Dispose the scene, articulation and world owned by the fixture. */
    destroy() {
      scene.destroy();
      tree?.destroy();
      world.destroy();
    }
  };
}

/** Ease into the overhead stop, hold, and return to the starting angle. */
export function fourBarTarget(step: number) {
  /** Cubic time ramp from zero to one. */
  function ramp(start: number, duration: number) {
    const fraction = Math.max(0, Math.min(1, (step - start) / duration));
    return fraction * fraction * (3 - 2 * fraction);
  }
  return -0.64 * (ramp(30, 150) - ramp(310, 150));
}
