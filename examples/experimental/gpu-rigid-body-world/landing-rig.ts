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
  type GPURigidBody,
  type GPURigidBodyJoint
} from '@luma.gl/graph/gpu-physics';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';

/** Three-component vector in the Z-up workcell. */
type Vector = [number, number, number];
/** Both solver lanes take 120 steps per second. */
export const LANDING_STEP = 1 / 120;
/** Vertical difference between the upper and lower landing decks. */
export const DECK_STEP = 0.15;
/** Servo spring stiffness in N/m. */
const STIFFNESS = 4200;
/** Servo damping in N·s/m. */
const DAMPING = 155;
/** Coefficient of sliding friction at the foot and deck. */
const FRICTION = 0.95;

/** Create both solver lanes from the same landing decks, scenegraph joints and mass properties. */
export function makeLandingRig({
  ordered = false,
  publish = false,
  payloadMass = 3,
  offset = 0.16,
  drop = 0.3,
  maximumForce = 220
} = {}) {
  const scene = new GroupNode({id: 'landing-rig'});
  const totalBaseMass = 4 + payloadMass;
  const baseOffset = (offset * payloadMass) / totalBaseMass;
  const chassisPosition: Vector = [baseOffset, 0, 0.7 + DECK_STEP + drop];
  const chassisShape = {type: 'box' as const, halfExtents: [0.58, 0.25, 0.11] as Vector};
  const baseInertia = getBodyInertia(totalBaseMass, chassisShape);
  const footShape = {type: 'box' as const, halfExtents: [0.17, 0.23, 0.07] as Vector};
  const feet = [-0.43, 0.43].map(lateral => [lateral, 0, chassisPosition[2] - 0.63] as Vector);
  const decks: GPURigidBody[] = [
    {
      position: [-1, 0, DECK_STEP - 0.125],
      shape: {type: 'box', halfExtents: [1, 0.75, 0.125]},
      mass: 0,
      friction: FRICTION
    },
    {
      position: [1, 0, -0.125],
      shape: {type: 'box', halfExtents: [1, 0.75, 0.125]},
      mass: 0,
      friction: FRICTION
    }
  ];
  const nodes = [...feet, chassisPosition].map((position, index) => {
    const node = new GroupNode({
      id: ['left-foot', 'right-foot', 'chassis'][index],
      matrix: new Matrix4().translate(position)
    });
    scene.add(node);
    return node;
  });
  /** Place a joint attachment in its owning body's local frame. */
  function attach(index: number, position: Vector, id: string) {
    const attachment = new GroupNode({id, matrix: new Matrix4().translate(position)});
    nodes[index].add(attachment);
    return attachment;
  }
  const joints: ScenegraphJoint[] = feet.map(([lateral], index) => {
    const id = index === 0 ? 'left-strut' : 'right-strut';
    return {
      id,
      first: attach(2, [lateral - baseOffset, 0, -0.07], id + '-hip'),
      second: attach(index, [0, 0, 0.56], id + '-shaft'),
      collideConnected: false,
      limits: [
        {axes: [0, 1], minimum: 0, maximum: 0},
        {axes: [0, 1, 2], angular: true, minimum: 0, maximum: 0}
      ],
      drives: [{axis: 2, positionTarget: 0, stiffness: STIFFNESS, damping: DAMPING, maximumForce}]
    };
  });
  const resolved = new JointController(joints).resolve([scene], new Set(nodes));
  const links = resolved.map(
    ({joint, first, second}): GPUArticulationLink => ({
      id: joint.id!,
      parent: -1,
      parentAnchor: first.position,
      childAnchor: second.position,
      axis: [0, 0, 1],
      jointType: 'prismatic',
      mass: 0.65,
      inertia: getBodyInertia(0.65, footShape)
    })
  );
  const bodyIndices = new Map<GroupNode, number>(
    nodes.map((node, index) => [node, decks.length + index])
  );
  const orderedJoints = ordered
    ? resolved.flatMap(joint => makeRigidBodyJointsFromScenegraph(joint, bodyIndices))
    : [];
  const bodies: GPURigidBody[] = [...decks];
  if (ordered) {
    for (const position of feet) {
      bodies.push({position, shape: footShape, mass: 0.65, friction: FRICTION, restitution: 0});
    }
    bodies.push({
      position: chassisPosition,
      mass: totalBaseMass,
      inertiaDiagonal: baseInertia,
      friction: FRICTION,
      restitution: 0,
      colliders: [{position: [-baseOffset, 0, 0], shape: chassisShape}]
    });
  }
  const world = new GPURigidBodyWorld({
    bodies,
    joints: orderedJoints,
    timeStep: LANDING_STEP,
    gravity: [0, 0, -9.81],
    solver: ordered ? 'ordered' : 'parallel',
    iterations: ordered ? 64 : 8,
    publishPoses: publish
  });
  const tree = ordered
    ? undefined
    : new GPUArticulationTree({
        links,
        floatingBase: {
          id: 'chassis',
          position: chassisPosition,
          mass: totalBaseMass,
          inertia: baseInertia
        },
        timeStep: LANDING_STEP,
        publishMatrices: publish
      });
  return {
    scene,
    nodes,
    decks,
    world,
    tree,
    footShape,
    chassisShape,
    baseOffset,
    offset,
    totalBaseMass,
    /** Register one indexed world-contact solve for the two feet and chassis. */
    addToGraph(graph: GPUCommandGraph) {
      world.addToGraph(
        graph,
        tree
          ? {
              tree,
              colliders: [
                ...links.map(link => ({link: link.id, shape: footShape, faceManifold: true})),
                {
                  link: 'chassis',
                  shape: chassisShape,
                  position: [-baseOffset, 0, 0],
                  faceManifold: true
                }
              ],
              friction: FRICTION
            }
          : undefined
      );
    },
    /** Extend the right strut after touchdown to level the chassis on the step. */
    command(step: number) {
      const fraction = Math.max(0, Math.min(1, (step - 45) / 72));
      const position = -DECK_STEP * fraction * fraction * (3 - 2 * fraction);
      const right = links[1].id;
      if (tree) {
        tree.setLinearDrive(links[0].id, {
          position: 0,
          stiffness: STIFFNESS,
          damping: DAMPING,
          maximumForce
        });
        tree.setLinearDrive(right, {
          position,
          stiffness: STIFFNESS,
          damping: DAMPING,
          maximumForce
        });
      } else {
        const index = orderedJoints.findIndex(
          joint => joint.id === right && joint.type === 'general' && joint.drive
        );
        const joint = orderedJoints[index] as Extract<GPURigidBodyJoint, {type: 'general'}>;
        world.setJointParameters(index, {
          ...joint,
          drive: {...joint.drive, positionTarget: position}
        });
      }
    },
    /** Sample chassis attitude, foot clearance and the upward reactions on the feet. */
    async read() {
      const [poses, contacts, status] = await Promise.all([
        tree ? tree.readState() : world.readState().then(states => states.slice(decks.length)),
        tree?.readSimulationDiagnostics(),
        world.readSimulationDiagnostics()
      ]);
      const [left, right, chassis] = poses;
      const up = new Matrix4().fromQuaternion(chassis.orientation).transformAsVector([0, 0, 1]);
      const penetrations = [left, right].map((foot, index) => {
        const rotation = new Matrix4().fromQuaternion(foot.orientation);
        const reach =
          Math.abs(rotation[2]) * footShape.halfExtents[0] +
          Math.abs(rotation[6]) * footShape.halfExtents[1] +
          Math.abs(rotation[10]) * footShape.halfExtents[2];
        return Math.max(0, (index === 0 ? DECK_STEP : 0) - foot.position[2] + reach);
      });
      const loads = [0, 0];
      contacts?.contactPairs?.forEach((pair, index) => {
        if (pair.collider < loads.length)
          loads[pair.collider] -= contacts.contactImpulses![index][2] / LANDING_STEP;
      });
      return {
        complete: status.complete && (contacts?.complete ?? true),
        tilt: Math.acos(Math.min(1, Math.max(-1, up[2]))),
        position: chassis.position,
        velocity: chassis.velocity,
        feet: [left.position, right.position],
        penetrations,
        loads: contacts ? loads : null
      };
    },
    /** Destroy the owned GPU allocations and scenegraph. */
    destroy() {
      scene.destroy();
      tree?.destroy();
      world.destroy();
    }
  };
}
