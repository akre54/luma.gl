// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Prepare a two-row closure for a planar revolute joint and solve it as a block. */
export const articulationLoopShader = /* wgsl */ `
struct LoopClosure {
  firstJacobian:array<f32,N>, secondJacobian:array<f32,N>,
  firstResponse:array<f32,N>, secondResponse:array<f32,N>,
  effectiveMass:mat2x2f, targetVelocity:vec2f
}

fn prepareLoop(dt:f32,inverse:ptr<function,array<array<f32,N>,N>>)->LoopClosure {
  var closure:LoopClosure;
  let first=i32(settings.loopFirst.w);
  let second=i32(settings.loopSecond.w);
  let firstPoint=solved[first].position.xyz+rotate(solved[first].rotation,settings.loopFirst.xyz);
  let secondPoint=solved[second].position.xyz+rotate(solved[second].rotation,settings.loopSecond.xyz);
  let normal=settings.loopNormal.xyz;
  for(var link=0u;link<N;link++){
    let direction=axis[link];
    let projection=abs(dot(direction,normal));
    let valid=select(length(cross(direction,normal))<1e-4,projection<1e-4,links[link].childAnchor.w!=0.);
    if(!valid){counters[1]=1u;return closure;}
  }
  let reference=select(vec3f(1,0,0),vec3f(0,1,0),abs(normal.x)>0.8);
  let tangent=normalize(cross(normal,reference));
  let bitangent=cross(normal,tangent);
  let error=secondPoint-firstPoint;
  if(abs(dot(error,normal))>1e-4){counters[1]=1u;return closure;}
  closure.targetVelocity=clamp(-0.35/dt*vec2f(dot(error,tangent),dot(error,bitangent)),vec2f(-2),vec2f(2));
  for(var joint=first;joint>=0;joint=i32(links[joint].parentAnchor.w)){
    let column=linearColumn(joint,firstPoint);
    closure.firstJacobian[joint]-=dot(column,tangent);
    closure.secondJacobian[joint]-=dot(column,bitangent);
  }
  for(var joint=second;joint>=0;joint=i32(links[joint].parentAnchor.w)){
    let column=linearColumn(joint,secondPoint);
    closure.firstJacobian[joint]+=dot(column,tangent);
    closure.secondJacobian[joint]+=dot(column,bitangent);
  }
  var firstMass=0.;var secondMass=0.;var crossMass=0.;
  for(var row=0u;row<N;row++){
    for(var column=0u;column<N;column++){
      closure.firstResponse[row]+=(*inverse)[row][column]*closure.firstJacobian[column];
      closure.secondResponse[row]+=(*inverse)[row][column]*closure.secondJacobian[column];
    }
    firstMass+=closure.firstJacobian[row]*closure.firstResponse[row];
    secondMass+=closure.secondJacobian[row]*closure.secondResponse[row];
    crossMass+=closure.firstJacobian[row]*closure.secondResponse[row];
  }
  let determinant=firstMass*secondMass-crossMass*crossMass;
  // Stop at a rank-deficient mechanism instead of publishing a partially solved step.
  if(!(determinant>1e-6*firstMass*secondMass)){counters[1]=1u;return closure;}
  closure.effectiveMass=mat2x2f(secondMass,-crossMass,-crossMass,firstMass)*(1./determinant);
  return closure;
}

fn solveLoop(closure:ptr<function,LoopClosure>)->f32 {
  var velocity=vec2f(0);
  for(var joint=0u;joint<N;joint++){
    velocity+=speed[joint]*vec2f((*closure).firstJacobian[joint],(*closure).secondJacobian[joint]);
  }
  let impulse=(*closure).effectiveMass*((*closure).targetVelocity-velocity);
  for(var joint=0u;joint<N;joint++){
    speed[joint]+=(*closure).firstResponse[joint]*impulse.x+(*closure).secondResponse[joint]*impulse.y;
  }
  return length(impulse);
}
`;
