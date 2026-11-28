// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {articulationLoopShader} from './articulation-loop-shader';
import {worldBoxDeclaration} from './articulation-world-contacts';
import {getPayloadDeclarations} from './articulation-payload-shader';

/** Joint-space forward dynamics. One invocation owns a whole small tree. */
export function getArticulationTreeShader(
  count: number,
  contact = false,
  publishMatrices = false,
  freePayload = false,
  boxCount = 0,
  colliderCount = 0,
  sliders = false,
  loop = false,
  floating = false
): string {
  return /* wgsl */ `
const LINK_COUNT = ${count}u;
const N = ${count + (floating ? 6 : 0)}u;
struct Link {
  parentAnchor:vec4f, childAnchor:vec4f, axisMass:vec4f, inertia:vec4f, rest:vec4f
}
struct Pose { position:vec4f, rotation:vec4f, velocity:vec4f, spin:vec4f }
struct Settings { base:vec4f, gravityStep:vec4f, boxCenterRadius:vec4f, boxExtentsLink:vec4f ${freePayload || boxCount ? `, payloadSizeMass:vec4f, surface:vec4f, pads:array<vec4f,${colliderCount ? 3 * colliderCount : 2}>` : ''} ${loop ? ', loopFirst:vec4f, loopSecond:vec4f, loopNormal:vec4f' : ''} ${floating ? ', baseInertiaMass:vec4f' : ''} }
@group(0) @binding(0) var<storage,read> links:array<Link>;
@group(0) @binding(1) var<storage,read> drives:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> coordinates:array<vec4f>;
@group(0) @binding(3) var<storage,read_write> poses:array<Pose>;
@group(0) @binding(4) var<storage,read_write> counters:array<u32>;
@group(0) @binding(5) var<uniform> settings:Settings;

${publishMatrices ? '@group(0) @binding(6) var<storage,read_write> matrices:array<mat4x4f>;' : ''}
var<private> angle:array<f32,N>;
var<private> speed:array<f32,N>;
var<private> solved:array<Pose,N>;
var<private> axis:array<vec3f,N>;
var<private> pivot:array<vec3f,N>;
var<private> acceleration:array<vec3f,N>;
var<private> angularAcceleration:array<vec3f,N>;
fn rotate(q:vec4f,v:vec3f)->vec3f{return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
fn product(a:vec4f,b:vec4f)->vec4f{return vec4f(a.w*b.xyz+b.w*a.xyz+cross(a.xyz,b.xyz),a.w*b.w-dot(a.xyz,b.xyz));}
fn parentJoint(joint:i32)->i32{
 ${floating ? `if(joint>=i32(LINK_COUNT)){return select(joint+1,-1,joint+1==i32(N));}` : ''}
 let parent=i32(links[joint].parentAnchor.w);
 return ${floating ? 'select(parent,i32(LINK_COUNT),parent<0)' : 'parent'};
}
fn bodyMass(body:u32)->f32{
 ${floating ? 'if(body==LINK_COUNT){return settings.baseInertiaMass.w;}' : ''}
 return links[body].axisMass.w;
}
fn inertia(i:u32,v:vec3f)->vec3f{
 let q=solved[i].rotation;
 ${floating ? 'if(i==LINK_COUNT){return rotate(q,settings.baseInertiaMass.xyz*rotate(vec4f(-q.xyz,q.w),v));}' : ''}
 return rotate(q,links[i].inertia.xyz*rotate(vec4f(-q.xyz,q.w),v));
}
fn linearColumn(j:i32,point:vec3f)->vec3f{
 ${
   floating
     ? `if(j>=i32(LINK_COUNT)){
  let coordinate=u32(j)-LINK_COUNT;
  var basis=vec3f(0);basis[coordinate%3u]=1.;
  if(coordinate<3u){return basis;}
  return cross(basis,point-solved[LINK_COUNT].position.xyz);
 }`
     : ''
 }
 ${sliders ? 'if(links[j].childAnchor.w!=0.){return axis[j];}' : ''}
 return cross(axis[j],point-pivot[j]);
}
fn angularColumn(j:i32)->vec3f{
 ${
   floating
     ? `if(j>=i32(LINK_COUNT)){
  let coordinate=u32(j)-LINK_COUNT;
  var basis=vec3f(0);
  if(coordinate>=3u){basis[coordinate-3u]=1.;}
  return basis;
 }`
     : ''
 }
 ${sliders ? 'if(links[j].childAnchor.w!=0.){return vec3f(0);}' : ''}
 return axis[j];
}
// Propagate pose, twist and bias acceleration (joint accelerations held at zero).
fn kinematics(){
 for(var i=0u;i<LINK_COUNT;i++){
  let link=links[i];let parent=i32(link.parentAnchor.w);
  var previous=Pose(vec4f(settings.base.xyz,0),vec4f(0,0,0,1),vec4f(0),vec4f(0));
  ${floating ? 'previous=solved[LINK_COUNT];' : ''}
  var linearBias=vec3f(0);var angularBias=vec3f(0);
  if(parent>=0){previous=solved[parent];linearBias=acceleration[parent];angularBias=angularAcceleration[parent];}
  let offset=rotate(previous.rotation,link.parentAnchor.xyz);
  pivot[i]=previous.position.xyz+offset;
  axis[i]=rotate(previous.rotation,link.axisMass.xyz);
  ${
    sliders
      ? `if(link.childAnchor.w!=0.){
   let rotation=normalize(product(previous.rotation,link.rest));
   let travel=offset+axis[i]*angle[i]+rotate(rotation,-link.childAnchor.xyz);
   let rate=axis[i]*speed[i];let spin=previous.spin.xyz;
   solved[i]=Pose(vec4f(previous.position.xyz+travel,1),rotation,vec4f(previous.velocity.xyz+cross(spin,travel)+rate,0),previous.spin);
   acceleration[i]=linearBias+cross(angularBias,travel)+cross(spin,cross(spin,travel))+2.*cross(spin,rate);
   angularAcceleration[i]=angularBias;continue;
  }`
      : ''
  }
  let halfAngle=0.5*angle[i];
  let rotation=normalize(product(product(previous.rotation,vec4f(link.axisMass.xyz*sin(halfAngle),cos(halfAngle))),link.rest));
  let radius=rotate(rotation,-link.childAnchor.xyz);
  let parentSpin=previous.spin.xyz;
  let spin=parentSpin+axis[i]*speed[i];
  let velocity=previous.velocity.xyz+cross(parentSpin,offset)+cross(spin,radius);
  let pivotBias=linearBias+cross(angularBias,offset)+cross(parentSpin,cross(parentSpin,offset));
  angularAcceleration[i]=angularBias+cross(parentSpin,axis[i]*speed[i]);
  acceleration[i]=pivotBias+cross(angularAcceleration[i],radius)+cross(spin,cross(spin,radius));
  solved[i]=Pose(vec4f(pivot[i]+radius,1),rotation,vec4f(velocity,0),vec4f(spin,0));
 }
}
${
  boxCount
    ? `${worldBoxDeclaration}
@group(0) @binding(${publishMatrices ? 7 : 6}) var<storage,read_write> boxes:array<WorldBox>;`
    : ''
}
${freePayload || boxCount ? getPayloadDeclarations(boxCount, colliderCount) : ''}
${loop ? articulationLoopShader : ''}
fn finite(v:vec4f)->bool{return all(abs(v)<=vec4f(3.4028234e38));}
@compute @workgroup_size(1) fn main(){
 if(counters[1]!=0u){return;}
 let dt=settings.gravityStep.w;
 var mass:array<array<f32,N>,N>;var inverse:array<array<f32,N>,N>;
 var force:array<f32,N>;var impulse:array<f32,N>;
 for(var i=0u;i<N;i++){
  inverse[i][i]=1.;
  if(i>=LINK_COUNT){continue;}
  angle[i]=coordinates[i].x;speed[i]=coordinates[i].y;
  // A fixed link contributes its mass to ancestors and has no generalized motion.
  if(dot(links[i].axisMass.xyz,links[i].axisMass.xyz)==0){mass[i][i]=1;}
 }
 ${
   floating
     ? `let basePose=poses[LINK_COUNT];
 solved[LINK_COUNT]=basePose;
 for(var component=0u;component<3u;component++){
  speed[LINK_COUNT+component]=basePose.velocity[component];
  speed[LINK_COUNT+3u+component]=basePose.spin[component];
 }`
     : ''
 }
 kinematics();
 for(var body=0u;body<${floating ? 'LINK_COUNT+1u' : 'LINK_COUNT'};body++){
  let center=solved[body].position.xyz;let spin=solved[body].spin.xyz;
  let linkMass=bodyMass(body);
  let load=linkMass*(settings.gravityStep.xyz-acceleration[body]);
  let moment= -inertia(body,angularAcceleration[body])-cross(spin,inertia(body,spin));
  for(var row=i32(body);row>=0;row=parentJoint(row)){
   let linear=linearColumn(row,center);
   force[row]+=dot(linear,load)+dot(angularColumn(row),moment);
   let angularMass=inertia(body,angularColumn(row));
   for(var column=i32(body);column>=0;column=parentJoint(column)){
    mass[row][column]+=linkMass*dot(linear,linearColumn(column,center))+dot(angularMass,angularColumn(column));
   }
  }
 }
 // Positive definite joint inertia admits elimination without pivoting.
 for(var k=0u;k<N;k++){
  let diagonal=mass[k][k];if(!(diagonal>0)){counters[1]=1u;return;}
  for(var j=0u;j<N;j++){mass[k][j]/=diagonal;inverse[k][j]/=diagonal;}
  for(var i=0u;i<N;i++){if(i!=k){
   let factor=mass[i][k];
   for(var j=0u;j<N;j++){mass[i][j]-=factor*mass[k][j];inverse[i][j]-=factor*inverse[k][j];}
  }}
 }
 for(var i=0u;i<N;i++){for(var j=0u;j<N;j++){speed[i]+=dt*inverse[i][j]*force[j];}}
 ${
   contact
     ? `// Closest point on a finite AABB. Inside, use the nearest exit face.
 let body=u32(settings.boxExtentsLink.w);
 let center=solved[body].position.xyz;
 let local=center-settings.boxCenterRadius.xyz;
 let extent=settings.boxExtentsLink.xyz;
 let outside=local-clamp(local,-extent,extent);
 let distance=length(outside);
 var normal=vec3f(0);
 var gap=distance-settings.boxCenterRadius.w;
 if(distance>0.0){normal=outside/distance;}else{
  let depth=extent-abs(local);var face=0u;
  if(depth.y<depth.x){face=1u;}if(depth.z<depth[face]){face=2u;}
  normal[face]=select(-1.0,1.0,local[face]>=0.0);
  gap= -depth[face]-settings.boxCenterRadius.w;
 }
 var jacobian:array<f32,N>;var response:array<f32,N>;
 for(var joint=i32(body);joint>=0;joint=parentJoint(joint)){
  jacobian[joint]=dot(normal,linearColumn(joint,center));
 }
 var inverseContactMass=0.0;
 for(var row=0u;row<N;row++){
  for(var column=0u;column<N;column++){response[row]+=inverse[row][column]*jacobian[column];}
  inverseContactMass+=jacobian[row]*response[row];
 }
 // Speculate across the remaining gap; correct existing overlap gradually.
 let separationSpeed=select(-gap/dt,min(2.0,0.2*max(0.0,-gap-0.0001)/dt),gap<0.0);
 var normalImpulse=0.0;`
     : ''
 }
${freePayload || boxCount ? 'preparePayload(dt,&inverse);if(counters[1]!=0u){return;}' : ''}
${loop ? 'var closure=prepareLoop(dt,&inverse);if(counters[1]!=0u){return;}' : ''}
 // Implicit spring/damper impulses in coupled joint space. Project each update
 // onto its torque budget and propagate the resulting velocity change to all axes.
 for(var sweep=0u;sweep<64u;sweep++){
  var change=0.0;
  for(var i=0u;i<LINK_COUNT;i++){
   let drive=drives[i];let resistance=dt*(drive.z+dt*drive.y);
   let residual=dt*drive.y*(drive.x-angle[i])-resistance*speed[i]-impulse[i];
   let next=clamp(impulse[i]+residual/(1.0+resistance*inverse[i][i]),-dt*drive.w,dt*drive.w);
   let delta=next-impulse[i];impulse[i]=next;change=max(change,abs(delta));
   for(var j=0u;j<N;j++){speed[j]+=inverse[j][i]*delta;}
  }
  ${
    contact
      ? `if(inverseContactMass>1e-9){
   var normalSpeed=0.0;
   for(var j=0u;j<N;j++){normalSpeed+=jacobian[j]*speed[j];}
   let next=max(0.0,normalImpulse+(separationSpeed-normalSpeed)/inverseContactMass);
   let delta=next-normalImpulse;normalImpulse=next;change=max(change,abs(delta));
   for(var j=0u;j<N;j++){speed[j]+=response[j]*delta;}
  }`
      : ''
  }
${freePayload || boxCount ? 'change=max(change,solvePayload());' : ''}
${loop ? 'change=max(change,solveLoop(&closure));' : ''}
  if(change<1e-7){break;}
 }
 for(var i=0u;i<LINK_COUNT;i++){angle[i]+=dt*speed[i];}
 ${
   floating
     ? `let baseVelocity=vec3f(speed[LINK_COUNT],speed[LINK_COUNT+1u],speed[LINK_COUNT+2u]);
 let baseSpin=vec3f(speed[LINK_COUNT+3u],speed[LINK_COUNT+4u],speed[LINK_COUNT+5u]);
 let baseRotation=normalize(basePose.rotation+0.5*dt*product(vec4f(baseSpin,0),basePose.rotation));
 solved[LINK_COUNT]=Pose(basePose.position+dt*vec4f(baseVelocity,0),baseRotation,vec4f(baseVelocity,0),vec4f(baseSpin,0));
 if(!finite(solved[LINK_COUNT].position)||!finite(baseRotation)||!finite(vec4f(baseVelocity,0))||!finite(vec4f(baseSpin,0))){counters[1]=1u;return;}`
     : ''
 }
 kinematics();
 for(var i=0u;i<LINK_COUNT;i++){let pose=solved[i];
  if(!finite(vec4f(angle[i],speed[i],impulse[i]/dt,0))||!finite(pose.position)||!finite(pose.rotation)||!finite(pose.velocity)||!finite(pose.spin)){counters[1]=1u;return;}
 }
 ${
   freePayload
     ? `payload.position+=dt*payload.velocity;
 payload.rotation=normalize(payload.rotation+0.5*dt*product(vec4f(payload.spin.xyz,0),payload.rotation));
 if(!finite(payload.position)||!finite(payload.rotation)||!finite(payload.velocity)||!finite(payload.spin)){counters[1]=1u;return;}
 poses[N]=payload;`
     : ''
 }
${
  colliderCount
    ? `publishLinkContacts();if(counters[1]!=0u){return;}`
    : boxCount
      ? `for(var index=0u;index<${boxCount}u;index++){
 if(!finite(vec4f(velocities[index],0))||!finite(vec4f(spins[index],0))){counters[1]=1u;return;}
 }
 for(var index=0u;index<${boxCount}u;index++){
 for(var pad=0u;pad<2u;pad++){
   if(index==0u){counters[2u+pad]=0u;}
   if(contacts[index*2u+pad].rows[0].accumulated>1e-8){counters[2u+pad]|=1u<<index;}
 }
 boxes[index].velocity=vec4f(velocities[index],boxes[index].velocity.w);boxes[index].spin=vec4f(spins[index],0);
 }`
      : ''
}
 for(var i=0u;i<LINK_COUNT;i++){coordinates[i]=vec4f(angle[i],speed[i],impulse[i]/dt,0);poses[i]=solved[i];}
 ${floating ? 'poses[LINK_COUNT]=solved[LINK_COUNT];' : ''}
 ${contact ? 'counters[2]=bitcast<u32>(normalImpulse/dt);' : ''}
 ${
   publishMatrices
     ? `for(var i=0u;i<${freePayload || floating ? 'LINK_COUNT+1u' : 'LINK_COUNT'};i++){
  let pose=poses[i];let q=pose.rotation;
  matrices[i]=mat4x4f(vec4f(rotate(q,vec3f(1,0,0)),0),vec4f(rotate(q,vec3f(0,1,0)),0),vec4f(rotate(q,vec3f(0,0,1)),0),pose.position);
 }`
     : ''
 }
 counters[0]++;
}
`;
}
