import test from 'node:test';
import assert from 'node:assert/strict';
import {followingSpeed} from '../src/traffic-safety.js';

test('approaching a stopped car brakes to a four metre gap without creeping closer',()=>{
 const car={speed:15,heading:0};
 const lead={gap:65,other:{type:'car',speed:0,heading:0}};
 for(let i=0;i<2000;i++) {
  const cap=followingSpeed(car,lead);
  car.speed=Math.max(0,car.speed+Math.max(-5*.02,Math.min(2*.02,cap-car.speed)));
  lead.gap-=car.speed*.02;
  assert.ok(lead.gap>=3.95,`gap fell to ${lead.gap}`);
 }
 assert.ok(car.speed<0.01);
 assert.ok(lead.gap<4.1);
});
test('a moving lead gets an earlier slowdown and adjacent traffic is not needed for the cap',()=>{
 const car={speed:15,heading:0};
 const lead={gap:12,other:{type:'car',speed:10,heading:0}};
 assert.ok(followingSpeed(car,lead)<10);
 assert.equal(followingSpeed(car,null),Infinity);
});
