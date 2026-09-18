import test from 'node:test';
import assert from 'node:assert/strict';
import {Simulation} from '../src/simulation.js';
import {preparePilotRequest,evaluatePilot,MODEL} from '../src/simple-jev-api.js';
test('compact requests cover each world with model and original candidate aliases',()=>{
 for(const type of ['town','city','highway']) {
  const state=new Simulation(42,type).decisionState();
  const p=preparePilotRequest(state);
  assert.equal(p.request.model,MODEL);
  assert.ok(JSON.stringify(p.request).length<6500);
  assert.ok(Object.values(p.aliases).every(id=>state.vectors[id]));
  assert.equal(p.request.questions.route,undefined);
 }
});
test('real adapter maps classifier choices to simulation controls and rejects invalid answers',async()=>{
 const state=new Simulation(42,'city').decisionState();
 const original=globalThis.fetch;
 try {
  globalThis.fetch=async(url,options)=>{
   assert.equal(url,'https://simple-jev-demo-api.featherless.ai/v1/classifier');
   assert.equal(options.headers.Authorization,undefined);
   const req=JSON.parse(options.body);
   return Response.json({answers:Object.fromEntries(Object.entries(req.questions).map(([id,q])=>{
    const ids=Object.keys(q.criteria);return [id,{type:'choice',choice:ids[0],confidence:1,probabilities:Object.fromEntries(ids.map((x,i)=>[x,i===0?1:0]))}];
   })),usage:{input_tokens:500,output_tokens:1}});
  };
  const out=await evaluatePilot(state);
  assert.equal(out.controls.velocity,state.vectors[out.selection.choice].velocity_mps);
  assert.equal(out.cost_usd,0);
  globalThis.fetch=async()=>Response.json({answers:{},usage:{input_tokens:1,output_tokens:1}});
  await assert.rejects(()=>evaluatePilot(state),/invalid driving choice/);
 } finally {globalThis.fetch=original;}
});
