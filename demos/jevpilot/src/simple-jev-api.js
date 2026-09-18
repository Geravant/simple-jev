/* Browser-only Simple Jev adapter. No accounts, proxy, or stored credentials.
 * Compact observations keep the driving demo near the public context budget.
 * Physics continues during inference; stale plans and hazards trigger local braking.
 * Original simulation and safety checks are retained from Standard Agents. */
import { prepareJevRequest, expandJevAnswers } from './jev-request.js';
import { decisionSelection } from './planning.js';
const BASE = 'https://simple-jev-demo-api.featherless.ai/v1';
// Production input prices per million tokens; developer-plan beta pricing.
export const MODELS = [
  {id:'featherless-ai/Qwen3.6-35B-A3B-classifier',label:'Qwen3.6 35B A3B · MoE',rate:0.28},
  {id:'featherless-ai/gemma-4-26B-A4B-classifier',label:'Gemma 4 26B A4B · MoE',rate:0.28},
  {id:'featherless-ai/Qwen3.8-27B-classifier',label:'Qwen3.8 27B',rate:0.30},
  {id:'featherless-ai/RWKV-small-classifier',label:'RWKV Small',rate:0.03},
  {id:'featherless-ai/RWKV-mid-classifier',label:'RWKV Mid',rate:0.10},
  {id:'featherless-ai/RWKV-std-classifier',label:'RWKV Standard',rate:0.20},
];
export let MODEL = MODELS[0].id;
export function setModel(id) {
  if (!MODELS.some(model=>model.id===id)) throw new Error('Unknown driving model');
  MODEL=id;
}
export function pricingFor(id=MODEL) {
  return {input_per_million:MODELS.find(model=>model.id===id).rate,output_per_million:0,source:'Featherless developer-plan beta pricing; subject to change. See https://featherless.ai'};
}
let nextCall = 0;
const round = x => Number.isFinite(x) ? Math.round(x * 10) / 10 : null;
export function preparePilotRequest(full, model=MODEL) {
  const prepared = prepareJevRequest(full);
  const columns = ['speed','steer','route_error','offroad_fraction','collision','stop_at_line'];
  prepared.request.model = model;
  prepared.request.state = {
    speed_mps: round(full.speed_mps),
    speed_ceiling_mps: round(full.speed_ceiling_mps),
    on_road: full.road?.on_road,
    recovery: !!full.recovery?.active,
    stop_reasons: full.stop_availability?.reasons,
    intersection: full.scene?.intersection ? {
      control: full.scene.intersection.control,
      distance_to_line_m: round(full.scene.intersection.stop_line_ahead_m),
      signal: full.scene.intersection.signal,
      stop_completed: full.scene.intersection.stop_completed,
      already_entered: full.scene.intersection.already_entered,
    } : null,
    columns,
    candidates: Object.fromEntries(Object.entries(prepared.aliases).map(([alias,id]) => {
      const v=full.vectors[id];
      return [alias,[round(v.velocity_mps),round(v.steering),round(v.route_error_m),round(v.offroad_fraction),!!v.collision_predicted,!!v.stop_at_line]];
    })),
  };
  // Local navigation retains the current route; the API chooses motion and path.
  delete prepared.request.questions.route;
  delete prepared.fixed.route;
  if (prepared.request.questions.vector) prepared.request.questions.vector.instructions = 'Choose a safe driving path. Avoid collisions and offroad paths; leave a generous gap behind traffic. Minimize route error, then prefer useful speed. When a stop is required, choose a stop_at_line approach. Recovery may use reverse.';
  return prepared;
}
export async function evaluatePilot(state, signal) {
  // Capture model and price together so switching cannot relabel an in-flight response.
  const model=MODEL, pricing=pricingFor(model);
  const started=performance.now(), prepared=preparePilotRequest(state,model);
  let data={answers:{},usage:{input_tokens:0,output_tokens:0}};
  const apiCall=Object.keys(prepared.request.questions).length>0;
  if(apiCall) {
    await new Promise(resolve=>setTimeout(resolve,Math.max(0,nextCall-Date.now())));
    signal?.throwIfAborted();
    nextCall=Date.now()+300;
    const response=await fetch(`${BASE}/classifier`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(prepared.request),signal,credentials:'omit',redirect:'error'});
    data=await response.json();
    if(!response.ok) {
      if(response.status===429) nextCall=Date.now()+Math.max(3000,(Number(response.headers.get('Retry-After'))||0)*1000);
      throw new Error(data.error?.message || (typeof data.detail==='string' ? data.detail : `Classifier HTTP ${response.status}`));
    }
  }
  const answers=expandJevAnswers(prepared,data.answers);
  const selection=decisionSelection(state,answers);
  if(!selection || !state.vectors[selection.choice]) throw new Error('Classifier returned an invalid driving choice.');
  const v=state.vectors[selection.choice];
  if((v.collision_imminent ?? v.collision_predicted) && v.velocity_mps!==0) throw new Error('Selected path risks collision. Holding the car before retry.');
  return {model,decision_source:apiCall?'jev':'only_eligible_action',request_bytes:new TextEncoder().encode(JSON.stringify(prepared.request)).length,candidate_ids:prepared.aliases,resolved_single_choices:Object.keys(prepared.fixed),answers,selection,batch_id:state.batch_id,controls:{steering:v.steering,velocity:v.velocity_mps},usage:data.usage || {input_tokens:0,output_tokens:0},latency_ms:Math.round(performance.now()-started),cost_usd:(data.usage?.input_tokens ?? 0)*pricing.input_per_million/1_000_000,pricing};
}
export async function pilotFetch(path, options={}) {
  if(path==='/api/status') return Response.json({auth_required:false,authenticated:false,configured:true,pricing:pricingFor()});
  try {return Response.json(await evaluatePilot(JSON.parse(options.body).state,options.signal));}
  catch(error) {return Response.json({error:error.message},{status:502});}
}
