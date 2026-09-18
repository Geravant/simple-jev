import test from 'node:test';
import assert from 'node:assert/strict';
import {buildRequest,answerFrom,categories} from '../cool-demo/vision/request.mjs';
test('vision request includes image content, fixed categories, and no catalog hints',()=>{
 const r=buildRequest('featherless-ai/Qwen3.8-27B-classifier','data:image/jpeg;base64,YWJj');
 assert.equal(r.messages[0].content[1].image_url.url,'data:image/jpeg;base64,YWJj');
 assert.equal(r.state,undefined);
 assert.deepEqual(Object.keys(r.questions.food.criteria),['hot_dog','sandwich','neither']);
 assert.ok(!JSON.stringify(r).includes('Unsplash'));
 assert.throws(()=>buildRequest('featherless-ai/RWKV-small-classifier','data:image/jpeg;base64,YWJj'),/vision model/);
});
test('invalid or incomplete answer probabilities cannot be displayed as results',()=>{
 const a={choice:'hot_dog',probabilities:{hot_dog:0.8,sandwich:0.15,neither:0.05}};
 assert.equal(answerFrom({answers:{food:a}}),a);
 for(const bad of [{...a,choice:'pizza'},{...a,probabilities:{hot_dog:1}},{...a,probabilities:{...a.probabilities,neither:NaN}}])assert.throws(()=>answerFrom({answers:{food:bad}}));
});

test('batch uses one shared image context with uniquely keyed questions',async()=>{
 const {buildBatchRequest}=await import('../cool-demo/vision/request.mjs');
 const photos=Array.from({length:5},(_,i)=>({id:String(i),image:'data:image/jpeg;base64,YWJj'}));
 const request=buildBatchRequest('Qwen',photos);
 assert.equal(request.messages.length,1);
 assert.equal(request.messages[0].content.filter(x=>x.type==='image_url').length,5);
 assert.deepEqual(Object.keys(request.questions),photos.map(x=>`photo_${x.id}`));
 assert.throws(()=>buildBatchRequest('Qwen',[photos[0],photos[0]]),/unique/);
});
