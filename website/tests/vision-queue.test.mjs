import test from 'node:test';
import assert from 'node:assert/strict';
import {retryDelay,fetchWithRetry,wait} from '../cool-demo/vision/queue.mjs';
test('retry delay honors seconds and HTTP dates with exponential fallback',()=>{
 assert.equal(retryDelay('3',0),3000);
 assert.equal(retryDelay(null,2),4000);
 assert.equal(retryDelay('Thu, 01 Jan 1970 00:00:10 GMT',0,1000),9000);
});
test('429 retries preserve the request and succeed after a bounded backoff',async()=>{
 let calls=0;const delays=[];
 const r=await fetchWithRetry('https://example.test',{method:'POST',body:'request'},{fetcher:async(u,o)=>{assert.equal(o.body,'request');calls++;return new Response('{}',{status:calls<3?429:200,headers:{'Retry-After':'2'}});},sleep:async ms=>delays.push(ms)});
 assert.equal(r.status,200);assert.equal(calls,3);assert.deepEqual(delays,[2000,2000]);
});
test('retry limit and non-rate-limit failures stop immediately',async()=>{
 let calls=0;const fetcher=async()=>{calls++;return new Response('{}',{status:429});};
 assert.equal((await fetchWithRetry('x',{}, {fetcher,sleep:async()=>{}})).status,429);assert.equal(calls,4);
 calls=0;await fetchWithRetry('x',{}, {fetcher:async()=>{calls++;return new Response('{}',{status:422});}});assert.equal(calls,1);
});
test('cancel interrupts backoff',async()=>{const c=new AbortController();const pending=wait(10000,c.signal);c.abort();await assert.rejects(pending,{name:'AbortError'});});
