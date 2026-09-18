import test from 'node:test';
import assert from 'node:assert/strict';
import {slide,spawn} from '../cool-demo/2048/engine.js';
import {chooseMove} from '../cool-demo/2048/choice.js';
import {classify,models} from '../cool-demo/2048/api.js';
test('merges once per move, tracks score, and leaves input unchanged',()=>{
 const board=[2,2,2,2,...Array(12).fill(0)];
 const result=slide(board,'LEFT');
 assert.deepEqual(result.board.slice(0,4),[4,4,0,0]);assert.equal(result.score,8);
 assert.deepEqual(board.slice(0,4),[2,2,2,2]);
 assert.deepEqual(slide([2,2,4,0,...Array(12).fill(0)],'RIGHT').board.slice(0,4),[0,0,4,4]);
});
test('vertical movement, unchanged moves, and spawning work',()=>{
 const board=[2,0,0,0,2,0,0,0,...Array(8).fill(0)];
 assert.equal(slide(board,'DOWN').board[12],4);
 assert.equal(slide(board,'LEFT').changed,false);
 assert.deepEqual(spawn(Array(16).fill(2)),Array(16).fill(2));
 assert.equal(spawn(Array(16).fill(0),()=>0)[0],2);
});
test('tie breaks remain within legal moves',()=>{
 assert.equal(chooseMove({UP:1,DOWN:1,LEFT:1},['UP','DOWN'],()=>0.99).action,'DOWN');
});
test('single legal move resolves locally; RWKV images cannot be submitted',async()=>{
 models.push('test-model','featherless-ai/RWKV-small-classifier');
 const original=globalThis.fetch;globalThis.fetch=()=>{throw Error('Unexpected request');};
 try{
 const result=await classify({model:'test-model',board:[...Array(12).fill(0),2,4,8,16]});
 assert.equal(result.action,'UP');assert.equal(result.local,true);
 await assert.rejects(()=>classify({model:'featherless-ai/RWKV-small-classifier',board:[2,...Array(15).fill(0)],mode:'image'}),/Gemma or Qwen/);
 }finally{globalThis.fetch=original;}
});
