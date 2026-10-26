import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, BrowserEngine } from "../pkg/tag_browser.js";
initSync({module:readFileSync(new URL("../pkg/tag_browser_bg.wasm",import.meta.url))});
const read = path => readFileSync(new URL(path,import.meta.url),"utf8");
const predictions=JSON.parse(read("./fixtures/mcts/loveletter-predictions.json"));
const native={schema_version:1,rollout_length:0,budgets:{iterations:24}};
const neural={...native,tree_policy:"puct",state_evaluator:{name:"network_value"},puct_preferences:{source:"probabilities",evaluator:{name:"network_policy"}}};
function command(e,v,command,generation="1") {
  const r=JSON.parse(e.handle(JSON.stringify({request_id:"1",generation,position_version:v?.position_version??null,command})));
  assert.equal(r.error,null,JSON.stringify(r.error));return r;
}
function file(e,type,text="",generation="1",extra={}) {
  const r=JSON.parse(e.handle_file(JSON.stringify({request_id:"1",generation,command:{type,...extra}}),text));
  assert.equal(r.error,null,JSON.stringify(r.error));return r;
}
function create(e,n,controllers,seed="42") {
  return command(e,null,{type:"create",game:"loveletter",parameters:{seed,rules:{winning_tokens:[1,1,1]}},controllers:controllers??Array.from({length:n},()=>({kind:"human"}))},"0").view;
}
function presentation(e,v,reveal,observer=null,public_only=false) {
  return command(e,v,{type:"presentation",observer,public_only,reveal_hidden_information:reveal}).view;
}
function prepare(e,v) {
  let r=command(e,v,{type:"prepare_ai"});let polls=0;
  while(r.proposal.pending) { assert.ok(polls++<10000);r=command(e,v,{type:"poll_ai",proposal:r.proposal.id}); }
  assert.equal(r.proposal.action,null);return r.proposal.id;
}
function step(e,v) {return command(e,v,{type:"commit_ai",proposal:prepare(e,v)}).view;}
const close=(a,b)=>assert.ok(Math.abs(a-b)<=2e-6+2e-5*Math.abs(b),`${a} != ${b}`);

test("actual Wasm Love Letter permitted inputs and full vector predictions match PyTorch",()=>{
  for(const f of predictions) {
    const e=new BrowserEngine();
    try {
      const id=file(e,"load_model",read(`./fixtures/mcts/loveletter-${f.players}.json`),"0").text;
      const v=create(e,f.players);
      const p=JSON.parse(file(e,"predict_model","","1",{model_id:id,observer:0}).text);
      assert.deepEqual(p.inputs,f.inputs);assert.deepEqual(p.value_players,Array.from({length:f.players},(_,s)=>s));
      p.logits.forEach((x,i)=>close(x,f.logits[i]));p.values.forEach((x,i)=>close(x,f.values[i]));
      assert.ok(!JSON.stringify(v).includes('"seed":"42"'));
    } finally {e.free();}
  }
});

test("actual Wasm neural/native games finish, reveal preserves suspended search and bundled continuations",()=>{
  for(const n of [2,3,4]) { for (const neuralSeat of [0,n-1]) {
    const e=new BrowserEngine(),restored=new BrowserEngine();
    try {
      const model_id=file(e,"load_model",read(`./fixtures/mcts/loveletter-${n}.json`),"0").text;
      const controllers=Array.from({length:n},(_,s)=>s===neuralSeat?{kind:"neural_mcts",seed:"7",model_id,configuration:neural}
        :s%2?{kind:"sampled_osla",seed:"8",repetitions_per_action:2}:{kind:"mcts",seed:"9",configuration:native});
      let v=create(e,n,controllers);
      assert.deepEqual(v.actions,[]);assert.deepEqual(v.board.loveletter.hand,[]);
      let pending=command(e,v,{type:"prepare_ai"});
      const shown=presentation(e,v,true);assert.ok(shown.board.loveletter.privileged.hands.length===n);
      assert.equal(shown.position_version,v.position_version);
      v=presentation(e,v,false);assert.ok(!v.board.loveletter.privileged);
      while(pending.proposal.pending) pending=command(e,v,{type:"poll_ai",proposal:pending.proposal.id});
      v=command(e,v,{type:"commit_ai",proposal:pending.proposal.id}).view;
      const saved=file(e,"export","","1",{scope:"session",cursor:v.cursor,delay_ms:0}).text;
      const loaded=file(restored,"import",saved,"0",{controllers:null});let other=loaded.live_view;
      assert.ok(!other.board.loveletter.privileged);
      for(let i=0;i<1000 && v.next_player!==null;i++) {
        v=step(e,v);other=step(restored,other);assert.deepEqual(v.board,other.board);
      }
      assert.equal(v.next_player,null);assert.ok(v.results.some(x=>x==="win"));
    } finally {e.free();restored.free();}
  }}
});

test("actual hidden-world equivalents have identical normal payloads and history excludes later facts",()=>{
  const left=new BrowserEngine();const engines=[left];
  try {
    const initial=create(left,3);let rightView;
    for(let seed=1;seed<512;seed++) {
      const right=new BrowserEngine();engines.push(right);
      const candidate=create(right,3,null,String(seed));
      if(JSON.stringify(initial.board.loveletter.hand)===JSON.stringify(candidate.board.loveletter.hand)) {
        rightView=candidate;break;
      }
    }
    assert.ok(rightView);assert.deepEqual(initial.board,rightView.board);assert.deepEqual(initial.actions,rightView.actions);
    let v=initial;
    for(let i=0;i<3;i++) v=command(left,v,{type:"human_move",actor:v.next_player,action:v.actions[0]}).view;
    const old=command(left,v,{type:"history",branch:0,cursor:0,observer:0}).view;
    assert.deepEqual(old.board,initial.board);assert.deepEqual(old.moves,[]);
    const spectator=presentation(left,v,false,null,true);
    assert.deepEqual(spectator.board.loveletter.hand,[]);assert.deepEqual(spectator.board.loveletter.private_events,[]);
  } finally {engines.forEach(e=>e.free());}
});

test("single human view remains during AI turns; multiple humans switch immediately",()=>{
  const e=new BrowserEngine();
  try {
    let v=create(e,3,[{kind:"human"},{kind:"random",seed:"7"},{kind:"human"}]);
    assert.equal(v.observer,0);
    v=command(e,v,{type:"human_move",actor:0,action:v.actions.find(a=>a.play!=="princess")??v.actions[0]}).view;
    if(v.next_player===1) {
      assert.equal(v.observer,0);assert.deepEqual(v.actions,[]);
      v=step(e,v);
    }
    if(v.next_player===2) assert.equal(v.observer,2);
    const fixed=presentation(e,v,false,0);assert.equal(fixed.observer,0);
    presentation(e,v,true,2,true);
    const reset=command(e,v,{type:"reset"}).view;
    assert.equal(reset.observer,0);assert.equal(reset.board.loveletter.public_only,false);
    assert.ok(!reset.board.loveletter.privileged);

  } finally {e.free();}
});



test("invalid Love Letter neural observer modes and completion settings preserve live play",()=>{
  const e=new BrowserEngine();
  try {
    const model_id=file(e,"load_model",read("./fixtures/mcts/loveletter-3.json"),"0").text;
    const original=create(e,3);
    for(const bad of [
      {configuration:neural,bindings:{value:"network_value",policy:"network_policy",value_observers:"all_players"}},
      {configuration:{...neural,value_completion:"fill_zeros"}},
    ]) {
      const request={request_id:"bad",generation:"1",position_version:null,command:{type:"create",game:"loveletter",parameters:{seed:"42",rules:{winning_tokens:[1,1,1]}},
        controllers:[{kind:"neural_mcts",seed:"7",model_id,...bad},{kind:"human"},{kind:"human"}]}};
      const result=JSON.parse(e.handle(JSON.stringify(request)));
      assert.ok(result.error);assert.equal(result.generation,"1");
      assert.deepEqual(command(e,original,{type:"view"}).view.board,original.board);
    }
  } finally {e.free();}
});


test("shared round winners remain multiple browser game winners",()=>{
  const e=new BrowserEngine();
  try {
    let v=create(e,4,null,"65");let count=0;
    while(v.next_player!==null && count++<100) v=command(e,v,{type:"human_move",actor:v.next_player,action:v.actions[0]}).view;
    assert.equal(count,11);assert.deepEqual(v.board.outcome,{kind:"shared_win",player:[1,2]});
    assert.deepEqual(v.results,["loss","win","win","loss"]);
  }finally{e.free();}
});


test("forking a former human into AI retains the remaining human's own view",()=>{
  const e=new BrowserEngine();
  try {
    let v=create(e,3);
    for(let i=0;i<2;i++)v=command(e,v,{type:"human_move",actor:v.next_player,action:v.actions[0]}).view;
    const fork=command(e,v,{type:"fork",branch:0,cursor:2,replacements:[null,{kind:"random",seed:"7"},{kind:"random",seed:"8"}]}).view;
    assert.equal(fork.observer,0);
    assert.deepEqual(fork.actions,[]);
  }finally{e.free();}
});


test("actual Wasm accepts a retained result observer without changing normal next-human selection",()=>{
  let found=false;
  for(let seed=0;seed<128&&!found;seed++) {
    const e=new BrowserEngine();
    try {
      const v=create(e,3,null,String(seed));
      const action=v.actions.find(a=>a.target?.card==="baron"&&a.target.target===1);
      if(!action)continue;
      const after=command(e,v,{type:"human_move",actor:0,action,observer:0}).view;
      assert.equal(after.observer,0);
      assert.ok(after.board.loveletter.private_events.some(e=>e.information.comparison));
      const fresh=command(e,after,{type:"view"}).view;
      assert.equal(fresh.observer,fresh.next_player);
      if(fresh.observer===2)assert.ok(!fresh.board.loveletter.private_events.some(e=>e.information.comparison));
      found=true;
    }finally{e.free();}
  }
  assert.ok(found,"No fixed-seed Baron opening found");
});
