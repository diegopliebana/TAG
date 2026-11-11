import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initSync, BrowserEngine } from "../pkg/tag_browser.js";

initSync({module:readFileSync(new URL("../pkg/tag_browser_bg.wasm",import.meta.url))});
function request(engine, view, command, generation="1") {
  const reply=JSON.parse(engine.handle(JSON.stringify({request_id:"1",generation,position_version:view?.position_version??null,command})));
  assert.equal(reply.error,null,JSON.stringify(reply.error));return reply;
}
function file(engine,command,text="",generation="1") {
  const reply=JSON.parse(engine.handle_file(JSON.stringify({request_id:"2",generation,command}),text));
  assert.equal(reply.error,null,JSON.stringify(reply.error));return reply;
}
function prepare(engine,view) {
  let reply=request(engine,view,{type:"prepare_ai"});let polls=0;
  while(reply.proposal.pending) {assert.ok(polls++<10000);reply=request(engine,view,{type:"poll_ai",proposal:reply.proposal.id});}
  return reply.proposal.id;
}

test("automatic Sushi Go! observer changes preserve one simultaneous decision",()=>{
  for(const controllers of [
    [{kind:"human"},{kind:"human"},{kind:"human"}],
    [{kind:"human"},{kind:"first_available",seed:"17"},{kind:"human"}],
  ]) {
    const engine=new BrowserEngine();
    try {
      let view=request(engine,null,{type:"create",game:"sushigo",parameters:{seed:"42"},controllers},"0").view;
      const initial=Array.from({length:3},(_,seat)=>request(engine,view,{type:"presentation",observer:seat,public_only:false,reveal_hidden_information:false}).view);
      view=request(engine,view,{type:"presentation",observer:null,public_only:false,reveal_hidden_information:false}).view;
      const version=view.position_version;
      const played=view.board.sushigo.played;
      const first=initial[0].actions[0];
      view=request(engine,view,{type:"human_move",actor:0,action:first}).view;
      assert.equal(view.observer,1);
      assert.equal(view.position_version,version);
      assert.equal(view.cursor,0);
      assert.deepEqual(view.board.sushigo.played,played);
      assert.deepEqual(view.board.sushigo.last_reveal,[]);
      const repeated=JSON.parse(engine.handle(JSON.stringify({request_id:"1",generation:"1",position_version:version,command:{type:"human_move",actor:0,action:first}})));
      assert.equal(repeated.error?.code,"invalid_action");
      assert.equal(request(engine,view,{type:"view",observer:null}).view.position_version,version);
      if(controllers[1].kind==="human") {
        assert.deepEqual(view.board.sushigo.hand,initial[1].board.sushigo.hand);
        assert.deepEqual(view.actions,initial[1].actions);
        view=request(engine,view,{type:"human_move",actor:1,action:view.actions[0]}).view;
      } else {
        assert.deepEqual(view.board.sushigo.hand,[]);
        assert.deepEqual(view.actions,[]);
        view=request(engine,view,{type:"commit_ai",proposal:prepare(engine,view)}).view;
      }
      assert.equal(view.observer,2);
      assert.equal(view.position_version,version);
      assert.equal(view.cursor,0);
      assert.deepEqual(view.board.sushigo.played,played);
      assert.deepEqual(view.board.sushigo.hand,initial[2].board.sushigo.hand);
      assert.deepEqual(view.actions,initial[2].actions);
      view=request(engine,view,{type:"human_move",actor:2,action:view.actions[0]}).view;
      assert.notEqual(view.position_version,version);
      assert.equal(view.cursor,1);
      assert.equal(view.accepted_move.action.length,3);
      assert.deepEqual(view.accepted_move.action.map(choice=>choice.actor),[0,1,2]);
      assert.ok(view.board.sushigo.played.every(cards=>cards.length===1));
      assert.deepEqual(view.waiting_seats,[0,1,2]);
    } finally {engine.free();}
  }
});

test("actual Wasm Sushi Go! resolves two and five seat decisions and restores exact continuation",()=>{
  for(const players of [2,5]) {
    const engine=new BrowserEngine(), restored=new BrowserEngine();
    try {
      const controllers=Array.from({length:players},(_,i)=>({kind:"first_available",seed:String(i+10)}));
      let view=request(engine,null,{type:"create",game:"sushigo",parameters:{seed:"123"},controllers},"0").view;
      assert.deepEqual(view.waiting_seats,Array.from({length:players},(_,i)=>i));
      const start=view.position_version;
      const proposal=prepare(engine,view);
      view=request(engine,view,{type:"commit_ai",proposal}).view;
      assert.equal(view.position_version,start);
      assert.equal(view.cursor,0);
      assert.deepEqual(view.board.sushigo.hand,[]);
      const forbidden=request(engine,null,{type:"view",observer:1}).view;
      assert.deepEqual(forbidden.board.sushigo.hand,[]);
      while(view.waiting_seats.length<players) view=request(engine,view,{type:"commit_ai",proposal:prepare(engine,view)}).view;
      assert.equal(view.cursor,1);
      const saved=file(engine,{type:"export",scope:"session",cursor:1,delay_ms:0}).text;
      let other=file(restored,{type:"import",controllers:null},saved,"0").live_view;
      let decisions=1;
      while(view.next_player!==null) {
        for(let seat=0;seat<players;seat++) {
          view=request(engine,view,{type:"commit_ai",proposal:prepare(engine,view)}).view;
          other=request(restored,other,{type:"commit_ai",proposal:prepare(restored,other)}).view;
        }
        assert.deepEqual(view.board,other.board);
        assert.ok(++decisions<=30);
      }
      assert.equal(view.board.sushigo.round,3);
      assert.equal(view.board.sushigo.round_scores.length,3);
      assert.equal(view.board.sushigo.finished,true);
      assert.ok(view.results.some(result=>result==="win"));
    } finally {engine.free();restored.free();}
  }
});

test("actual Wasm keeps ordered Chopsticks choices hidden until both seats reveal",()=>{
  let selected=null;
  for(let seed=0;seed<128 && !selected;seed++) {
    const engine=new BrowserEngine();
    const view=request(engine,null,{type:"create",game:"sushigo",parameters:{seed:String(seed)},controllers:[{kind:"human"},{kind:"human"}]},"0").view;
    if(view.board.sushigo.hand.includes("Chopsticks")) selected={engine,view};
    else engine.free();
  }
  assert.ok(selected,"Expected a controlled initial hand with Chopsticks");
  const {engine}=selected;
  try {
    let view=selected.view;
    const initialVersion=view.position_version;
    view=request(engine,view,{type:"human_move",actor:0,action:{Pick:"Chopsticks"}}).view;
    assert.equal(view.position_version,initialVersion);
    assert.deepEqual(view.board.sushigo.last_reveal,[]);
    assert.equal(view.observer,1);
    assert.ok(view.actions.length>0);
    assert.equal(view.board.sushigo.hand.length,10);
    view=request(engine,view,{type:"human_move",actor:1,action:view.actions[0]}).view;
    assert.equal(view.cursor,1);
    assert.equal(view.observer,0);
    const ordered=view.actions.find(action=>action.Chopsticks && action.Chopsticks.first!==action.Chopsticks.second);
    assert.ok(ordered,"Chopsticks should offer an ordered two-card choice");
    const sameVersion=view.position_version;
    view=request(engine,view,{type:"human_move",actor:0,action:ordered}).view;
    assert.equal(view.position_version,sameVersion);
    assert.equal(view.cursor,1);
    assert.equal(view.observer,1);
    view=request(engine,view,{type:"human_move",actor:1,action:view.actions[0]}).view;
    assert.equal(view.cursor,2);
    assert.deepEqual(view.board.sushigo.last_reveal[0],[0,ordered]);
  } finally {engine.free();}
});

test("actual Wasm privileged Sushi Go! reveal shows opponent hands while played cards stay public",()=>{
  const engine=new BrowserEngine();
  try {
    let view=request(engine,null,{type:"create",game:"sushigo",parameters:{seed:"41"},controllers:[{kind:"human"},{kind:"human"}]},"0").view;
    assert.equal(view.board.sushigo.opponent_hands,undefined);
    assert.equal(view.board.sushigo.known_hands,undefined);
    const original=view.actions;
    view=request(engine,view,{type:"presentation",observer:null,public_only:false,reveal_hidden_information:true}).view;
    assert.equal(view.board.sushigo.opponent_hands[0],null);
    assert.equal(view.board.sushigo.opponent_hands[1].length,10);
    assert.deepEqual(view.actions,original);
    view=request(engine,view,{type:"presentation",observer:null,public_only:false,reveal_hidden_information:false}).view;
    assert.equal(view.board.sushigo.opponent_hands,undefined);
    view=request(engine,view,{type:"human_move",actor:0,action:view.actions[0]}).view;
    assert.equal(view.observer,1);
    assert.equal(view.board.sushigo.known_hands,undefined);
    view=request(engine,view,{type:"human_move",actor:1,action:view.actions[0]}).view;
    assert.ok(view.board.sushigo.played.every(cards=>cards.length===1));
    assert.equal(view.board.sushigo.opponent_hands,undefined);
  } finally {engine.free();}
});
