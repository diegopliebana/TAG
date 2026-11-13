import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {initSync,BrowserEngine} from "../pkg/tag_browser.js";

initSync({module:readFileSync(new URL("../pkg/tag_browser_bg.wasm",import.meta.url))});
function request(engine,view,command,generation="1") {
  const reply=JSON.parse(engine.handle(JSON.stringify({request_id:"1",generation,position_version:view?.position_version??null,command})));
  assert.equal(reply.error,null,JSON.stringify(reply.error));return reply.view;
}
function file(engine,command,text="",generation="1") {
  const reply=JSON.parse(engine.handle_file(JSON.stringify({request_id:"2",generation,command}),text));
  assert.equal(reply.error,null,JSON.stringify(reply.error));return reply;
}
function prepare(engine,view) {
  const command = (type,id) => JSON.parse(engine.handle(JSON.stringify({request_id:"3",generation:"1",
    position_version:view.position_version,command:{type,...(id?{proposal:id}:{})}})));
  let reply=command("prepare_ai"),polls=0;
  while(reply.proposal.pending) {assert.ok(polls++<10000);reply=command("poll_ai",reply.proposal.id);}
  assert.equal(reply.error,null,JSON.stringify(reply.error));
  return reply.proposal;
}
test("actual Wasm The Game MCTS places a card from the browser controller",()=>{
  const engine=new BrowserEngine();
  try {
    let view=request(engine,null,{type:"create",game:"thegame",
      parameters:{seed:"42",first_player:0,communication:"LimitedChoices"},
      controllers:[{kind:"mcts",seed:"17",configuration:{schema_version:1,
        structured_actions:true,rollout_length:0,budgets:{iterations:16},
        state_evaluator:{name:"game_heuristic"}}}]},"0");
    assert.equal(view.controllers[0].kind,"mcts");
    view=request(engine,view,{type:"commit_ai",proposal:prepare(engine,view).id});
    assert.equal(view.board.thegame.placements.length,1);
  } finally {engine.free();}
});
test("actual Wasm The Game neural MCTS uses a loaded model for speech and placement",()=>{
  const engine=new BrowserEngine();
  try {
    const model=readFileSync(new URL("./fixtures/mcts/thegame-two.json",import.meta.url),"utf8");
    const id=file(engine,{type:"load_model"},model,"0").text;
    const neural={kind:"neural_mcts",seed:"17",model_id:id,
      bindings:{value:"network_value",policy:"network_policy",value_observers:"deciding_player"},
      configuration:{schema_version:1,structured_actions:true,tree_policy:"puct",rollout_length:0,
        budgets:{iterations:8},state_evaluator:{name:"network_value"},
        puct_preferences:{source:"probabilities",evaluator:{name:"network_policy"}}}};
    let view=request(engine,null,{type:"create",game:"thegame",
      parameters:{seed:"42",first_player:0,communication:"LimitedChoices"},
      controllers:[neural,{kind:"human"}]},"0");
    const prediction=JSON.parse(file(engine,{type:"predict_model",model_id:id,observer:0}).text);
    assert.deepEqual(prediction.value_players,[0,1]);
    assert.equal(prediction.logits.length,406);
    assert.equal(prediction.values.length,2);
    view=request(engine,view,{type:"commit_ai",proposal:prepare(engine,view).id});
    view=request(engine,view,{type:"human_move",actor:1,action:view.actions[0]});
    view=request(engine,view,{type:"commit_ai",proposal:prepare(engine,view).id});
    assert.equal(view.board.thegame.placements.length,1);
  } finally {engine.free();}
});
test("actual Wasm The Game protects hands and restores state and replay",()=>{
  const engine=new BrowserEngine();
  try {
    let view=request(engine,null,{type:"create",game:"thegame",parameters:{seed:"42",first_player:0,communication:"LimitedChoices"},controllers:[{kind:"human"},{kind:"human"}]},"0");
    assert.deepEqual(view.outcome_score,{scope:"team",values:[0],unit:"cards_placed",maximum:98});
    assert.equal(view.actions.length,13);
    const firstHand=view.board.thegame.hand;
    view=request(engine,view,{type:"human_move",actor:0,action:view.actions[0]});
    assert.equal(view.observer,1);assert.equal(view.board.thegame.messages.length,0);
    assert.notDeepEqual(view.board.thegame.hand,firstHand);
    const secondHand=view.board.thegame.hand;
    view=request(engine,view,{type:"human_move",actor:1,action:view.actions[0]});
    assert.equal(view.board.thegame.messages.length,1);
    const spectator=request(engine,view,{type:"presentation",observer:null,public_only:true,reveal_hidden_information:false});
    assert.deepEqual(spectator.board.thegame.hand,[]);assert.deepEqual(spectator.actions,[]);
    const revealed=request(engine,view,{type:"presentation",observer:null,public_only:true,reveal_hidden_information:true});
    assert.equal(revealed.board.thegame.opponent_hands.length,2);
    assert.deepEqual(revealed.board.thegame.opponent_hands[0],firstHand);
    assert.deepEqual(revealed.board.thegame.opponent_hands[1],secondHand);
    assert.deepEqual(revealed.board.thegame.hand,[]);assert.deepEqual(revealed.actions,[]);
    for(const scope of ["state","branch","session"]) {
      const saved=file(engine,{type:"export",scope,cursor:view.cursor,delay_ms:0}).text;
      const restored=new BrowserEngine();
      try {
        const result=file(restored,{type:"import",controllers:scope==="state"?[{kind:"human"},{kind:"human"}]:null},saved,"0");
        assert.deepEqual(result.live_view.board.thegame.pile_tops,view.board.thegame.pile_tops);
        assert.deepEqual(result.live_view.board.thegame.messages,view.board.thegame.messages);
      } finally {restored.free();}
    }
  } finally {engine.free();}
});

test("actual Wasm The Game finishes one- and five-seat Random teams",()=>{
  for(const seats of [1,5]) {
    const engine=new BrowserEngine();
    try {
      const controllers=Array.from({length:seats},(_,seat)=>({kind:"random",seed:String(seat+7)}));
      let view=request(engine,null,{type:"create",game:"thegame",parameters:{seed:"42",first_player:0,communication:"LimitedChoices"},controllers},"0");
      let steps=0;
      while(view.next_player!==null) {
        const version=view.position_version;
        const proposal=JSON.parse(engine.handle(JSON.stringify({request_id:"1",generation:"1",position_version:version,command:{type:"prepare_ai"}})));
        assert.equal(proposal.error,null,JSON.stringify(proposal.error));
        view=request(engine,view,{type:"commit_ai",proposal:proposal.proposal.id});
        assert.ok(++steps<500);
      }
      assert.ok(["shared_win","team_loss"].includes(view.board.outcome.kind));
      assert.ok(view.results.every(result=>result===view.results[0]));
      assert.equal(view.outcome_score.scope,"team");
      assert.equal(view.outcome_score.unit,"cards_placed");
      assert.equal(view.outcome_score.values[0],view.board.counters[0]);
    } finally {engine.free();}
  }
});
