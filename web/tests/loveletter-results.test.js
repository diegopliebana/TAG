import test from "node:test";
import assert from "node:assert/strict";
import { actionResult, shouldShowActionResult, ActionResultStep } from "../loveletter-results.js";

test("result screens require an active human unless always-show is explicitly enabled",()=>{
  const view=(kinds,active)=>({controllers:kinds.map(kind=>({kind})),board:{loveletter:{round:1,active}}});
  const human=view(["human","mcts"],[true,true]);
  assert.equal(shouldShowActionResult(human,human),true);
  const eliminated=view(["human","mcts"],[false,true]);
  assert.equal(shouldShowActionResult(human,eliminated),false);
  assert.equal(shouldShowActionResult(human,eliminated,true),true);
  const aiOnly=view(["random","mcts"],[true,true]);
  assert.equal(shouldShowActionResult(aiOnly,aiOnly),false);
  assert.equal(shouldShowActionResult(aiOnly,aiOnly,true),true);
  const anotherHuman=view(["human","mcts","human"],[false,true,true]);
  assert.equal(shouldShowActionResult(anotherHuman,anotherHuman),true);
  anotherHuman.board.loveletter.active[2]=false;
  assert.equal(shouldShowActionResult(anotherHuman,anotherHuman),false);
});

test("the completed round's eligibility controls its result; screens resume for active humans in the new round",()=>{
  const before={controllers:[{kind:"human"},{kind:"random"}],board:{loveletter:{round:1,active:[true,true],public_events:[]}}};
  const after=structuredClone(before);
  after.board.loveletter.round=2;
  after.board.loveletter.last_round={summary:{round:1},events:[{eliminated:{player:0}}]};
  assert.equal(shouldShowActionResult(before,after),false);
  assert.equal(shouldShowActionResult(before,after,true),true);
  assert.equal(shouldShowActionResult(after,after),true);
  after.board.loveletter.last_round.events=[];
  assert.equal(shouldShowActionResult(before,after),true);
});

function transition(events,privateEvents=[],observer=0) {
  const prefix=[{round_started:{round:1,starter:0}}];
  const before={branch:0,cursor:0,board:{loveletter:{round:1,public_events:prefix}}};
  const after={branch:0,cursor:1,observer,board:{loveletter:{round:1,tokens:[0,0,0],public_only:false,
    public_events:[...prefix,...events],private_events:privateEvents,last_private_events:[]}}};
  return {before,after};
}
const played=action=>({played:{actor:0,action}});
const result=t=>actionResult(t.before,t.after);

test("Guard results clearly distinguish hit/miss without revealing an incorrect guess's actual target hand",()=>{
  for(const correct of [false,true]) {
    const t=transition([played({guess:{target:1,card:"priest"}}),{guard_result:{actor:0,target:1,guess:"priest",correct}}]);
    t.after.board.loveletter.privileged={hands:[["king"],["princess"]]};
    const section=result(t).sections[0];
    assert.match(section.title,correct?/Correct guess/:/Incorrect guess/);
    assert.equal(section.cards.length,1);
    assert.equal(section.cards[0].card,"priest");
    assert.equal(section.cards[0].status,correct?"correct":"incorrect");
    assert.equal(result(t).played.card,"guard");
    assert.ok(!JSON.stringify(result(t)).includes("princess"));
    if(correct)assert.equal(section.cards[0].card,"priest");
    else assert.match(section.text,/guess was incorrect/);
  }
});
test("Baron uses private event-time comparison cards; other observers see only public discards",()=>{
  const events=[played({target:{card:"baron",target:1}}),{compared:{actor:0,target:1,eliminated:1}},{discarded:{player:1,card:"guard"}}];
  const t=transition(events,[{event:2,information:{comparison:{opponent:1,own_card:"king",other_card:"guard"}}}]);
  t.after.board.loveletter.hand=["princess"];
  assert.deepEqual(result(t).sections[0].cards.map(c=>c.card),["king","guard"]);
  assert.deepEqual(result(t).sections[0].cards.map(c=>c.status),["winner","eliminated"]);
  assert.match(result(t).sections[0].text,/Player 1.*eliminated/);
  const spectator=transition(events,[],2);
  assert.deepEqual(result(spectator).sections[0].cards.map(c=>c.card),[null,"guard"]);
  assert.ok(!JSON.stringify(result(spectator)).includes("king"));
  events[1].compared.eliminated=null;events.pop();
  assert.match(result(transition(events)).sections[0].text,/Equal ranks/);
});
test("a round rollover retains the completed round's private result and announces shared tokens before the next round",()=>{
  const t=transition([played({target:{card:"baron",target:1}}),{compared:{actor:0,target:1,eliminated:null}},
    {round_ended:{winners:[0,1],reason:"deck_empty",revealed_hands:[[0,"prince"],[1,"prince"]]}}]);
  const c=t.after.board.loveletter;
  c.last_round={summary:{round:1,tokens:[1,1,0]},events:c.public_events};
  c.round=2;c.public_events=[{round_started:{round:2,starter:1}}];
  c.private_events=[{event:1,information:{received:{card:"princess",source:"deal"}}}];
  c.last_private_events=[{event:2,information:{comparison:{opponent:1,own_card:"prince",other_card:"prince"}}}];
  const sections=result(t).sections;
  assert.deepEqual(sections[0].cards.map(c=>c.card),["prince","prince"]);
  assert.match(sections[1].title,/Round 1 won by Player 0 and Player 1/);
  assert.match(sections[1].text,/\+1 favor token \(1 total\)/);
  assert.equal(sections.at(-1).title,"Round 2 begins");
  assert.ok(!JSON.stringify(sections).includes("princess"));
});
test("Priest information is private and a self-targeted Prince distinguishes the played card from the discarded hand",()=>{
  const events=[played({target:{card:"priest",target:1}})];
  const t=transition(events,[{event:1,information:{priest:{target:1,card:"king"}}}]);
  assert.equal(result(t).sections[0].cards[0].card,"king");
  assert.equal(result(transition(events,[],2)).sections[0].cards.length,0);
  const prince=transition([played({target:{card:"prince",target:0}}),{discarded:{player:0,card:"prince"}},
    {discarded:{player:0,card:"princess"}},{eliminated:{player:0}}]);
  assert.equal(result(prince).sections[0].cards[0].card,"princess");
  assert.match(result(prince).sections[0].text,/Princess eliminates/);
});
test("result summaries ignore jumps and announce complete shared game wins",()=>{
  const t=transition([played({play:"handmaid"}),{game_ended:{winners:[0,2]}}]);
  assert.match(result(t).sections[0].text,/Player 0 and Player 2 share the win/);
  t.after.cursor=5;assert.equal(result(t),null);
});

class Element {
  constructor(document){Object.assign(this,{ownerDocument:document,children:[],listeners:{},attributes:{},open:false});}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  addEventListener(type,callback){this.listeners[type]=callback;}
  setAttribute(name,value){this.attributes[name]=value;}
  showModal(){this.open=true;}
  close(){this.open=false;this.listeners.close?.();}
}
test("the result step waits for acknowledgement and removes private card DOM on close or cancellation",async()=>{
  const document={createElement:()=>new Element(document)};
  const dialog=new Element(document),title=new Element(document),body=new Element(document);
  const step=new ActionResultStep(dialog,title,body);
  const summary={title:"Baron",observer:0,sections:[{title:"Comparison",text:"You win",cards:[{card:"king",label:"Your card"}]}]};
  let resumed=false;const wait=step.show(summary).then(ack=>{resumed=ack;});
  await Promise.resolve();assert.equal(resumed,false);assert.equal(dialog.open,true);assert.ok(body.children.length);
  dialog.close();await wait;assert.equal(resumed,true);assert.deepEqual(body.children,[]);
  const cancelled=step.show(summary);step.cancel();assert.equal(await cancelled,false);assert.deepEqual(body.children,[]);
});

test("action information screens render played and guessed cards with accessible hit/miss overlays",async()=>{
  const document={createElement:()=>new Element(document)};
  const dialog=new Element(document),title=new Element(document),body=new Element(document);
  const step=new ActionResultStep(dialog,title,body);
  const nodes=e=>[e,...e.children.flatMap(nodes)];
  for(const correct of [false,true]) {
    const t=transition([played({guess:{target:1,card:"priest"}}),{guard_result:{actor:0,target:1,guess:"priest",correct}}]);
    const wait=step.show(result(t));
    const rendered=nodes(body);
    assert.ok(rendered.some(e=>e.alt==="1 Guard"));
    assert.ok(rendered.some(e=>e.alt==="2 Priest"));
    const mark=rendered.find(e=>e.attributes["aria-label"]===(correct?"Correct guess":"Incorrect guess"));
    assert.ok(mark);assert.equal(mark.textContent,correct?String.fromCodePoint(0x2713):"X");
    assert.match(mark.className,correct?/--correct/:/--incorrect/);
    step.cancel();await wait;assert.deepEqual(body.children,[]);
  }
});
