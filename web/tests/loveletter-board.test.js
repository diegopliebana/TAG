import test from "node:test";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";
import { renderBoard } from "../board.js";
import { loveLetterActionLabel } from "../loveletter-board.js";
import { readParameters, setupFields } from "../setup.js";
class Element {
  constructor(document,tag) {Object.assign(this,{ownerDocument:document,tag,children:[],attributes:{}});}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  setAttribute(name,value){this.attributes[name]=value;}
}
const all=node=>[node,...node.children.flatMap(all)];
test("Love Letter cards, controls, alt text and private DOM are cleared when reveal is disabled",()=>{
  const document={createElement:tag=>new Element(document,tag)};
  const container=document.createElement("div");
  const view={game:"loveletter",observer:0,next_player:0,controllers:[{kind:"human"},{kind:"human"}],
    actions:[{guess:{target:1,card:"princess"}},{play:"handmaid"}],board:{width:16,height:1,loveletter:{
      round:1,tokens:[0,0],rules:{winning_tokens:[6,5,4]},active:[true,true],protected:[false,false],
      draw_count:10,hand:["guard","handmaid"],reserve:[],discards:[[],[]],public_events:[],private_events:[],
      last_private_events:[],privileged:{hands:[["guard","handmaid"],["princess"]],removed:"king",draw_pile:["countess"]}}}};
  let chosen;
  renderBoard(container,view,a=>chosen=a);
  assert.ok(all(container).some(e=>e.alt==="9 Princess"));
  const decks=all(container).filter(e=>e.className==="letter-deck");
  assert.equal(decks.length,2);
  assert.equal(decks[0].children[0].children[0].alt,"7 King");
  assert.equal(decks[1].children[0].children[0].alt,"8 Countess");
  assert.match(decks[1].children[0].children[1].textContent,/^1\./);

  const button=all(container).find(e=>e.tag==="button"&&e.textContent.includes("guess"));
  button.onclick();assert.deepEqual(chosen,view.actions[0]);
  delete view.board.loveletter.privileged;
  renderBoard(container,view,()=>{});
  assert.ok(!all(container).some(e=>e.alt==="9 Princess"||e.alt==="7 King"||e.alt==="8 Countess"));
  assert.ok(all(container).some(e=>e.alt==="Hidden card"));
  view.actions=[];view.board.loveletter.hand=[];view.board.loveletter.public_only=true;
  renderBoard(container,view,()=>{});
  assert.ok(!all(container).some(e=>e.tag==="button"));
  assert.ok(!all(container).some(e=>e.alt==="1 Guard"));
});
test("Love Letter setup and action labels follow classic ranks and separate target from Guard guess",()=>{
  const fields=setupFields("loveletter").map(f=>({name:f.name,value:String(f.value)}));
  fields.find(f=>f.name==="seed").value="18446744073709551615";
  assert.deepEqual(readParameters("loveletter",fields),{rules:{winning_tokens:[6,5,4]},first_player:0,seed:"18446744073709551615"});
  assert.match(loveLetterActionLabel({target:{card:"king",target:2}}),/7 King.*Player 2/);
  fields.find(f=>f.name==="first_player").value="3";
  assert.throws(()=>readParameters("loveletter",fields),/valid seat/);
});


test("Love Letter artwork files and attribution retain their recorded hashes",()=>{
  const base=new URL("../assets/games/loveletter/",import.meta.url);
  const origin=JSON.parse(readFileSync(new URL("origin.json",base),"utf8"));
  assert.equal(Object.keys(origin.cards).length,9);
  for(const [name,hash] of Object.entries(origin.sha256)) {
    assert.equal(createHash("sha256").update(readFileSync(new URL(name,base))).digest("hex"),hash);
  }
});
