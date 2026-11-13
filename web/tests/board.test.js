import test from "node:test";
import assert from "node:assert/strict";
import { renderBoard } from "../board.js";

// Minimal DOM double tracks element identity, focus and event handlers.
class Element {
  constructor(ownerDocument, tag) { this.ownerDocument = ownerDocument; this.tag = tag; this.children = []; this.listeners = {}; this.attributes = {}; this.textContent = ""; }
  append(...elements) { this.children.push(...elements); }
  replaceChildren(...elements) { this.children = elements; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(event, listener) { this.listeners[event] = listener; }
  createTHead() { return this.insert("thead"); }
  createTBody() { return this.insert("tbody"); }
  insertRow() { return this.insert("tr"); }
  insertCell() { return this.insert("td"); }
  insert(tag) { const node = new Element(this.ownerDocument, tag); this.append(node); return node; }
  focus() { this.ownerDocument.activeElement = this; }
  buttons() { return [...(this.tag === "button" ? [this] : []), ...this.children.flatMap(child => child.buttons())]; }
}
function container() {
  const document = { activeElement: null, createElement(tag) { return new Element(this, tag); } };
  return new Element(document, "div");
}
function view(game) {
  return { game, next_player: 0, controllers: [{ kind: "human" }, { kind: "human" }], board: { width: 2, height: 2, cells: [null, null, null, null] }, actions: game === "tictactoe" ? [{ x: 0, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 0 }, { x: 1, y: 1 }] : [{ column: 0 }, { column: 1 }] };
}

test("board updates retain controls, use current actions and preserve available focus", () => {
  const board = container(); const first = view("tictactoe"); const choices = [];
  renderBoard(board, first, action => choices.push(action));
  const table = board.children[0]; const buttons = board.buttons(); buttons[1].focus();
  renderBoard(board, first, action => choices.push(action), { busy: true });
  assert.ok(buttons.every(button => button.disabled));
  const next = view("tictactoe"); next.board.cells[0] = 0; next.next_player = 1; next.actions.shift();
  renderBoard(board, next, action => choices.push(action));
  assert.equal(board.children[0], table); assert.deepEqual(board.buttons(), buttons);
  assert.equal(buttons[0].textContent, "X"); assert.equal(buttons[0].disabled, true);
  assert.equal(board.ownerDocument.activeElement, buttons[1]);
  buttons[1].listeners.click(); assert.equal(choices[0], next.actions.find(action => action.x === 1 && action.y === 0));
  buttons[0].listeners.click(); assert.equal(choices.length, 1);
});

test("Connect 4 columns retain identity and game changes rebuild their structure", () => {
  const board = container(); const first = view("connect4"); const choices = [];
  renderBoard(board, first, action => choices.push(action));
  const buttons = board.buttons(); const next = view("connect4"); next.actions = [{ column: 1 }];
  renderBoard(board, next, action => choices.push(action));
  assert.deepEqual(board.buttons(), buttons); assert.equal(buttons[0].disabled, true);
  buttons[1].listeners.click(); assert.equal(choices[0], next.actions[0]);
  const old = board.children[0]; renderBoard(board, view("tictactoe"), () => {});
  assert.notEqual(board.children[0], old); assert.equal(board.buttons().length, 4);
});

test("Sushi Go! keeps opponent cards face down until privileged reveal", () => {
  const board=container(), action={Pick:"Tempura"}, double={Chopsticks:{first:"Wasabi",second:"SalmonNigiri"}}, picked=[];
  const view={game:"sushigo",observer:0,waiting_seats:[0,1],controllers:[{kind:"human"},{kind:"human"}],
    board:{width:12,height:5,sushigo:{round:1,hand_size:10,draw_count:88,scores:[0,0],pudding:[0,0],played:[[{card:"Sashimi",on_wasabi:false}],[]],round_scores:[],discard:["Dumpling"],last_reveal:[[0,{Pick:"Wasabi"}]],hand:["Tempura"],hand_ids:[0,1],public_only:false}},actions:[action,double]};
  renderBoard(board,view,choice=>picked.push(choice));
  assert.ok(board.children[0].children[4].children.length>0);
  assert.ok(JSON.stringify(board).includes("sashimi.png"));
  assert.ok(JSON.stringify(board).includes("CardBack.png"));
  assert.ok(!JSON.stringify(board).includes("Discarded cards"));
  assert.ok(!JSON.stringify(board).includes("Last reveal"));
  assert.ok(!JSON.stringify(board).includes("Previously seen hand"));
  board.buttons().at(-1).onclick();assert.deepEqual(picked,[double]);
  const nextView={...view,observer:1,waiting_seats:[1],actions:[action],board:{...view.board,sushigo:{...view.board.sushigo,hand:["Dumpling"],known_hands:[["Wasabi"],["Dumpling"]],public_only:false}}};
  renderBoard(board,nextView,choice=>picked.push(choice));
  assert.ok(JSON.stringify(board).includes("Player 1's hand"));
  assert.ok(!JSON.stringify(board).includes("Previously seen hand"));
  assert.ok(!JSON.stringify(board).includes("wasabi.png"));
  assert.ok(!JSON.stringify(board).includes("Remembered passing hand"));
  assert.ok(!JSON.stringify(board).includes("Pass the device"));
  assert.ok(JSON.stringify(board).includes("Sashimi"));
  const privileged={...nextView,board:{...nextView.board,sushigo:{...nextView.board.sushigo,opponent_hands:[["Tempura"],null]}}};
  renderBoard(board,privileged,()=>{});
  assert.ok(JSON.stringify(board).includes("tempura.png"));
  const buttons=board.buttons();assert.equal(buttons.length,1);
  assert.ok(!JSON.stringify(board).includes("Cancel this choice round"));
});

test("The Game displays only the observed hand and offered actions", () => {
  const board=container(), chosen=[];
  const action={Play:{card:12,pile:"AscendingOne"}};
  const view={game:"thegame",observer:0,next_player:0,controllers:[{kind:"human"},{kind:"human"}],
    board:{width:4,height:1,thegame:{pile_tops:[1,1,100,100],placements:[],messages:[],hand_counts:[7,7],draw_count:84,played_this_turn:0,turns_completed:0,active_player:0,phase:"Play",hand:[12,34],public_only:false}},actions:[action]};
  renderBoard(board,view,choice=>chosen.push(choice));
  assert.ok(JSON.stringify(board).includes("Player 0's hand"));
  assert.ok(!JSON.stringify(board).includes("opponent_hands"));
  assert.equal(board.buttons().length,2);
  board.buttons()[0].onclick();
  assert.equal(board.buttons().length,3);
  board.buttons().at(-1).onclick();assert.deepEqual(chosen,[action]);
  const spectator={...view,actions:[],board:{...view.board,thegame:{...view.board.thegame,hand:[],public_only:true}}};
  renderBoard(board,spectator,()=>{});
  assert.ok(!JSON.stringify(board).includes("Player 0's hand"));
  assert.equal(board.buttons().length,0);
  const revealed={...spectator,board:{...spectator.board,thegame:{...spectator.board.thegame,opponent_hands:[[12,34],[56,78]]}}};
  renderBoard(board,revealed,()=>{});
  assert.ok(JSON.stringify(board).includes("All hands (privileged reveal)"));
  assert.ok(JSON.stringify(board).includes("Player 0"));
  assert.ok(JSON.stringify(board).includes("Player 1"));
  for(const card of [12,34,56,78]) assert.ok(JSON.stringify(board).includes(String(card)));
  assert.equal(board.buttons().length,0);
  const stale={...revealed,board:{...revealed.board,thegame:{...revealed.board.thegame,opponent_hands:[null,[56,78]]}}};
  renderBoard(board,stale,()=>{});
  assert.ok(JSON.stringify(board).includes("Hand unavailable. Save the session, then reload the page to update the game engine."));
  assert.ok(!JSON.stringify(board).includes("No cards"));
});

test("The Game keeps every hand card visible while showing one card's piles",()=>{
  const board=container(),chosen=[];
  const hand=[58,21,41,4,10,2,91];
  const actions=hand.flatMap(card=>["AscendingOne","AscendingTwo","DescendingOne","DescendingTwo"].map(pile=>({Play:{card,pile}})));
  const action_paths=hand.flatMap(card=>[0,1,2,3].map(pile=>[4000+card,pile]));
  const view={game:"thegame",observer:0,next_player:0,controllers:[{kind:"human"}],actions,
    board:{width:4,height:1,thegame:{pile_tops:[1,1,100,100],placements:[],messages:[],hand_counts:[7],draw_count:91,played_this_turn:0,turns_completed:0,active_player:0,phase:"Play",hand,action_paths,public_only:false}}};
  renderBoard(board,view,choice=>chosen.push(choice));
  assert.equal(board.buttons().length,7);
  board.buttons()[5].onclick();
  assert.equal(board.buttons().length,11);
  assert.equal(board.buttons()[5].attributes["aria-pressed"],"true");
  board.buttons().at(-1).onclick();assert.deepEqual(chosen,[actions[23]]);
});

test("candidate artwork updates existing layers and keeps empty pieces hidden", () => {
  const board = container(); const first = view("tictactoe");
  const sprites = { x: "x.svg", o: "o.png", tttCell: "cell.svg", c4Cell: "blue.svg", socket: "socket.svg", red: "red.svg", yellow: "yellow.svg" };
  renderBoard(board, first, () => {}, { busy: true, sprites });
  const button = board.buttons()[0]; const piece = button.children[1];
  assert.equal(piece.hidden, true); assert.equal(button.children[0].src, "cell.svg");
  first.board.cells[0] = 0;
  renderBoard(board, first, () => {}, { busy: true, sprites, emphasis: [[0, 0]] });
  assert.equal(board.buttons()[0], button); assert.equal(piece.hidden, false); assert.equal(piece.src, "x.svg");
  assert.equal(button.attributes["data-winning"], true);
  first.board.cells[0] = null;
  renderBoard(board, first, () => {}, { busy: true, sprites });
  assert.equal(piece.hidden, true); assert.equal(button.attributes["data-winning"], false);
});
