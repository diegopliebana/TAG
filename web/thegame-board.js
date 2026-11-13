// The four pile controls offer only moves supplied by the Rust forward model.
const names = ["AscendingOne", "AscendingTwo", "DescendingOne", "DescendingTwo"];
const pileName = pile => ({AscendingOne:"Ascending 1",AscendingTwo:"Ascending 2",DescendingOne:"Descending 1",DescendingTwo:"Descending 2"})[pile] ?? pile;
export function theGameActionLabel(action) {
  if (action === "EndTurn") return "End turn and draw";
  if (action.Play) return `Play ${action.Play.card} on ${pileName(action.Play.pile)}`;
  const message=action.Speak;
  if (message === "NoRequest") return "No request";
  const [kind,pile]=Object.entries(message ?? {})[0] ?? [];
  return `${({LeavePile:"Please leave",SmallJump:"I can make a small jump on",CanUsePile:"I can use"})[kind] ?? kind} ${pileName(pile)}`;
}
export class TheGameBoard {
  constructor(container, view, sprites) {
    Object.assign(this,{container,game:view.game,width:view.board.width,height:view.board.height,sprites});
    this.selectedCard=null;
    this.observer=null;
    this.root=container.ownerDocument.createElement("section");this.root.className="thegame-board";
    container.replaceChildren(this.root);
  }
  update(view,choose,busy) {
    const d=this.container.ownerDocument,data=view.board.thegame;
    const legalPlays=view.actions.filter(action=>action.Play);
    const actionPaths=data.action_paths;
    const hasPaths=Array.isArray(actionPaths)&&actionPaths.length===view.actions.length;
    const pathFor=(action)=>hasPaths?actionPaths[view.actions.indexOf(action)]:undefined;
    const cardKey=card=>{
      const index=view.actions.findIndex(action=>action.Play?.card===card);
      return index<0?undefined:actionPaths[index]?.[0];
    };
    const cardAvailable=card=>hasPaths
      ? cardKey(card)!==undefined
      : legalPlays.some(action=>action.Play.card===card);
    if(this.observer!==view.observer || !data.hand.includes(this.selectedCard) || !cardAvailable(this.selectedCard)) this.selectedCard=null;
    this.observer=view.observer;
    const summary=d.createElement("p");summary.className="thegame-summary";
    summary.textContent=`${data.draw_count} cards in the deck · ${98-data.placements.length} cards to place · Turn ${data.turns_completed+1}`;
    const piles=d.createElement("div");piles.className="thegame-piles";
    names.forEach((name,index)=>{
      const section=d.createElement("section");section.className="thegame-pile";
      const heading=d.createElement("h3");heading.textContent=pileName(name);
      const top=d.createElement("strong");top.textContent=data.pile_tops[index];
      const hint=d.createElement("p");hint.textContent=index<2?"Place a higher card, or exactly 10 lower":"Place a lower card, or exactly 10 higher";
      section.append(heading,top,hint);piles.append(section);
    });
    const team=d.createElement("div");team.className="thegame-team";
    data.hand_counts.forEach((count,seat)=>{
      const item=d.createElement("span");item.textContent=`Player ${seat}: ${count} cards${seat===data.active_player?" · active":""}`;team.append(item);
    });
    const messages=d.createElement("section");messages.className="thegame-messages";
    const title=d.createElement("h3");title.textContent="Team messages";messages.append(title);
    const last=data.messages.at(-1);
    const line=d.createElement("p");line.textContent=last?last.messages.map((message,seat)=>`Player ${seat}: ${theGameActionLabel({Speak:message})}`).join(" · "):"No messages revealed yet";messages.append(line);
    const hand=d.createElement("section");hand.className="thegame-hand";
    const handTitle=d.createElement("h3");handTitle.textContent=data.public_only?"Hands are hidden":`Player ${view.observer}'s hand`;
    hand.append(handTitle);
    if (!data.public_only) {
      const cards=d.createElement("div");cards.className="thegame-cards";
      for(const card of data.hand){
        const tile=d.createElement("button");tile.type="button";tile.className="thegame-card";tile.textContent=card;
        const available=cardAvailable(card);
        tile.disabled=busy || !available || view.next_player!==view.observer || view.controllers[view.observer].kind!=="human";
        tile.setAttribute("aria-pressed",this.selectedCard===card?"true":"false");
        tile.setAttribute("aria-label",`${card}: ${available?"choose a pile":"no legal pile"}`);
        tile.onclick=()=>{if(!tile.disabled){this.selectedCard=this.selectedCard===card?null:card;this.update(view,choose,busy);}};
        cards.append(tile);
      }hand.append(cards);
    }
    if(data.opponent_hands){
      const revealed=d.createElement("section");revealed.className="thegame-revealed-hands";
      const title=d.createElement("h3");title.textContent="All hands (privileged reveal)";revealed.append(title);
      data.opponent_hands.forEach((cards,seat)=>{
        const row=d.createElement("section");row.className="thegame-revealed-seat";
        const heading=d.createElement("h4");heading.textContent=`Player ${seat}`;row.append(heading);
        const tiles=d.createElement("div");tiles.className="thegame-cards";
        if(cards===null){const notice=d.createElement("p");notice.textContent="Hand unavailable. Save the session, then reload the page to update the game engine.";tiles.append(notice);}
        else {
          for(const card of cards){const tile=d.createElement("span");tile.className="thegame-card";tile.textContent=card;tiles.append(tile);}
          if(!tiles.children.length){const empty=d.createElement("p");empty.textContent="No cards";tiles.append(empty);}
        }
        row.append(tiles);revealed.append(row);
      });
      hand.append(revealed);
    }
    const choices=d.createElement("section");choices.className="thegame-actions";
    choices.setAttribute("role","group");choices.setAttribute("aria-label","Available The Game actions");
    const canAct=!busy && view.next_player===view.observer && view.controllers[view.observer].kind==="human";
    const offered=view.actions.filter(action=>{
      const path=pathFor(action);
      return path ? path.length===1 || path[0]===cardKey(this.selectedCard) : !action.Play || action.Play.card===this.selectedCard;
    });
    if(legalPlays.length && this.selectedCard===null){
      const hint=d.createElement("p");hint.textContent="Select a card to see its legal piles.";choices.append(hint);
    }
    for(const action of offered){
      const button=d.createElement("button");button.type="button";button.textContent=theGameActionLabel(action);
      button.disabled=!canAct;button.onclick=()=>{if(!button.disabled)choose(action);};choices.append(button);
    }
    this.root.replaceChildren(summary,piles,team,messages,hand,choices);
  }
}
