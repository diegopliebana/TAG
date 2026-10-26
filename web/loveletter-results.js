// Explain an accepted transition using event-time public facts and one observer's
// private experiences. Never infer an earlier hand from the new live hand/deck.
import { cardTitle, renderLetterCard } from "./loveletter-board.js";

export function shouldShowActionResult(before, after, always = false) {
  const current = after?.board?.loveletter, previous = before?.board?.loveletter;
  if (!current) return false;
  if (always) return true;
  let active = current.active;
  if (previous && previous.round !== current.round) {
    // The next deal already exists in Rust. Check the completed action's round,
    // rather than treating newly reactivated humans as participants in that action.
    if (current.last_round?.summary.round !== previous.round) return false;
    active = [...previous.active];
    for (const event of current.last_round.events.slice(previous.public_events.length)) {
      if (event.eliminated) active[event.eliminated.player] = false;
    }
  }
  return after.controllers.some((seat, player) => seat.kind === "human" && active?.[player]);
}

export function actionResult(before, after) {
  const previous=before?.board?.loveletter, current=after?.board?.loveletter;
  if(!previous || !current || before.branch!==after.branch || after.cursor!==before.cursor+1) return null;
  const sameRound=previous.round===current.round;
  const transcript=sameRound?current.public_events:current.last_round?.events;
  if(!transcript || (!sameRound && current.last_round?.summary.round!==previous.round))return null;
  const start=previous.public_events.length, events=transcript.slice(start);
  const privateEvents=current.public_only?[]:(sameRound?current.private_events:current.last_private_events)??[];
  const experiences=privateEvents.filter(e=>e.event>=start).map(e=>e.information);
  const played=events.find(e=>e.played)?.played;
  if(!played)return null;
  const name=played.action.guess?"guard":played.action.target?.card??played.action.play;
  const sections=[];
  const add=(title,text,cards=[])=>sections.push({title,text,cards});
  if(name==="priest" && played.action.target) {
    const e=experiences.find(p=>p.priest)?.priest;
    add("Priest: look at a hand",e?`Player ${e.target} showed this card to Player ${after.observer}.`:
      `Player ${played.actor} looked at Player ${played.action.target?.target}'s hand privately.`,
      e?[{card:e.card,label:`Player ${e.target}'s card at the time of the effect`}]:[]);
  }
  if(name==="prince" && played.action.target) {
    const target=played.action.target?.target;
    const discard=events.filter(e=>e.discarded).slice(1).find(e=>e.discarded.player===target)?.discarded;
    const received=target===after.observer?experiences.find(p=>p.received)?.received:null;
    add("Prince: discard and replace",`Player ${target} discarded ${discard?cardTitle(discard.card):"their card"}.`+
      (discard?.card==="princess"?" Discarding Princess eliminates them.":" They draw a replacement card."),[
        ...(discard?[{card:discard.card,label:`Player ${target}'s discarded hand`}]:[]),
        ...(received?[{card:received.card,label:`Player ${target}'s replacement card`}]:[]),
      ]);
  }
  events.forEach((event,index)=>{
    if(event.guard_result) {
      const e=event.guard_result;
      add(e.correct?"Correct guess!":"Incorrect guess",
        `Player ${e.actor} guessed ${cardTitle(e.guess)} for Player ${e.target}. `+
        (e.correct?`Player ${e.target} is eliminated from this round.`:"The guess was incorrect."),
        [{card:e.guess,label:`Guess for Player ${e.target}`,status:e.correct?"correct":"incorrect"}]);
    }
    if(event.compared) {
      const e=event.compared;
      const comparison=experiences.find(p=>p.comparison &&
        [e.actor,e.target].includes(after.observer) &&
        p.comparison.opponent===(after.observer===e.actor?e.target:e.actor))?.comparison;
      const known=p=>comparison?(p===after.observer?comparison.own_card:comparison.other_card):
        events.slice(index+1).find(ev=>ev.discarded?.player===p)?.discarded.card??null;
      add("Baron comparison",e.eliminated===null?"Equal ranks: neither player is eliminated.":`Player ${e.eliminated} has the lower card and is eliminated.`,
        [e.actor,e.target].map(p=>({card:known(p),label:`Player ${p}'s compared hand`,
          status:e.eliminated===null?"equal":e.eliminated===p?"eliminated":"winner"})));
    }
    if(event.exchanged) {
      const e=event.exchanged;
      const own=experiences.find(p=>p.exchange)?.exchange;
      add("King: hands exchanged",`Players ${e.actor} and ${e.target} exchanged hands.`,own?[
        {card:own.given,label:`Player ${after.observer} gave this card`},
        {card:own.received,label:`Player ${after.observer} received this card`},
      ]:[]);
    }
    if(event.eliminated && !events.some(e=>e.compared || e.guard_result?.correct))
      add("Eliminated",`Player ${event.eliminated.player} is out of this round.`);
    if(event.protection?.enabled) add("Handmaid protection",`Player ${event.protection.player} is protected until their next turn.`);
    if(event.round_ended) {
      const e=event.round_ended;
      const tokens=current.last_round?.summary.tokens??current.tokens;
      add(`Round ${previous.round} won by ${e.winners.map(p=>`Player ${p}`).join(" and ")}`,
        (e.reason==="deck_empty"?"The draw pile is empty; the highest surviving hands win. ":"The last remaining player wins. ")+
        e.winners.map(p=>`Player ${p}: +1 favor token (${tokens[p]} total).`).join(" "),
        e.revealed_hands.map(([p,c])=>({card:c,label:`Player ${p}'s round-end hand`})));
    }
    if(event.game_ended) add("Game won",`${event.game_ended.winners.map(p=>`Player ${p}`).join(" and ")} ${event.game_ended.winners.length>1?"share the win":"wins the game"}.`);
  });
  if(!sections.length)add("Card played",`Player ${played.actor} discarded ${cardTitle(name)}. `+
    (["guard","priest","baron","prince","king"].includes(name)?"No eligible target is available, so the card has no effect.":"No further card effect applies."));
  if(!sameRound) {
    const starter=current.public_events.find(e=>e.round_started)?.round_started.starter;
    if(starter!==undefined)add(`Round ${current.round} begins`,`Player ${starter} starts the next round.`);
  }
  return {title:`Player ${played.actor} played ${cardTitle(name)}`,observer:current.public_only?null:after.observer,
    played:{card:name,actor:played.actor,target:played.action.guess?.target??played.action.target?.target??null},sections};
}

function informationCard(document,entry) {
  const figure=renderLetterCard(document,entry.card,`${entry.label}: ${cardTitle(entry.card)}`);
  const tick=String.fromCodePoint(0x2713);
  const markers={correct:[tick,"Correct guess"],incorrect:["X","Incorrect guess"],
    winner:[tick,"Higher hand"],eliminated:["X","Eliminated"],equal:["=","Equal ranks"]};
  const marker=markers[entry.status];
  if(marker) {
    const [image,caption]=figure.children;
    const frame=document.createElement("div");frame.className="letter-information-image";
    const mark=document.createElement("span");mark.className=`letter-card-mark letter-card-mark--${entry.status}`;
    mark.textContent=marker[0];mark.setAttribute("role","img");mark.setAttribute("aria-label",marker[1]);
    frame.append(image,mark);figure.replaceChildren(frame,caption);
  }
  return figure;
}

export class ActionResultStep {
  constructor(dialog,title,body) {
    Object.assign(this,{dialog,title,body,resolve:null});
    dialog.addEventListener("close",()=>{if(!dialog.open)this.finish(true);});
  }
  show(result) {
    this.cancel();
    this.title.textContent=result.title;
    const document=this.body.ownerDocument;
    const context=document.createElement("p");context.textContent=result.observer===null?"Public action information":`Player ${result.observer}'s view`;
    this.body.append(context);
    if(result.played) {
      const overview=document.createElement("div");overview.className="letter-action-overview";
      overview.append(renderLetterCard(document,result.played.card,`Played by Player ${result.played.actor}: ${cardTitle(result.played.card)}`));
      if(result.played.target!==null && result.played.target!==undefined) {
        const target=document.createElement("p");target.className="letter-action-target";
        target.textContent=`${String.fromCodePoint(0x2192)} Player ${result.played.target}`;overview.append(target);
      }
      this.body.append(overview);
    }
    for(const section of result.sections) {
      const article=document.createElement("section");
      const title=document.createElement("h3");title.textContent=section.title;
      const text=document.createElement("p");text.textContent=section.text;
      article.append(title,text);
      const cards=document.createElement("div");cards.className="letter-hand";
      for(const entry of section.cards) cards.append(informationCard(document,entry));
      article.append(cards);this.body.append(article);
    }
    const finished=new Promise(resolve=>this.resolve=resolve);
    this.dialog.returnValue="";this.dialog.showModal();return finished;
  }
  finish(acknowledged) {
    const resolve=this.resolve;this.resolve=null;
    this.title.textContent="";this.body.replaceChildren();
    resolve?.(acknowledged);
  }
  cancel() {this.finish(false);if(this.dialog.open)this.dialog.close();}
}
