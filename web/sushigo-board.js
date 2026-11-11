// Sushi Go! cards and legal choices come from one player's Rust observation.
export const cardNames = {
  Tempura: "Tempura", Sashimi: "Sashimi", Dumpling: "Dumpling",
  Maki1: "Maki (1 roll)", Maki2: "Maki (2 rolls)", Maki3: "Maki (3 rolls)",
  SquidNigiri: "Squid Nigiri", SalmonNigiri: "Salmon Nigiri", EggNigiri: "Egg Nigiri",
  Wasabi: "Wasabi", Chopsticks: "Chopsticks", Pudding: "Pudding",
};
const cardImages = {
  Tempura: "tempura.png", Sashimi: "sashimi.png", Dumpling: "dumpling.png",
  Maki1: "maki_1.png", Maki2: "maki_2.png", Maki3: "maki_3.png",
  SquidNigiri: "squidnigiri.png", SalmonNigiri: "salmonnigiri.png", EggNigiri: "eggnigiri.png",
  Wasabi: "wasabi.png", Chopsticks: "chopsticks.png", Pudding: "pudding.png",
};
function cardView(document, card, onWasabi=false) {
  const figure=document.createElement("figure");figure.className="sushigo-card";
  const image=document.createElement("img");image.src=`./assets/sushigo/${cardImages[card]}`;
  image.alt="";image.loading="lazy";
  const caption=document.createElement("figcaption");caption.textContent=`${cardNames[card]??card}${onWasabi?" on Wasabi":""}`;
  figure.append(image,caption);return figure;
}
function cardBack(document) {
  const figure=document.createElement("figure");figure.className="sushigo-card sushigo-card-back";
  const image=document.createElement("img");image.src="./assets/sushigo/CardBack.png";image.alt="Face-down card";
  figure.append(image);return figure;
}
export const sushiActionLabel = action => action.Pick
  ? `Play ${cardNames[action.Pick] ?? action.Pick}`
  : action.Chopsticks
    ? `Use Chopsticks: ${cardNames[action.Chopsticks.first]} then ${cardNames[action.Chopsticks.second]}`
    : "Unknown choice";

export class SushiGoBoard {
  constructor(container, view, sprites) {
    Object.assign(this, {container, game:view.game, width:view.board.width, height:view.board.height, sprites});
    const d=container.ownerDocument;
    this.root=d.createElement("section");this.root.className="sushigo-board";
    this.summary=d.createElement("p");this.summary.className="sushigo-summary";
    this.public=d.createElement("div");this.public.className="sushigo-public";
    this.opponents=d.createElement("div");this.opponents.className="sushigo-opponents";
    this.history=d.createElement("div");this.history.className="sushigo-history";
    this.hand=d.createElement("div");this.hand.className="sushigo-hand";
    this.actions=d.createElement("div");this.actions.className="sushigo-actions";
    this.actions.setAttribute("role","group");this.actions.setAttribute("aria-label","Available Sushi Go! choices");
    this.root.append(this.summary,this.public,this.opponents,this.history,this.hand,this.actions);container.replaceChildren(this.root);
  }
  update(view, choose, busy) {
    const data=view.board.sushigo, d=this.container.ownerDocument;
    const waiting=view.waiting_seats ?? [];
    this.summary.textContent=`Round ${data.round} of 3 · ${data.hand_size} cards per hand · ${data.draw_count} cards in draw pile`;
    const publicRows=[];
    for (let seat=0;seat<view.controllers.length;seat++) {
      const row=d.createElement("section");row.className="sushigo-seat";
      const title=d.createElement("h3");title.textContent=`Player ${seat} · ${data.scores[seat]} points · ${data.pudding[seat]} Pudding`;
      const cards=d.createElement("div");cards.className="sushigo-cards sushigo-played";
      for(const entry of data.played[seat]??[]) cards.append(cardView(d,entry.card,entry.on_wasabi));
      if(!cards.children.length) {const empty=d.createElement("p");empty.textContent="No cards played this round";cards.append(empty);}
      row.append(title,cards);publicRows.push(row);
    }
    this.public.replaceChildren(...publicRows);
    const opponents=[];
    for(let seat=0;seat<view.controllers.length;seat++) {
      if(seat===view.observer) continue;
      const hand=data.opponent_hands?.[seat];
      const section=d.createElement("section");section.className="sushigo-opponent";
      const heading=d.createElement("h3");heading.textContent=`Player ${seat}'s hand${hand?" · privileged reveal":" · hidden"}`;
      const cards=d.createElement("div");cards.className="sushigo-cards";
      if(hand) for(const card of hand) cards.append(cardView(d,card));
      else for(let index=0;index<data.hand_size;index++) cards.append(cardBack(d));
      section.append(heading,cards);opponents.push(section);
    }
    this.opponents.replaceChildren(...opponents);
    const history=[];
    const rounds=d.createElement("p");
    rounds.textContent=data.round_scores.length ? `Previous round points: ${data.round_scores.map((scores,index)=>`Round ${index+1}: ${scores.join(" / ")}`).join("; ")}` : "No round has ended yet";
    history.push(rounds);
    this.history.replaceChildren(...history);
    this.hand.replaceChildren();this.actions.replaceChildren();
    if (!data.public_only) {
      const title=d.createElement("h3");title.textContent=`Player ${view.observer}'s hand`;
      this.hand.append(title);
      const cards=d.createElement("div");cards.className="sushigo-cards";
      for (const card of data.hand) cards.append(cardView(d,card));
      this.hand.append(cards);
    }
    const enabled=!busy && waiting.includes(view.observer) && view.controllers[view.observer].kind==="human";
    for (const action of view.actions) {
      const button=d.createElement("button");button.type="button";button.className="sushigo-choice";
      const images=d.createElement("span");images.className="sushigo-choice-images";
      for(const card of action.Pick?[action.Pick]:action.Chopsticks?[action.Chopsticks.first,action.Chopsticks.second]:[]) {
        const image=d.createElement("img");image.src=`./assets/sushigo/${cardImages[card]}`;image.alt="";images.append(image);
      }
      const label=d.createElement("span");label.textContent=sushiActionLabel(action);
      button.append(images,label);button.disabled=!enabled;
      button.onclick=()=>{if(!button.disabled)choose(action);};this.actions.append(button);
    }
  }
}
