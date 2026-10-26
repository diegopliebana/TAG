// Render only data provided by Rust's permitted observation or explicit reveal.
import { cardImage, artworkCredit } from "./loveletter-art.js";
export { cardImage } from "./loveletter-art.js";
export const cards = ["guard", "priest", "baron", "handmaid", "prince", "king", "countess", "princess"];
export const ranks = [1, 2, 3, 4, 5, 7, 8, 9];
const effects = ["Guess another player's card (not Guard).", "Look at another player's hand.", "Compare hands; the lower rank is eliminated.", "Protection until your next turn.", "A player discards their hand and draws again.", "Exchange hands with another player.", "Must play with Prince or King in hand.", "Discarding Princess eliminates you."];
export const cardTitle = card => card ? `${ranks[cards.indexOf(card)]} ${card[0].toUpperCase()}${card.slice(1)}` : "Hidden card";
const title = cardTitle;
const art = cardImage;
export function renderLetterCard(document, name, label) {
  const figure=document.createElement("figure");figure.className="letter-card";
  const image=document.createElement("img");image.src=art(name);image.alt=title(name);
  const caption=document.createElement("figcaption");caption.textContent=label??title(name);
  figure.append(image,caption);return figure;
}
export function loveLetterActionLabel(action) {
  if (action.guess) return `Play Guard → Player ${action.guess.target}: guess ${title(action.guess.card)}`;
  if (action.target) return `Play ${title(action.target.card)} → Player ${action.target.target}`;
  return `Play ${title(action.play)}`;
}
function description(event) {
  if (event.information) return `After public event ${event.event + 1}: ${description(event.information)}`;
  const [kind, e] = Object.entries(event)[0];
  switch(kind) {
    case "round_started": return `Round ${e.round}: Player ${e.starter} starts.`;
    case "received": return e.card ? `You received ${title(e.card)} (${e.source}).` : `Player ${e.player} received a card (${e.source}).`;
    case "played": return `Player ${e.actor}: ${loveLetterActionLabel(e.action)}.`;
    case "discarded": return `Player ${e.player} discarded ${title(e.card)}.`;
    case "guard_result": return `Player ${e.actor} guessed ${title(e.guess)} for Player ${e.target}: ${e.correct ? "correct" : "incorrect"}.`;
    case "compared": return `Players ${e.actor} and ${e.target} compared hands; ${e.eliminated === null ? "equal ranks" : `Player ${e.eliminated} eliminated`}.`;
    case "exchanged": return `Players ${e.actor} and ${e.target} exchanged hands.`;
    case "protection": return `Player ${e.player}: protection ${e.enabled ? "begins" : "ends"}.`;
    case "eliminated": return `Player ${e.player} was eliminated from this round.`;
    case "round_ended": return `Round winners: ${e.winners.map(p=>`Player ${p}`).join(", ")}.${e.revealed_hands.length ? " Revealed: " + e.revealed_hands.map(([p,c])=>`Player ${p}: ${title(c)}`).join(", ") : ""}`;
    case "game_ended": return `Game winners: ${e.winners.map(p=>`Player ${p}`).join(", ")}.`;
    case "priest": return `You saw Player ${e.target}'s ${title(e.card)}.`;
    case "comparison": return `Your ${title(e.own_card)} compared with Player ${e.opponent}'s ${title(e.other_card)}.`;
    case "exchange": return `Exchanged with Player ${e.other}: gave ${title(e.given)}, received ${title(e.received)}.`;
    default: return kind.replaceAll("_", " ");
  }
}
export class LoveLetterBoard {
  constructor(container, view, sprites) {
    Object.assign(this, { container, game: view.game, width: view.board.width, height: view.board.height, sprites });
  }
  update(view, choose, busy) {
    const document = this.container.ownerDocument, facts = view.board.loveletter;
    const element = (tag, text, className) => {
      const node = document.createElement(tag); if (text !== undefined) node.textContent = text;
      if (className) node.className = className; return node;
    };
    const card = (name,label) => renderLetterCard(document,name,label);
    const root = element("section", undefined, "loveletter-table");
    root.setAttribute("aria-label", "Love Letter table");
    root.append(element("h2", `Round ${facts.round} · ${facts.draw_count} cards in draw pile`));
    root.append(element("p", facts.public_only ? "Public spectator view" : `Player ${view.observer}'s view`));
    if (facts.last_round) {
      const result=facts.last_round.summary;
      const notice=element("aside",undefined,"letter-round-notice");notice.setAttribute("role","status");
      notice.append(element("h3",`Round ${result.round}: ${result.winners.map(p=>`Player ${p}`).join(" and ")} won`));
      notice.append(element("p",result.winners.map(p=>`Player ${p}: +1 favor token (${result.tokens[p]} total)`).join("; ")));
      root.append(notice);
    }
    const seats = element("div", undefined, "letter-seats");
    for (let seat = 0; seat < view.controllers.length; seat++) {
      const section = element("section", undefined, "letter-seat");
      section.append(element("h3", `Player ${seat} · ${facts.tokens[seat]}/${facts.rules.winning_tokens[view.controllers.length - 2]} favor tokens`));
      section.append(element("p", `${facts.active[seat] ? "In round" : "Eliminated"}${facts.protected[seat] ? " · Protected" : ""}${view.next_player === seat ? " · To play" : ""}`));
      const hand = element("div", undefined, "letter-hand");
      let known = facts.privileged?.hands[seat];
      if (!known && !facts.public_only && seat === view.observer) known = facts.hand;
      if (!known && facts.active[seat]) {
        // The number of held cards is public: only the deciding player holds two.
        known = Array(view.next_player === seat ? 2 : 1).fill(null);
        const events = facts.public_events;
        const ended = events.findLast?.(e => e.round_ended)?.round_ended;
        const reveal = ended?.revealed_hands?.find(([player]) => player === seat);
        if (reveal) known = [reveal[1]];
      }
      for (const name of known ?? []) hand.append(card(name));
      section.append(hand);
      section.append(element("p", `Discards: ${facts.discards[seat].map(title).join(", ") || "none"}`));
      seats.append(section);
    }
    root.append(seats);
    if (facts.reserve.length) root.append(element("p", `Face-up reserve: ${facts.reserve.map(title).join(", ")}`));
    if (facts.privileged) {
      const audit = element("aside", undefined, "letter-reveal");
      audit.append(element("h3", "Privileged hidden information"));
      audit.append(element("h4","Removed card"));
      const removed=element("div",undefined,"letter-deck");
      if(facts.privileged.removed) removed.append(card(facts.privileged.removed));
      else removed.append(element("p","None"));
      audit.append(removed,element("h4","Draw pile (top card first)"));
      const pile=element("div",undefined,"letter-deck");pile.setAttribute("aria-label","Draw pile in draw order");
      facts.privileged.draw_pile.forEach((name,index)=>pile.append(card(name,`${index+1}. ${title(name)}`)));
      if(!facts.privileged.draw_pile.length)pile.append(element("p","Empty"));
      audit.append(pile);
      root.append(audit);
    }
    if (view.actions.length) root.append(element("h3", "Actions"));
    const controls = element("div", undefined, "letter-actions");
    // Group choices by played card so target and Guard guesses remain explicit.
    const offered = new Map();
    for (const action of view.actions) {
      const name = action.guess ? "guard" : action.target?.card ?? action.play;
      if (!offered.has(name)) offered.set(name, []); offered.get(name).push(action);
    }
    for (const [name, actions] of offered) {
      const section = element("section", undefined, "letter-choice");
      section.append(card(name)); section.append(element("p", effects[cards.indexOf(name)]));
      if (actions.length === 1) {
        const button = element("button", loveLetterActionLabel(actions[0]));
        button.type = "button"; button.disabled = busy || view.read_only;
        button.onclick = () => { if (!button.disabled) choose(actions[0]); }; section.append(button);
      } else {
        const label = element("label", "Target / guess ");
        const select = element("select"); select.setAttribute("aria-label", `${title(name)} target and guess`);
        actions.forEach((action, index) => {
          const option = element("option", loveLetterActionLabel(action)); option.value = String(index); select.append(option);
        });
        select.value = "0";
        const button = element("button", `Play ${title(name)}`); button.type = "button";
        select.disabled = button.disabled = busy || view.read_only;
        button.onclick = () => { if (!button.disabled) choose(actions[Number(select.value)]); };
        label.append(select); section.append(label, button);
      }
      controls.append(section);
    }
    root.append(controls);
    const history = (name, events) => {
      if (!events?.length) return;
      const details = element("details"); details.append(element("summary", name));
      const list = element("ol"); for (const event of events) list.append(element("li", description(event)));
      details.append(list); root.append(details);
    };
    history("Current public events", facts.public_events);
    history("Your private experiences this round", facts.private_events);
    history("Previous round public events", facts.last_round?.events);
    history("Your previous round private experiences", facts.last_private_events);
    const reference = element("details"); reference.append(element("summary", "Card effects and artwork credits"));
    for (let i = 0; i < cards.length; i++) reference.append(element("p", `${title(cards[i])}: ${effects[i]}`));
    const credit = element("a", artworkCredit.label); credit.href = new URL(artworkCredit.path, import.meta.url).href;
    reference.append(credit); root.append(reference);
    // Rebuild private nodes so switching observer or disabling reveal removes
    // previous secrets from text, image sources, alt text and accessibility data.
    this.container.replaceChildren(root);
  }
}
