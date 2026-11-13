import {createGame} from "./human-play.js";
const ui=id=>document.getElementById(id);
async function until(test,message,limit=400) {
  for(let i=0;i<limit;i++){if(test())return;await new Promise(resolve=>setTimeout(resolve,25));}
  throw new Error(`${message}: ${ui("status").textContent}`);
}
export async function runTheGameChecks() {
  await createGame("thegame",{players:1,seed:"42",first_player:0},[{kind:"human"}]);
  if(!ui("board").querySelector(".thegame-piles")||!ui("board").textContent.includes("cards in the deck")) throw new Error("The Game table did not render");
  let clicks=0;
  while(!["The team wins","The team loses"].includes(ui("result").textContent)) {
    if(!ui("board").querySelector(".thegame-actions button:enabled")) ui("board").querySelector(".thegame-cards button:enabled")?.click();
    await until(()=>Boolean(ui("board").querySelector(".thegame-actions button:enabled")),"No solo action");
    ui("board").querySelector(".thegame-actions button:enabled").click();
    await until(()=>!ui("status").textContent.includes("Applying your request"),"Solo move did not complete");
    if(++clicks>400)throw new Error("Solo game did not finish");
  }
  const soloResult=ui("result").textContent;
  await createGame("thegame",{players:5,seed:"42",first_player:0},Array.from({length:5},(_,seat)=>({kind:"random",seed:String(seat+7)})),false);
  ui("ai-delay").value="0";ui("ai-delay").dispatchEvent(new Event("change"));
  ui("play-pause").click();
  await until(()=>["The team wins","The team loses"].includes(ui("result").textContent),"Five-seat game did not finish",2000);
  return {soloClicks:clicks,soloResult,fiveSeatResult:ui("result").textContent};
}
