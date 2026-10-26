// Run inside the real app with its module worker, setup and card controls.
import { createGame } from "./human-play.js";
const ui=id=>document.getElementById(id);
const check=(ok,message)=>{if(!ok)throw new Error(message);};
async function until(predicate,message,dismiss=true) {
  for(let i=0;i<800;i++){
    if(dismiss && ui("action-result").open)ui("action-result-continue").click();
    if(predicate())return;await new Promise(r=>setTimeout(r,25));
  }
  throw new Error(`${message}: ${ui("status").textContent}`);
}
export async function runLoveLetterChecks() {
  await window.gameReady;
  check(!ui("always-action-results").checked,"Always-show results should default to off");
  ui("mute").checked=true;ui("mute").dispatchEvent(new Event("change"));
  ui("reduced").checked=true;ui("reduced").dispatchEvent(new Event("change"));
  const counts=[];
  for(const n of [2,3,4]) {
    await createGame("loveletter",{seed:"42",rules:{winning_tokens:[1,1,1]}},Array.from({length:n},()=>({kind:"human"})));
    check(ui("players").children.length===n,"Seat count differs from setup");
    check(ui("board").querySelectorAll(".letter-seat").length===n,"Card seats missing");
    check(ui("board").textContent.includes("Player 0's view"),"Initial observer differs");
    const table=ui("board").querySelector(".loveletter-table");
    const frame=()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    ui("board-display").value="scrollable";ui("board-display").dispatchEvent(new Event("change"));
    ui("board-zoom").value="100";ui("board-zoom").dispatchEvent(new Event("input"));await frame();
    const originalWidth=table.getBoundingClientRect().width;
    ui("board-zoom").value="150";ui("board-zoom").dispatchEvent(new Event("input"));await frame();
    check(Math.abs(table.getBoundingClientRect().width/originalWidth-1.5)<.02,"Card table zoom did not resize");
    ui("board-zoom-reset").click();await frame();
    check(Math.abs(table.getBoundingClientRect().width-originalWidth)<2,"Zoom reset did not restore size");
    ui("board-display").value="fit";ui("board-display").dispatchEvent(new Event("change"));await frame();
    check(table.getBoundingClientRect().width<=ui("board").clientWidth,"Fit exceeds the viewport width");
    const art=[...ui("board").querySelectorAll("img")];await Promise.all(art.map(image=>image.decode()));
    check(art.every(image=>image.naturalWidth>0),"Artwork did not load");
    ui("reveal-hidden").checked=true;ui("reveal-hidden").dispatchEvent(new Event("change"));
    await until(()=>ui("board").querySelector(".letter-reveal"),"Reveal did not appear");
    ui("reveal-hidden").checked=false;ui("reveal-hidden").dispatchEvent(new Event("change"));
    check(!ui("board").querySelector(".letter-reveal"),"Privileged DOM survived disabling reveal");
    await until(()=>ui("board").querySelector("button:enabled"),"Private view did not return");
    const selected=ui("board").querySelector("button:enabled");selected.click();
    await until(()=>ui("action-result").open,"Action result did not appear",false);
    check(ui("action-result-body").textContent.length>20,"Action result is empty");
    check(ui("action-result-body").querySelector(".letter-action-overview img"),"Played card image is missing");
    check(!ui("board").querySelector("button:enabled"),"Play was enabled during the result step");
    ui("action-result-continue").click();
    await until(()=>ui("details").textContent.includes("| 1 moves |")&&!ui("reset").disabled,"Human move did not complete");
    ui("previous").click();
    await until(()=>ui("status").textContent.includes("Reviewing move 0"),"History did not open");
    check(!ui("board").querySelector("button:enabled"),"Historical actions remain active");
    check(ui("board").textContent.includes("Player 0's view"),"History used a later observer");
    ui("branch-end").click();await until(()=>!ui("reset").disabled&&!ui("status").textContent.includes("Reviewing")&&!ui("status").textContent.includes("Loading"),"Branch end did not return");
    counts.push(n);
  }
  await createGame("loveletter",{seed:"42",rules:{winning_tokens:[1,1,1]}},[
    {kind:"human"},{kind:"sampled_osla",seed:"8",repetitions_per_action:1},{kind:"human"}]);
  ui("ai-delay").value="0";ui("ai-delay").dispatchEvent(new Event("change"));
  let steps=0;
  while(!/wins|share the win/.test(ui("result").textContent)&&steps++<100) {
    const before=ui("details").textContent;
    const button=ui("board").querySelector("button:enabled");
    if(button)button.click();else if(!ui("ai-step").disabled)ui("ai-step").click();
    else {await new Promise(r=>setTimeout(r,25));continue;}
    await until(()=>ui("details").textContent!==before&&!ui("reset").disabled,"Human/AI move did not settle");
  }
  check(/wins|share the win/.test(ui("result").textContent),"Human/AI game did not finish");
  await createGame("loveletter",{seed:"42"},[
    {kind:"first_available",seed:"7"},{kind:"first_available",seed:"8"}]);
  ui("ai-step").click();
  await until(()=>ui("details").textContent.includes("| 1 moves |")&&!ui("ai-step").disabled,"AI-only play did not skip results",false);
  check(!ui("action-result").open,"AI-only game showed a default result screen");
  ui("always-action-results").checked=true;
  ui("ai-step").click();
  await until(()=>ui("action-result").open,"Always-show override did not display a result",false);
  ui("action-result-continue").click();
  await until(()=>ui("details").textContent.includes("| 2 moves |")&&!ui("ai-step").disabled,"Forced result did not continue");
  ui("always-action-results").checked=false;
  // Leave an ordinary opening position visible for manual visual review.
  await createGame("loveletter",{seed:"42"},[{kind:"human"},{kind:"human"},{kind:"human"}]);
  return {passed:true,playerCounts:counts,humanAiSteps:steps,reveal:true,history:true,artwork:true};
}
