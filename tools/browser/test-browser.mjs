// Dependency-free real Chrome/Edge proof using Node HTTP/WebSocket and DevTools.
import { createServer } from "node:http";
import { readFile, mkdtemp, readdir } from "node:fs/promises";
import { existsSync, mkdirSync } from "node:fs";
import { resolve, sep, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";
const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const native = spawnSync("cargo", ["run", "--offline", "-p", "tag-browser", "--example", "native_workload"], { cwd: root, encoding: "utf8", windowsHide: true });
if (native.status !== 0) throw new Error(native.stderr || "Native comparison generation failed.");
console.log("Native comparison workload recorded.");
const candidate = process.env.TAG_BROWSER_EXE || ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find(existsSync);
if (!candidate) throw new Error("Set TAG_BROWSER_EXE to an installed Chrome/Edge executable.");
const mime = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".wasm": "application/wasm", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".ogg": "audio/ogg" };
const server = createServer(async (request, response) => {
  try {
    let pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    if (pathname.endsWith("/")) pathname += "index.html";
    if (!pathname.startsWith("/web/") && !pathname.startsWith("/examples/saves/") && pathname !== "/.scratch/browser-proof/workload.json") { response.writeHead(404).end(); return; }
    const path = resolve(root, `.${pathname}`);
    if (!path.startsWith(resolve(root, "web") + sep) && !path.startsWith(resolve(root, "examples/saves") + sep) && path !== resolve(root, ".scratch/browser-proof/workload.json")) { response.writeHead(403).end(); return; }
    response.setHeader("Content-Type", mime[extname(path)] || "application/octet-stream");
    response.end(await readFile(path));
  } catch { response.writeHead(404).end(); }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
mkdirSync(resolve(root, ".scratch/browser-proof"), { recursive: true });
const profile = await mkdtemp(resolve(root, ".scratch/browser-proof/chrome-"));
const browser = spawn(candidate, ["--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"], { windowsHide: true, stdio: ["ignore", "ignore", "pipe"] });
let browserLog = "";
browser.stderr.on("data", (data) => { browserLog = (browserLog + data).slice(-4096); });
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let websocket;
try {
  let port;
  for (let i = 0; i < 100; i++) { try { port = (await readFile(resolve(profile, "DevToolsActivePort"), "utf8")).split("\n")[0]; break; } catch { await delay(100); } }
  if (!port) throw new Error("Headless browser did not expose its DevTools endpoint.");
  console.log("Browser DevTools port:", port);
  const info = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  websocket = new WebSocket(info.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { websocket.addEventListener("open", resolve, { once: true }); websocket.addEventListener("error", reject, { once: true }); });
  console.log("DevTools websocket connected.");
  let sequence = 0; const waiting = new Map();
  websocket.addEventListener("message", ({ data }) => { const message = JSON.parse(data); if (message.id) { const pending = waiting.get(message.id); if (!pending) return; waiting.delete(message.id); clearTimeout(pending.timer); if (message.error) pending.reject(new Error(JSON.stringify(message.error))); else pending.resolve(message.result); } });
  websocket.addEventListener("close", () => {
    for (const pending of waiting.values()) { clearTimeout(pending.timer); pending.reject(new Error(`Browser closed before checks completed.\n${browserLog}`)); }
    waiting.clear();
  });
  const command = (method, params = {}, sessionId) => new Promise((resolve, reject) => { const id = ++sequence; const timer = setTimeout(() => { waiting.delete(id); reject(new Error(`DevTools command timed out: ${method}\n${browserLog}`)); }, method === "Runtime.evaluate" ? 60000 : 20000); waiting.set(id, { resolve, reject, timer }); websocket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) })); });
  console.log("Creating browser test target.");
  const { targetId } = await command("Target.createTarget", { url: "about:blank" });
  console.log("Attaching browser test target.");
  const { sessionId } = await command("Target.attachToTarget", { targetId, flatten: true });
  await command("Runtime.enable", {}, sessionId); await command("Page.enable", {}, sessionId);
  const downloads = resolve(root, ".scratch/browser-proof/downloads"); mkdirSync(downloads, { recursive: true });
  const previousDownloads = new Set(await readdir(downloads));
  await command("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: downloads });
  const evaluate = async (expression) => {

    const result = await command("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }, sessionId);
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
  await command("Page.navigate", { url: `${origin}/web/tests/browser.html` }, sessionId);
  for (let i = 0; i < 100; i++) { if (await evaluate("typeof window.browserChecks !== 'undefined'")) break; await delay(100); }
  const results = await evaluate("window.browserChecks");
  if (!results?.workloads) throw new Error("Browser workload checks did not complete.");
  console.log(JSON.stringify({ browser: info.Browser, ...results }));
  // Exercise the actual graphical setup/play UI and its worker.
  await command("Page.navigate", { url: `${origin}/web/` }, sessionId);
  for (let i = 0; i < 100; i++) { if (await evaluate("typeof window.gameReady !== 'undefined'")) break; await delay(100); }
  const human = await evaluate("(async () => { const checks = await import('./tests/human-play.js'); return checks.runHumanChecks(); })()");
  console.log(JSON.stringify({ humanPlay: human }));
  const ai = await evaluate("(async () => { const checks = await import('./tests/ai-play.js'); return checks.runAiChecks(); })()");
  console.log(JSON.stringify({ aiPlay: ai }));
  const review = await evaluate("(async () => { const checks = await import('./tests/review.js'); return checks.runReviewChecks(); })()");
  console.log(JSON.stringify({ review }));
  const files = await evaluate("(async () => { const checks = await import('./tests/files.js'); return checks.runFileChecks(); })()");
  console.log(JSON.stringify({ files }));
  let downloaded = [];
  for (let i = 0; i < 100; i++) {
    downloaded = (await readdir(downloads)).filter(name => !previousDownloads.has(name) && name.endsWith(".json"));
    if (downloaded.length >= files.downloads) break;
    await delay(100);
  }
  if (downloaded.length < files.downloads) throw new Error("Requested browser downloads did not complete.");
  for (const name of downloaded) {
    const artifact = JSON.parse(await readFile(resolve(downloads, name), "utf8"));
    if (!["tag-state", "tag-replay"].includes(artifact.kind)) throw new Error("Downloaded file has an unexpected format.");
  }
  const downloadedText = await readFile(resolve(downloads, downloaded[0]), "utf8");
  await evaluate(`(async () => { const { BrowserClient } = await import('./client.js'); const client = new BrowserClient(); try { await client.requestFile({type:'import',controllers:null}, ${JSON.stringify(downloadedText)}); return true; } finally { client.dispose(); } })()`);
  console.log("Downloaded files verified on disk and re-imported.");
  await command("Page.bringToFront", {}, sessionId);
  await evaluate("document.querySelector('#board button:enabled').focus()");
  await command("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, text: "\r", unmodifiedText: "\r" }, sessionId);
  await command("Input.dispatchKeyEvent", { type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 }, sessionId);
  for (let i = 0; i < 100; i++) { if (await evaluate("document.querySelector('#details').textContent.includes('| 1 moves |') && document.querySelector('#board button:enabled') !== null")) break; await delay(100); }
  if (!await evaluate("document.querySelector('#details').textContent.includes('| 1 moves |') && document.activeElement === document.querySelector('#board button:enabled')")) throw new Error(`Keyboard play/focus failed: ${JSON.stringify(await evaluate("({details:document.querySelector('#details').textContent,status:document.querySelector('#status').textContent,focused:document.activeElement?.outerHTML,available:document.querySelector('#board button:enabled')?.outerHTML})"))}`);
  await command("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true }, sessionId);
  if (!await evaluate("document.documentElement.scrollWidth <= innerWidth")) throw new Error("Human play overflows a narrow viewport.");
  const humanScreenshots = resolve(root, ".scratch/human-play/screenshots"); mkdirSync(humanScreenshots, { recursive: true });
  const humanMobile = await command("Page.captureScreenshot", { format: "png" }, sessionId);
  await (await import("node:fs/promises")).writeFile(resolve(humanScreenshots, "mobile.png"), Buffer.from(humanMobile.data, "base64"));
  await command("Emulation.setDeviceMetricsOverride", { width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false }, sessionId);
  console.log("Human play keyboard focus and narrow layout checks passed.");
  for (const [width, height] of [[1280, 1000], [390, 844]]) {
    await command("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width < 840 }, sessionId);
    console.log(JSON.stringify({ layout: await evaluate("(async () => { const checks = await import('./tests/review.js'); return checks.runLayoutChecks(); })()") }));
  }
  await command("Emulation.setDeviceMetricsOverride", { width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false }, sessionId);
  // Learned-model files drive the actual graphical setup and worker play.
  await command("Page.navigate", { url: `${origin}/web/` }, sessionId);
  for (let i = 0; i < 100; i++) { if (await evaluate("typeof window.gameReady !== 'undefined'")) break; await delay(100); }
  console.log(JSON.stringify({ learnedPlay: await evaluate("(async () => { const checks = await import('./tests/learned-play.js'); return checks.runLearnedPlayChecks(); })()") }));
  // Exercise the Stage 6 preview using real image/audio decoders and engine-produced samples.
  await command("Page.navigate", { url: `${origin}/web/presentation.html` }, sessionId);
  for (let i = 0; i < 100; i++) { if (await evaluate("typeof window.presentationReady !== 'undefined'")) break; await delay(100); }
  const preview = await evaluate("window.presentationReady");
  if (preview?.samples !== 7 || preview?.assets !== 13) throw new Error("Presentation samples/assets were not prepared.");
  await evaluate("document.querySelector('#setup-open').click()");
  if (!await evaluate("document.querySelector('#setup-preview').open")) throw new Error("Setup preview did not open.");
  await evaluate("document.querySelector('#setup-game').value = 'connect4'; document.querySelector('#setup-game').dispatchEvent(new Event('change'))");
  if (!await evaluate("[...document.querySelectorAll('#setup-parameters input')].map(input => Number(input.value)).join(',') === '7,6,4'")) throw new Error("Setup preview defaults differ from the game.");
  await evaluate("document.querySelector('#setup-preview').close()");
  await evaluate("document.querySelector('#theme').value = 'dark'; document.querySelector('#theme').dispatchEvent(new Event('change')); document.querySelector('#reduced').checked = true; document.querySelector('#mute').checked = true; document.querySelector('#mute').dispatchEvent(new Event('change'))");
  if (!await evaluate("document.body.dataset.theme === 'dark'")) throw new Error("Preview theme did not change.");
  for (let scene = 0; scene < preview.samples; scene++) {
    await evaluate(`document.querySelector('#scene').value = '${scene}'; document.querySelector('#scene').dispatchEvent(new Event('change')); document.querySelector('#finished').click()`);
    if (!await evaluate("document.querySelector('#result').textContent === 'Draw' || document.querySelector('#result').textContent.includes('wins')")) throw new Error("Terminal preview position failed.");
    await evaluate("Promise.all([...document.querySelectorAll('#board img')].filter(image => !image.hidden).map(image => image.decode()))");
    if (!await evaluate("[...document.querySelectorAll('#board img')].filter(image => !image.hidden).every(image => image.complete && image.naturalWidth > 0)")) throw new Error("Board artwork failed to load.");
  }
  await evaluate("document.querySelector('#scene').value = '4'; document.querySelector('#scene').dispatchEvent(new Event('change')); document.querySelector('#step').click()");
  for (let i = 0; i < 100; i++) { if (await evaluate("!document.querySelector('#step').disabled")) break; await delay(100); }
  if (!await evaluate("document.querySelector('#position-details').textContent.includes('Move 1 of') && document.querySelector('#board').getAnimations({subtree:true}).length === 0")) throw new Error("Reduced-motion sample step failed.");
  await command("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true }, sessionId);
  if (!await evaluate("document.documentElement.scrollWidth <= innerWidth")) throw new Error("Preview overflows a narrow viewport.");
  const decoded = await evaluate("(async () => { const manifest = await (await fetch('assets/manifest.json')).json(); const context = new AudioContext(); const results = []; try { for (const asset of manifest.assets.filter(asset => asset.path.endsWith('.ogg'))) { const data = await (await fetch('assets/' + asset.path)).arrayBuffer(); const buffer = await context.decodeAudioData(data); results.push({id:asset.id, seconds:buffer.duration}); } return results; } finally { await context.close(); } })()");
  if (decoded.length !== 6 || decoded.some(cue => !(cue.seconds > 0))) throw new Error("Sound cues failed to decode.");
  const screenshots = resolve(root, ".scratch/presentation/screenshots"); mkdirSync(screenshots, { recursive: true });
  const mobileShot = await command("Page.captureScreenshot", { format: "png" }, sessionId);
  await (await import("node:fs/promises")).writeFile(resolve(screenshots, "mobile.png"), Buffer.from(mobileShot.data, "base64"));
  await command("Emulation.setDeviceMetricsOverride", { width: 1280, height: 1000, deviceScaleFactor: 1, mobile: false }, sessionId);
  for (const theme of ["light", "dark"]) {
    await evaluate(`document.querySelector('#theme').value = '${theme}'; document.querySelector('#theme').dispatchEvent(new Event('change')); document.querySelector('#finished').click()`);
    const shot = await command("Page.captureScreenshot", { format: "png" }, sessionId);
    await (await import("node:fs/promises")).writeFile(resolve(screenshots, `${theme}.png`), Buffer.from(shot.data, "base64"));
  }
  console.log(JSON.stringify({ presentation: preview, checks: ["setup defaults", "themes", "all terminal samples", "images", "reduced motion", "narrow viewport", "audio decoding"], sounds: decoded }));
  console.log("Presentation screenshots saved under .scratch/presentation/screenshots.");
  await command("Browser.close").catch(() => {});
} finally { websocket?.close(); browser.kill(); server.close(); }
