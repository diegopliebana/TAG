// One module worker owns the Wasm engine; browser rendering runs on the main thread.
import init, { BrowserEngine } from "./pkg/tag_browser.js";
try {
  await init();
  const engine = new BrowserEngine();
  self.onmessage = ({ data }) => {
    try {
      const file = data?.type === "file";
      const reply = JSON.parse(file ? engine.handle_file(data.envelope, data.text) : engine.handle(data));
      // Correlate size/JSON errors using the worker's known request envelope.
      if (!reply.request_id) {
        const request = JSON.parse(file ? data.envelope : data);
        if (typeof request.request_id !== "string" || typeof request.generation !== "string") throw new Error("Invalid command envelope.");
        reply.request_id = request.request_id;
        reply.request_generation = request.generation;
      }
      self.postMessage(reply);
    }
    catch (error) { self.postMessage({ type: "failure", message: String(error) }); }
  };
  self.postMessage({ type: "ready" });
} catch (error) {
  self.postMessage({ type: "failure", message: String(error) });
}
