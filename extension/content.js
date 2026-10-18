(async () => {
  await import(chrome.runtime.getURL("console.js"));
  const { CaptionController } = await import(chrome.runtime.getURL("player.mjs"));
  const { watchVideoId } = await import(chrome.runtime.getURL("settings.mjs"));
  const KEY = "soundthingPreferences";
  const controller = new CaptionController({ window, document, send: (message) => chrome.runtime.sendMessage(message) });
  const read = async () => {
    const saved = (await chrome.storage.local.get(KEY))[KEY] || {};
    controller.configure({ enabled: saved.enabled === true, server: saved.server || "http://127.0.0.1:8000" });
  };
  await read();
  // Bridge only this tab's video to the worker; the page never supplies a server/path.
  window.addEventListener("message", async (event) => {
    const message = event.data;
    if (event.source !== window || event.origin !== location.origin || message?.type !== "soundthing:clear-cache"
        || typeof message.requestId !== "string" || message.requestId.length > 100) return;
    let result;
    try {
      const youtubeId = watchVideoId(location.href);
      if (!youtubeId) throw new Error("Open a YouTube watch page to clear its cache.");
      result = await chrome.runtime.sendMessage({ type: "soundthing:api", action: "clearCache", youtubeId, url: message.url });
      if (result?.ok && result.cleared) controller.invalidateCache(youtubeId);
    } catch (error) {
      result = { ok: false, error: error.message || "Couldn't clear the video cache." };
    }
    window.postMessage({ type: "soundthing:clear-cache-result", requestId: message.requestId, result }, location.origin);
  });
  chrome.storage.onChanged.addListener((changes, area) => { if (area === "local" && changes[KEY]) read(); });
  chrome.runtime.onMessage.addListener((message, sender, reply) => {
    if (sender.id !== chrome.runtime.id) return;
    if (message?.type === "soundthing:inspect") { controller.refresh(); reply(controller.state); }
    if (message?.type === "soundthing:retry") { controller.retry(); reply(controller.state); }
  });
  let interval = setInterval(() => controller.refresh(), 500);
  document.addEventListener("yt-navigate-finish", () => controller.refresh());
  window.addEventListener("pagehide", () => { clearInterval(interval); interval = null; controller.disposeSession(); });
  window.addEventListener("pageshow", () => {
    if (interval === null) { read(); interval = setInterval(() => controller.refresh(), 500); }
  });
})().catch((error) => console.error("Soundthing couldn't initialize:", error));
