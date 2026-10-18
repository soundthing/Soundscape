(async () => {
  const { CaptionController } = await import(chrome.runtime.getURL("player.mjs"));
  const KEY = "soundthingPreferences";
  const controller = new CaptionController({ window, document, send: (message) => chrome.runtime.sendMessage(message) });
  const read = async () => {
    const saved = (await chrome.storage.local.get(KEY))[KEY] || {};
    controller.configure({ enabled: saved.enabled === true, server: saved.server || "http://127.0.0.1:8000" });
  };
  await read();
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
