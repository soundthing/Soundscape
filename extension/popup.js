import { readPreferences, updatePreferences } from "./preferences.mjs";
import { checkConnection } from "./api.mjs";
import { watchVideoId } from "./settings.mjs";

const $ = (id) => document.getElementById(id);
let refreshing = false;

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const preferences = await readPreferences();
    $("enabled").checked = preferences.enabled;
    if (document.activeElement !== $("server")) $("server").value = preferences.server;
    let state = { status: "disabled" };
    if (preferences.enabled) {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && watchVideoId(tab.url)) {
        try { state = await chrome.tabs.sendMessage(tab.id, { type: "soundthing:inspect" }); }
        catch { state = { status: "error", error: "Refresh this YouTube page to activate the updated extension." }; }
      } else state = { status: "unsupported" };
    }
    const labels = {
      disabled: ["Soundthing is off", "Turn on to caption the videos you open."],
      unsupported: ["Ready for YouTube", "Open a standard YouTube video. Shorts, embeds, and live streams aren't supported."],
      queued: ["Video queued", "Your local server will download and process this video."],
      downloading: ["Downloading your video", "You can close this popup. The local server will keep working."],
      processing: ["Creating sound captions", "The caption track will appear inside the player when ready."],
      reconnecting: ["Reconnecting to Soundthing", "Check that the local server is running. Retrying automatically; you can close this popup."],
      done: ["Captions are on", "Synchronized with the video, at the bottom right of the player."],
      "cache-cleared": ["Video cache cleared", "Refresh YouTube or regenerate this video's captions."],
      error: ["Something needs attention", state.error || "Check the local server, then retry this video."],
    };
    const [title, detail] = labels[state.status] || labels.queued;
    $("status-title").textContent = title;
    $("status-detail").textContent = detail;
    if (state.status === "done") {
      const direction = { stereo: "Stereo direction waves are on.", surround51: "5.1 surround direction waves are on.", unavailable: "Directional audio is unavailable for this track.", pending: "Checking directional audio…" };
      $("status-detail").textContent += ` ${direction[state.directionMode] || direction.pending}`;
    }
    $("status-dot").className = state.status === "done" ? "ready" : state.status === "error" ? "error" : "";
    $("retry").hidden = !["error", "reconnecting", "cache-cleared"].includes(state.status);
    $("retry").textContent = state.status === "cache-cleared" ? "Regenerate captions" : state.status === "reconnecting" ? "Retry now" : "Retry this video";
    $("demo").hidden = !state.demo;
  } catch (error) {
    $("status-title").textContent = "Couldn't load extension settings";
    $("status-detail").textContent = error.message;
  } finally { refreshing = false; }
}

$("enabled").addEventListener("change", async () => {
  try { await updatePreferences({ enabled: $("enabled").checked }); }
  catch (error) { $("status-detail").textContent = error.message; }
  await refresh();
});
$("retry").addEventListener("click", async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.tabs.sendMessage(tab.id, { type: "soundthing:retry" });
  } catch { $("status-detail").textContent = "Refresh this YouTube page, then try again."; }
  await refresh();
});
$("connection").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button");
  button.disabled = true;
  try {
    await checkConnection($("server").value);
    await updatePreferences({ server: $("server").value });
    $("connection-message").textContent = "Connected. Existing server jobs will finish; this address is used for new requests.";
  } catch (error) { $("connection-message").textContent = error.message; }
  finally { button.disabled = false; }
});
$("workspace").addEventListener("click", () => chrome.tabs.create({ url: chrome.runtime.getURL("index.html") }));
refresh();
setInterval(refresh, 1000);
