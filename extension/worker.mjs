import { apiUrl } from "./core.mjs";
import { requestJson, requestText } from "./api.mjs";
import { watchVideoId } from "./settings.mjs";
import { validateDirection } from "./direction.mjs";

export async function handlePlayerRequest(message, sender, { extensionId, getTab, preferences, fetcher = fetch }) {
  if (sender.id !== extensionId || sender.frameId !== 0 || !Number.isInteger(sender.tab?.id)) throw new Error("Unsupported message sender.");
  if (!["ensure", "status", "captions", "direction"].includes(message.action)) throw new Error("Unknown player request.");
  const tab = await getTab(sender.tab.id);
  const id = watchVideoId(tab.url);
  // sender.url can retain the document's original URL across YouTube SPA routes.
  // Its origin authenticates the frame; the browser's current tab URL binds the ID.
  let senderOrigin;
  try { senderOrigin = new URL(sender.url).origin; } catch { /* Reject below. */ }
  if (!id || id !== message.youtubeId || senderOrigin !== "https://www.youtube.com") throw new Error("The YouTube video changed. Refresh the player state.");
  const prefs = await preferences();
  if (!prefs.enabled) throw new Error("Soundthing is off.");
  const videoId = `youtube_${id}`;
  const options = { signal: AbortSignal.timeout(10000) };
  if (message.action === "ensure") {
    const data = await requestJson(`${prefs.server}/api/youtube`, { ...options, method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ youtube_id: id }) }, fetcher);
    if (data?.video_id !== videoId) throw new Error("The server returned an unexpected video ID.");
    return { videoId };
  }
  if (message.action === "status") {
    const data = await requestJson(apiUrl(prefs.server, videoId), options, fetcher);
    if (!["queued", "downloading", "processing", "done", "error"].includes(data?.status)) throw new Error("Unknown processing status.");
    return { status: data.status, error: data.status === "error" ? String(data.error || "Processing failed.") : "", demo: data.demo === true };
  }
  if (message.action === "direction") {
    const direction = validateDirection(await requestJson(apiUrl(prefs.server, videoId, "/direction"), options, fetcher));
    return { direction };
  }
  const vtt = await requestText(apiUrl(prefs.server, videoId, "/captions/conditioned.vtt"), options, fetcher);
  if (!vtt.trimStart().startsWith("WEBVTT")) throw new Error("The server returned an invalid VTT track.");
  return { vtt };
}
