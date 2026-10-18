// Installed in both worlds so normal and extension DevTools contexts can use it.
(() => {
  const pending = new Map();
  window.addEventListener("message", (event) => {
    if (event.source !== window || event.origin !== location.origin || event.data?.type !== "soundthing:clear-cache-result") return;
    const request = pending.get(event.data.requestId);
    if (!request) return;
    pending.delete(event.data.requestId);
    clearTimeout(request.timer);
    const result = event.data.result;
    if (result?.ok) {
      console.info(`[Soundthing] Cache cleared for ${result.video_id}. Refresh YouTube or choose Regenerate captions in the popup.`, result);
      request.resolve(result);
    }
    else request.reject(new Error(result?.error || "Couldn't clear the video cache."));
  });
  const clearCache = (url = location.href) => new Promise((resolve, reject) => {
    if (typeof url !== "string") { reject(new Error("Provide a YouTube URL or video ID.")); return; }
    const requestId = crypto.randomUUID();
    const timer = setTimeout(() => {
      pending.delete(requestId);
      reject(new Error("Cache request timed out. Reload the Soundthing extension and refresh YouTube."));
    }, 15000);
    pending.set(requestId, { resolve, reject, timer });
    window.postMessage({ type: "soundthing:clear-cache", requestId, url }, location.origin);
  });
  window.soundthing = { ...window.soundthing, clearCache };
})();
