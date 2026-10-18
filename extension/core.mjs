export const DEFAULT_SERVER = "http://127.0.0.1:8000";
export const ACTIVE_STATUSES = new Set(["queued", "processing"]);
export const VIDEO_EXTENSIONS = new Set(["mp4", "mov", "webm"]);

export function normalizeServer(value) {
  let url;
  try { url = new URL(value); } catch { throw new Error("Enter a local server URL, such as http://127.0.0.1:8000."); }
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname) ||
      url.username || url.password || url.pathname !== "/" || url.search || url.hash) {
    throw new Error("Use http://127.0.0.1 or http://localhost with your server port, without a path.");
  }
  return url.origin;
}

export function validateVideo(file) {
  const extension = file.name.split(".").pop().toLowerCase();
  if (!VIDEO_EXTENSIONS.has(extension)) throw new Error("Choose an MP4, MOV, or WebM video.");
  if (!file.size) throw new Error("This file is empty. Choose a video with content.");
  return extension;
}

export function uploadFilename(file, token) {
  const extension = validateVideo(file);
  const stem = file.name.slice(0, -(extension.length + 1)).replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 60) || "video";
  return `${stem}-${token}.${extension}`;
}

export function apiUrl(server, videoId, resource = "") {
  if (typeof videoId !== "string" || !/^[A-Za-z0-9_-]+$/.test(videoId)) throw new Error("The server returned an invalid video ID.");
  if (!["", "/video", "/captions/baseline.vtt", "/captions/conditioned.vtt", "/direction"].includes(resource)) {
    throw new Error("Unknown video resource.");
  }
  return `${normalizeServer(server)}/api/videos/${videoId}${resource}`;
}

export function youtubeVideoId(value) {
  const input = value.trim();
  if (/^[A-Za-z0-9_-]{11}$/.test(input)) return input;
  let url;
  try { url = new URL(input); } catch { throw new Error("Paste a YouTube video link or its 11-character video ID."); }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.port) {
    throw new Error("Paste a YouTube video link or its 11-character video ID.");
  }
  let id;
  if (url.hostname === "youtu.be") id = url.pathname.split("/")[1];
  else if (["youtube.com", "www.youtube.com", "m.youtube.com"].includes(url.hostname)) {
    const [, route, pathId] = url.pathname.split("/");
    id = route === "watch" ? url.searchParams.get("v") : ["shorts", "live", "embed"].includes(route) ? pathId : null;
  } else if (url.hostname === "studio.youtube.com") {
    const [, route, pathId] = url.pathname.split("/");
    if (route === "video") id = pathId;
  }
  if (!id || !/^[A-Za-z0-9_-]{11}$/.test(id)) throw new Error("That link doesn't contain a valid YouTube video ID.");
  return id;
}

export function studioUrl(value) {
  return `https://studio.youtube.com/video/${youtubeVideoId(value)}/edit`;
}

export function downloadName(name, mode) {
  const safe = name.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_").replace(/^[. ]+|[. ]+$/g, "").slice(0, 150) || "video.mp4";
  if (mode === "video") return `soundthing/${safe}`;
  if (!["baseline", "conditioned"].includes(mode)) throw new Error("Unknown caption mode.");
  return `soundthing/${safe.replace(/\.[^.]+$/, "")}.${mode}.vtt`;
}

export function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
