export const DEFAULT_SERVER = "http://127.0.0.1:8000";

export function normalizeServer(value) {
  let url;
  try { url = new URL(value); } catch { throw new Error("Enter a local server URL, such as http://127.0.0.1:8000."); }
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname) ||
      url.username || url.password || url.pathname !== "/" || url.search || url.hash) {
    throw new Error("Use http://127.0.0.1 or http://localhost with your server port, without a path.");
  }
  return url.origin;
}

export function apiUrl(server, videoId, resource = "") {
  if (typeof videoId !== "string" || !/^[A-Za-z0-9_-]+$/.test(videoId)) throw new Error("The server returned an invalid video ID.");
  if (!["", "/video", "/captions/baseline.vtt", "/captions/conditioned.vtt", "/direction", "/env_segments"].includes(resource)) {
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
