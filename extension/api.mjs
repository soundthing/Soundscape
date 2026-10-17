import { apiUrl, normalizeServer } from "./core.mjs";

export class ApiError extends Error {
  constructor(message, { statusCode = 0, retryable = false } = {}) {
    super(message);
    this.statusCode = statusCode;
    this.retryable = retryable;
  }
}

function connectionError(error) {
  return new ApiError(error.name === "TimeoutError" || error.name === "AbortError"
    ? "The server took too long to respond. Check that Soundthing is running."
    : "Couldn't reach Soundthing. Start the local server and check its address in Connection.", { retryable: true });
}

export async function requestJson(url, options = {}, fetcher = fetch) {
  let response;
  try {
    response = await fetcher(url, { credentials: "omit", redirect: "error", ...options });
  } catch (error) {
    throw connectionError(error);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : `Server request failed (${response.status}).`;
    throw new ApiError(detail, { statusCode: response.status, retryable: response.status >= 500 || [408, 429].includes(response.status) });
  }
  try { return await response.json(); } catch (error) {
    if (["TypeError", "TimeoutError", "AbortError"].includes(error.name)) throw connectionError(error);
    throw new ApiError("Soundthing returned an unreadable response.");
  }
}

export async function requestText(url, options = {}, fetcher = fetch) {
  let response;
  try { response = await fetcher(url, { credentials: "omit", redirect: "error", ...options }); }
  catch (error) { throw connectionError(error); }
  if (!response.ok) throw new ApiError(`Couldn't load the caption track (${response.status}).`, {
    statusCode: response.status, retryable: response.status >= 500 || [408, 429].includes(response.status),
  });
  try { return await response.text(); }
  catch (error) { throw connectionError(error); }
}

export async function checkConnection(server, fetcher) {
  const data = await requestJson(`${normalizeServer(server)}/openapi.json`, { signal: AbortSignal.timeout(8000) }, fetcher);
  if (!data?.paths?.["/api/videos"]) throw new Error("This address isn't a Soundthing server.");
}

export async function uploadVideo(server, file, filename, fetcher) {
  const body = new FormData();
  body.append("file", file, filename);
  const data = await requestJson(`${normalizeServer(server)}/api/videos`, { method: "POST", body }, fetcher);
  apiUrl(server, data?.video_id);
  return data.video_id;
}

export async function getStatus(job, fetcher) {
  const data = await requestJson(apiUrl(job.server, job.videoId), { signal: AbortSignal.timeout(8000) }, fetcher);
  if (!["queued", "processing", "done", "error"].includes(data?.status)) throw new Error("Soundthing returned an unknown processing status.");
  return { status: data.status, error: data.status === "error" ? String(data.error || "Processing failed.") : "" };
}
