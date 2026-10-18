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
    : "Couldn't reach the local server. Start it and try again.", { retryable: true });
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
