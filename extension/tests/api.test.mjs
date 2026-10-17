import test from "node:test";
import assert from "node:assert/strict";
import { ApiError, checkConnection, getStatus, requestJson, requestText, uploadVideo } from "../api.mjs";

const server = "http://localhost:8000";

test("connection requires the existing Soundthing upload endpoint", async () => {
  await checkConnection(server, async (url, options) => {
    assert.equal(url, `${server}/openapi.json`);
    assert.equal(options.credentials, "omit");
    assert.equal(options.redirect, "error");
    return Response.json({ paths: { "/api/videos": {} } });
  });
  await assert.rejects(checkConnection(server, async () => Response.json({ paths: {} })), /isn't a Soundthing/);
  await assert.rejects(checkConnection(server, async () => Response.json(null)), /isn't a Soundthing/);
});

test("upload sends multipart video bytes using the backend's existing file field", async () => {
  const file = new File(["video bytes"], "clip.mp4", { type: "video/mp4" });
  const id = await uploadVideo(server, file, "clip-unique.mp4", async (url, options) => {
    assert.equal(url, `${server}/api/videos`);
    assert.equal(options.method, "POST");
    const uploaded = options.body.get("file");
    assert.equal(uploaded.name, "clip-unique.mp4");
    assert.equal(await uploaded.text(), "video bytes");
    return Response.json({ video_id: "clip-unique" });
  });
  assert.equal(id, "clip-unique");
  await assert.rejects(uploadVideo(server, file, "clip-unique.mp4", async () => Response.json({})), /invalid video ID/);
});

test("job status distinguishes processing failures from connection errors", async () => {
  const job = { server, videoId: "clip" };
  for (const status of ["queued", "processing", "done"]) {
    assert.deepEqual(await getStatus(job, async () => Response.json({ status })), { status, error: "" });
  }
  assert.deepEqual(await getStatus(job, async () => Response.json({ status: "error", error: "ffmpeg not found" })), { status: "error", error: "ffmpeg not found" });
  await assert.rejects(getStatus(job, async () => { throw new TypeError("Failed to fetch"); }), /Couldn't reach Soundthing/);
  await assert.rejects(getStatus(job, async () => Response.json({ status: "unexpected" })), /unknown processing status/);
  await assert.rejects(getStatus(job, async () => Response.json(null)), /unknown processing status/);
});

test("HTTP, timeout, non-JSON, and invalid response failures are actionable", async () => {
  await assert.rejects(requestJson(server, {}, async () => Response.json({ detail: "unsupported file type '.exe'" }, { status: 400 })), /unsupported file type/);
  await assert.rejects(requestJson(server, {}, async () => new Response("Not Found", { status: 404 })), /404/);
  await assert.rejects(requestJson(server, {}, async () => new Response("<html>wrong server</html>")), /unreadable response/);
  await assert.rejects(requestJson(server, {}, async () => { throw new DOMException("timed out", "TimeoutError"); }), /too long/);
});

test("temporary outages are retryable but invalid input remains a terminal error", async () => {
  for (const status of [400, 404, 408, 429, 503]) {
    await assert.rejects(requestJson(server, {}, async () => Response.json({ detail: "fixture" }, { status })), (error) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.statusCode, status);
      assert.equal(error.retryable, [408, 429, 503].includes(status));
      return true;
    });
  }
  await assert.rejects(requestJson(server, {}, async () => { throw new TypeError("network disconnected"); }), (error) => error.retryable === true);
});

test("VTT requests preserve recovery signals and reject redirects / credential forwarding", async () => {
  assert.equal(await requestText(server, {}, async (url, options) => {
    assert.equal(options.credentials, "omit");
    assert.equal(options.redirect, "error");
    return new Response("WEBVTT\n\n");
  }), "WEBVTT\n\n");
  await assert.rejects(requestText(server, {}, async () => new Response("", { status: 404 })), (error) => error.statusCode === 404);
  await assert.rejects(requestText(server, {}, async () => new Response("", { status: 503 })), (error) => error.retryable === true);
});
