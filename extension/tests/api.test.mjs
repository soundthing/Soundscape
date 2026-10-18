import test from "node:test";
import assert from "node:assert/strict";
import { ApiError, requestJson, requestText } from "../api.mjs";

const server = "http://localhost:8000";

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
