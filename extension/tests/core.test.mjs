import test from "node:test";
import assert from "node:assert/strict";
import { apiUrl, normalizeServer, youtubeVideoId } from "../core.mjs";

test("server configuration stays within the extension's local host permissions", () => {
  assert.equal(normalizeServer("http://localhost:8123/"), "http://localhost:8123");
  assert.equal(normalizeServer("http://127.0.0.1:8000"), "http://127.0.0.1:8000");
  for (const value of ["https://localhost", "http://localhost.evil.test", "http://192.168.1.2", "http://user:secret@localhost", "http://localhost/api", "http://localhost?token=x", "file:///tmp/video", "nonsense"]) {
    assert.throws(() => normalizeServer(value), undefined, value);
  }
});

test("YouTube URL parsing supports creator links and rejects lookalike hosts and playlists", () => {
  const id = "aB3dE_fG-12";
  for (const value of [id, `https://www.youtube.com/watch?v=${id}&t=45`, `https://youtu.be/${id}?si=example`, `https://youtube.com/shorts/${id}`, `https://m.youtube.com/live/${id}`, `https://www.youtube.com/embed/${id}`, `https://studio.youtube.com/video/${id}/edit`]) {
    assert.equal(youtubeVideoId(value), id);
  }
  for (const value of [`https://youtube.com.evil.test/watch?v=${id}`, `https://evil.test/${id}`, `https://youtube.com/playlist?list=${id}`, `https://youtube.com@evil.test/watch?v=${id}`, `javascript:alert(1)`, "https://youtube.com/watch?v=short", `https://www.youtube.com:8443/watch?v=${id}`]) {
    assert.throws(() => youtubeVideoId(value), undefined, value);
  }
});

test("API paths cannot escape their intended locations", () => {
  assert.equal(apiUrl("http://localhost:8000", "my-video", "/captions/baseline.vtt"), "http://localhost:8000/api/videos/my-video/captions/baseline.vtt");
  for (const id of [undefined, null, "../secret", "", "video?query=x", "video/extra"]) assert.throws(() => apiUrl("http://localhost", id));
  assert.throws(() => apiUrl("http://localhost", "video", "/../secret"));
});
