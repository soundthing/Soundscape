import { readPreferences } from "./preferences.mjs";
import { handlePlayerRequest } from "./worker.mjs";

chrome.runtime.onMessage.addListener((message, sender, reply) => {
  if (message?.type !== "soundthing:api") return;
  handlePlayerRequest(message, sender, {
    extensionId: chrome.runtime.id,
    getTab: (id) => chrome.tabs.get(id),
    preferences: readPreferences,
  }).then((data) => reply({ ok: true, ...data }), (error) => reply({
    ok: false, error: error.message || "Couldn't reach the local Soundthing server.",
    retryable: error.retryable === true, statusCode: error.statusCode || 0,
  }));
  // The Python server owns long downloads; extension requests stay short.
  return true;
});
