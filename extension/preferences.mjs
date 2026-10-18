import { DEFAULT_SERVER, normalizeServer } from "./core.mjs";
import { PREFERENCES_KEY } from "./settings.mjs";

export async function readPreferences(storage = chrome.storage.local) {
  const saved = (await storage.get(PREFERENCES_KEY))[PREFERENCES_KEY] || {};
  let server = DEFAULT_SERVER;
  try { server = normalizeServer(saved.server || server); } catch { /* Use localhost. */ }
  return { enabled: saved.enabled === true, server };
}

export async function updatePreferences(update, storage = chrome.storage.local) {
  const preferences = { ...await readPreferences(storage), ...update };
  preferences.server = normalizeServer(preferences.server);
  preferences.enabled = preferences.enabled === true;
  await storage.set({ [PREFERENCES_KEY]: preferences });
  return preferences;
}
