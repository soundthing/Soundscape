// Developer choice: "keep-playing" or "pause-until-ready". Both paths are tested.
export const PLAYBACK_POLICY = "keep-playing";
export const PREFERENCES_KEY = "soundthingPreferences";
export const POLL_INTERVAL_MS = 1500;

export function watchVideoId(value) {
  try {
    const url = new URL(value);
    const id = url.searchParams.get("v");
    return url.origin === "https://www.youtube.com" && url.pathname === "/watch" && /^[A-Za-z0-9_-]{11}$/.test(id || "") ? id : null;
  } catch { return null; }
}
