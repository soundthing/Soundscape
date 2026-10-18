// Developer choice: "keep-playing" or "pause-until-ready". Both paths are tested.
export const PLAYBACK_POLICY = "keep-playing";
export const PREFERENCES_KEY = "soundthingPreferences";
export const POLL_INTERVAL_MS = 1500;
// Display only: each cue fades through this extra span after its VTT end time.
export const CAPTION_TAIL_SEC = 0.75;

// Display-only direction response; backend channel levels and cached tracks stay unchanged.
export const DOA_VISUAL = Object.freeze({
  noiseFloor: 0.04,
  loudnessExponent: 2.4,
  directionBias: 0.45,
  maxDisplacementPx: 44,
  spanScale: 0.8,
  pulseDepth: 0.2,
  pulseHz: 1.4,
  swayPx: 8,
  swayHz: 0.14,
});

export function watchVideoId(value) {
  try {
    const url = new URL(value);
    const id = url.searchParams.get("v");
    return url.origin === "https://www.youtube.com" && url.pathname === "/watch" && /^[A-Za-z0-9_-]{11}$/.test(id || "") ? id : null;
  } catch { return null; }
}
