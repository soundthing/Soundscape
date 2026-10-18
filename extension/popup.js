import { readPreferences, updatePreferences } from "./preferences.mjs";
import { watchVideoId } from "./settings.mjs";

const toggle = document.getElementById("enabled");
const mark = document.getElementById("title-mark");
const detail = document.getElementById("status-detail");
const retry = document.getElementById("retry");
const clickCaption = document.getElementById("click-caption");
const beach = document.querySelector(".beach");
const surfCaption = document.querySelector(".surf-caption");
const surfWords = ["splish", "splosh", "splish", "splash"];
const surfMargin = 24;
let nextSurfWord = 0;
let surfAnimation;
let refreshing = false;
let saving = false;
let revision = 0;
let clickAnimation;

function showEnabled(enabled) {
  toggle.checked = enabled;
  document.body.dataset.enabled = String(enabled);
  mark.textContent = enabled ? "!" : ".";
  document.title = `soundscape${mark.textContent}`;
  if (!enabled) surfAnimation?.cancel();
}

function popSurfCaption() {
  if (document.body.dataset.enabled !== "true" || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  surfAnimation?.cancel();
  surfCaption.textContent = surfWords[nextSurfWord];
  nextSurfWord = (nextSurfWord + 1) % surfWords.length;

  // Measure the rotated word, including its overhang. Reserve extra vertical
  // room for the entrance/exit motion so the complete caption stays inset.
  surfCaption.style.left = "0px";
  surfCaption.style.top = "0px";
  const shore = beach.getBoundingClientRect();
  const word = surfCaption.getBoundingClientRect();
  const offsetX = word.left - shore.left, offsetY = word.top - shore.top;
  const left = surfMargin - offsetX + Math.random() * Math.max(0, shore.width - 2 * surfMargin - word.width);
  const shoreY = shore.height * (.8 - .13 * (left + word.width / 2) / shore.width);
  const minimumY = surfMargin + 6 - offsetY;
  const maximumY = Math.max(minimumY, shore.height - surfMargin - 6 - offsetY - word.height);
  const top = Math.min(maximumY, Math.max(minimumY, shoreY + Math.random() * 12 - 6));
  surfCaption.style.left = `${left}px`;
  surfCaption.style.top = `${top}px`;
  surfAnimation = surfCaption.animate([
    { opacity: 0, transform: "translateY(4px) rotate(-12deg)" },
    { opacity: 1, transform: "translateY(0) rotate(-12deg)", offset: .18 },
    { opacity: 1, transform: "translateY(0) rotate(-12deg)", offset: .6 },
    { opacity: 0, transform: "translateY(-5px) rotate(-12deg)" },
  ], { duration: 1100, easing: "ease-out" });
}

beach.addEventListener("animationiteration", event => {
  // The waves are half a cycle apart: one finishes as the other hits shore.
  if (event.animationName === "shore-wash") popSurfCaption();
});

matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", event => {
  if (event.matches) surfAnimation?.cancel();
});

function showState(state) {
  const descriptions = {
    disabled: "Enable sound captions",
    unsupported: "Enable sound captions",
    queued: "Waiting to start…",
    downloading: "Getting your video…",
    processing: "Making captions…",
    reconnecting: "Reconnecting to the local server…",
    done: "On for this video.",
    "cache-cleared": "Video cache cleared.",
    error: state.error || "Couldn't process this video.",
  };
  document.body.dataset.status = state.status;
  const message = descriptions[state.status] || descriptions.queued;
  const text = state.demo ? `${message} Demo captions.` : message;
  if (detail.textContent !== text) detail.textContent = text;
  retry.hidden = !["error", "reconnecting", "cache-cleared"].includes(state.status);
  retry.textContent = state.status === "cache-cleared" ? "Regenerate" : "Try again";
}

async function refresh() {
  if (refreshing || saving) return;
  refreshing = true;
  const currentRevision = revision;
  try {
    const preferences = await readPreferences();
    if (saving || currentRevision !== revision) return;
    showEnabled(preferences.enabled);
    toggle.disabled = false;
    let state = { status: "disabled" };
    if (preferences.enabled) {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && watchVideoId(tab.url)) {
        try { state = await chrome.tabs.sendMessage(tab.id, { type: "soundthing:inspect" }); }
        catch { state = { status: "error", error: "Refresh this YouTube tab to get started." }; }
      } else state = { status: "unsupported" };
    }
    if (!saving && currentRevision === revision) showState(state || { status: "queued" });
  } catch (error) {
    if (currentRevision === revision) showState({ status: "error", error: error.message });
  } finally { refreshing = false; }
}

function popClick() {
  clickAnimation?.cancel();
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  clickAnimation = clickCaption.animate(reduced ? [
    { opacity: 0 }, { opacity: 1, offset: .15 }, { opacity: 1, offset: .75 }, { opacity: 0 },
  ] : [
    { opacity: 0, transform: "translateY(6px) rotate(-14deg) scale(.8)" },
    { opacity: 1, transform: "translateY(0) rotate(-14deg) scale(1.08)", offset: .18 },
    { opacity: 1, transform: "translateY(0) rotate(-14deg) scale(1)", offset: .35 },
    { opacity: 1, transform: "translateY(0) rotate(-14deg) scale(1)", offset: .7 },
    { opacity: 0, transform: "translateY(-5px) rotate(-14deg) scale(1)" },
  ], { duration: 850, easing: "ease-out" });
}

toggle.addEventListener("change", async () => {
  const enabled = toggle.checked;
  const hadFocus = document.activeElement === toggle;
  revision++;
  saving = true;
  toggle.disabled = true;
  showEnabled(enabled);
  showState({ status: enabled ? "queued" : "disabled" });
  if (enabled) popClick();
  else clickAnimation?.cancel();
  try {
    await updatePreferences({ enabled });
  } catch (error) {
    showEnabled(!enabled);
    showState({ status: "error", error: `Couldn't save your setting. ${error.message}` });
    return;
  } finally {
    saving = false;
    toggle.disabled = false;
    if (hadFocus) toggle.focus({ preventScroll: true });
  }
  await refresh();
});

retry.addEventListener("click", async () => {
  retry.disabled = true;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.tabs.sendMessage(tab.id, { type: "soundthing:retry" });
    await refresh();
  } catch { showState({ status: "error", error: "Refresh your YouTube tab, then try again." }); }
  finally { retry.disabled = false; }
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.soundthingPreferences) refresh();
});

refresh();
setInterval(refresh, 1000);
