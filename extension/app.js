import {
  ACTIVE_STATUSES, DEFAULT_SERVER, apiUrl, downloadName, formatBytes,
  normalizeServer, studioUrl, uploadFilename, validateVideo,
} from "./core.mjs";
import { checkConnection, getStatus, uploadVideo } from "./api.mjs";

const $ = (id) => document.getElementById(id);
const STORAGE_KEY = "soundthingWorkspace";
const importButtons = [$("hero-import"), $("library-import"), $("browse-files")];
const downloadIds = new Set();
let server = DEFAULT_SERVER;
let jobs = [];
let filter = "all";
let uploading = false;
let toastTimer;
let pollTimer;
let previewJob;
let previewUrl;
let previewGeneration = 0;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 7000);
}

async function save() {
  try {
    await chrome.storage.local.set({ [STORAGE_KEY]: { server, jobs } });
  } catch {
    toast("Chrome couldn't save your library. Keep this tab open and download your files before closing it.");
  }
}

function button(text, className, handler) {
  const node = element("button", className, text);
  node.type = "button";
  node.addEventListener("click", handler);
  return node;
}

function render() {
  $("video-count").textContent = jobs.length;
  const visible = jobs.filter((job) => filter === "all" || (filter === "processing" ? ACTIVE_STATUSES.has(job.status) : job.status === filter));
  $("video-grid").replaceChildren(...visible.map(renderCard));
  $("empty-state").hidden = visible.length > 0;
  $("empty-state").querySelector("h3").textContent = jobs.length ? "No videos in this view yet." : "A home for your next video.";
  $("empty-state").querySelector("p").textContent = jobs.length ? "Try another filter or import a new clip." : "Import a clip to generate sound captions, compare results, and download your files.";
}

function renderCard(job) {
  const labels = { queued: "Queued", processing: "Processing", done: "Ready", error: "Needs attention" };
  const card = element("article", "video-card");
  const visual = element("div", "video-visual");
  const icon = element("span", "", "▻");
  icon.setAttribute("aria-hidden", "true");
  visual.append(icon, element("span", `status-chip ${job.status}`, labels[job.status]));
  const body = element("div", "card-body");
  const title = element("h3", "video-name", job.name);
  title.title = job.name;
  body.append(title, element("p", "video-meta", `${formatBytes(job.size)} · ${new Date(job.createdAt).toLocaleDateString(undefined, { month: "short", day: "numeric" })}`));
  if (job.status === "done") {
    const actions = element("div", "card-actions");
    actions.append(button("▷ Preview", "button secondary", () => openPreview(job)), button("↓ Original video", "button primary", (event) => download(job, "video", event.currentTarget)));
    const captions = element("div", "caption-actions");
    captions.append(button("↓ Conditioned VTT", "text-button", (event) => download(job, "conditioned", event.currentTarget)), button("↓ Baseline VTT", "text-button", (event) => download(job, "baseline", event.currentTarget)));
    body.append(actions, captions);
  } else if (job.status === "error") {
    body.append(element("p", "card-message", job.error || "Processing failed. Check the server, then import the video again."));
  } else {
    body.append(element("p", "card-message", job.connectionError || "Preparing both caption tracks. You can leave this tab and come back."));
  }
  if (job.connectionError) {
    body.append(button("Check again", "text-button", () => { clearTimeout(pollTimer); poll(); }));
  }
  body.append(button("Remove from library", "text-button remove-button", async () => {
    jobs = jobs.filter((item) => item.id !== job.id);
    await save();
    render();
    toast("Removed from this library. Files on your server are unchanged.");
  }));
  card.append(visual, body);
  return card;
}

async function connect(value, persist = true) {
  const badge = $("connection-badge");
  const message = $("connection-message");
  const submit = $("connection-form").querySelector("button");
  submit.disabled = true;
  try {
    const candidate = normalizeServer(value);
    await checkConnection(candidate);
    server = candidate;
    $("server-url").value = server;
    badge.classList.remove("offline");
    badge.replaceChildren(element("span"), document.createTextNode("Local server connected"));
    message.textContent = "Connected. Your videos will be processed on this machine.";
    if (persist) await save();
    return true;
  } catch (error) {
    badge.classList.add("offline");
    badge.replaceChildren(element("span"), document.createTextNode("Server disconnected"));
    message.textContent = error.message;
    return false;
  } finally {
    submit.disabled = false;
  }
}

async function importFiles(files) {
  if (uploading || !files.length) return;
  uploading = true;
  importButtons.forEach((item) => { item.disabled = true; });
  $("connection-form").querySelector("button").disabled = true;
  const uploadServer = server;
  const progress = $("upload-progress");
  progress.hidden = false;
  try {
    await checkConnection(uploadServer);
    for (const [index, file] of [...files].entries()) {
      try {
        validateVideo(file);
        progress.textContent = `Uploading ${index + 1} of ${files.length}: ${file.name}. Keep this tab open until the upload finishes.`;
        const id = crypto.randomUUID();
        const videoId = await uploadVideo(uploadServer, file, uploadFilename(file, id));
        jobs.unshift({ id, videoId, name: file.name, size: file.size, createdAt: Date.now(), server: uploadServer, status: "queued", error: "" });
        await save();
        render();
      } catch (error) {
        toast(`${file.name}: ${error.message}`);
      }
    }
  } catch (error) {
    toast(error.message);
    await connect(uploadServer, false);
  } finally {
    uploading = false;
    importButtons.forEach((item) => { item.disabled = false; });
    $("connection-form").querySelector("button").disabled = false;
    progress.hidden = true;
    $("file-input").value = "";
  }
}

let polling = false;
async function poll() {
  if (polling) return;
  polling = true;
  clearTimeout(pollTimer);
  try {
    const pending = jobs.filter((job) => ACTIVE_STATUSES.has(job.status));
    const updates = await Promise.all(pending.map(async (job) => {
      try { return { id: job.id, ...await getStatus(job), connectionError: "" }; }
      catch (error) { return { id: job.id, connectionError: error.message }; }
    }));
    let changed = false;
    for (const update of updates) {
      const job = jobs.find((item) => item.id === update.id);
      if (job && Object.entries(update).some(([key, value]) => job[key] !== value)) {
        Object.assign(job, update);
        changed = true;
      }
    }
    if (changed) {
      await save();
      render();
    }
  } finally {
    polling = false;
    pollTimer = setTimeout(poll, 3000);
  }
}

async function download(job, mode, trigger) {
  trigger.disabled = true;
  try {
    const latest = await getStatus(job);
    if (latest.status !== "done") throw new Error(latest.error || "This video isn't ready on the server yet.");
    const resource = mode === "video" ? "/video" : `/captions/${mode}.vtt`;
    const id = await chrome.downloads.download({ url: apiUrl(job.server, job.videoId, resource), filename: downloadName(job.name, mode), saveAs: true, conflictAction: "uniquify" });
    downloadIds.add(id);
    toast("Download started. Chrome will ask where to save the file.");
  } catch (error) {
    toast(error.message);
  } finally {
    trigger.disabled = false;
  }
}

chrome.downloads.onChanged.addListener((delta) => {
  if (!downloadIds.has(delta.id)) return;
  if (delta.state?.current === "complete") {
    downloadIds.delete(delta.id);
    toast("Your file has been downloaded.");
  } else if (delta.state?.current === "interrupted") {
    downloadIds.delete(delta.id);
    toast(`Download interrupted${delta.error?.current ? `: ${delta.error.current}` : ""}. Try downloading again.`);
  }
});

function clearPreviewTrack() {
  $("preview-video").querySelectorAll("track").forEach((track) => track.remove());
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}

async function setCaptionTrack() {
  const generation = ++previewGeneration;
  clearPreviewTrack();
  const job = previewJob;
  if (!job) return;
  $("preview-message").textContent = "Loading captions…";
  const mode = $("caption-mode").value;
  try {
    // A same-origin blob avoids cross-origin <track> restrictions. The original
    // video is streamed from localhost without loading the whole file into RAM.
    const response = await fetch(apiUrl(job.server, job.videoId, `/captions/${mode}.vtt`), { credentials: "omit", redirect: "error", signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error("The caption track isn't available. Check your server and try again.");
    const text = await response.text();
    if (!text.trimStart().startsWith("WEBVTT")) throw new Error("The server returned an invalid caption track.");
    if (generation !== previewGeneration) return;
    previewUrl = URL.createObjectURL(new Blob([text], { type: "text/vtt" }));
    const track = element("track");
    Object.assign(track, { kind: "captions", label: mode, srclang: "en", default: true, src: previewUrl });
    $("preview-video").append(track);
    track.track.mode = "showing";
    $("preview-message").textContent = mode === "conditioned" ? "Environment-conditioned sound captions. Results follow your server's configured model backends." : "Baseline captions use audio without environment conditioning.";
  } catch (error) {
    if (generation === previewGeneration) $("preview-message").textContent = error.message;
  }
}

function openPreview(job) {
  previewJob = job;
  $("preview-title").textContent = job.name;
  $("caption-mode").value = "conditioned";
  $("preview-video").src = apiUrl(job.server, job.videoId, "/video");
  $("preview-dialog").showModal();
  setCaptionTrack();
}

$("preview-dialog").addEventListener("close", () => {
  previewGeneration++;
  previewJob = null;
  $("preview-video").pause();
  $("preview-video").removeAttribute("src");
  clearPreviewTrack();
  $("preview-video").load();
});
$("preview-video").addEventListener("error", () => { if (previewJob) $("preview-message").textContent = "This video couldn't be played. Download the original, or check that your server is still running. Some MOV codecs aren't supported in Chrome."; });
$("close-preview").addEventListener("click", () => $("preview-dialog").close());
$("caption-mode").addEventListener("change", setCaptionTrack);

importButtons.forEach((item) => item.addEventListener("click", () => $("file-input").click()));
$("file-input").addEventListener("change", (event) => importFiles(event.target.files));
const dropZone = $("drop-zone");
for (const type of ["dragenter", "dragover"]) {
  dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
}
for (const type of ["dragleave", "drop"]) {
  dropZone.addEventListener(type, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); });
}
dropZone.addEventListener("drop", (event) => importFiles(event.dataTransfer.files));
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

document.querySelectorAll("[data-filter]").forEach((item) => item.addEventListener("click", () => {
  filter = item.dataset.filter;
  document.querySelectorAll("[data-filter]").forEach((candidate) => {
    candidate.classList.toggle("active", candidate === item);
    candidate.setAttribute("aria-pressed", String(candidate === item));
  });
  render();
}));

$("youtube-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const url = studioUrl($("youtube-url").value);
    await chrome.tabs.create({ url });
    $("youtube-message").textContent = "Studio is open. Sign in to the channel that owns this video, then use its menu → Download. Import the downloaded MP4 here.";
  } catch (error) {
    $("youtube-message").textContent = error.message;
  }
});

$("connection-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!uploading) connect($("server-url").value);
});

window.addEventListener("beforeunload", (event) => {
  if (uploading) { event.preventDefault(); event.returnValue = ""; }
});

async function init() {
  try {
    const stored = (await chrome.storage.local.get(STORAGE_KEY))[STORAGE_KEY];
    if (stored) {
      server = normalizeServer(stored.server || DEFAULT_SERVER);
      jobs = (Array.isArray(stored.jobs) ? stored.jobs : []).filter((job) => {
        try {
          apiUrl(job.server, job.videoId);
          return typeof job.id === "string" && typeof job.name === "string" && Number.isFinite(job.size) && Number.isFinite(job.createdAt) && ["queued", "processing", "done", "error"].includes(job.status);
        } catch { return false; }
      });
    }
  } catch { toast("Saved settings couldn't be loaded. Using the default local server."); }
  $("server-url").value = server;
  render();
  connect(server, false);
  poll();
}

init();
