import { PLAYBACK_POLICY, POLL_INTERVAL_MS, watchVideoId } from "./settings.mjs";
import { activeCaption, parseVtt } from "./vtt.mjs";
import { DirectionWaves } from "./direction.mjs";

// FontFace registers the bundled font with the document, including shadow trees.
const fontLoads = new WeakMap();
function loadCaptionFont(document) {
  if (!document.fonts || !document.defaultView?.FontFace || fontLoads.has(document)) return;
  const source = new URL("./assets/Monocraft.ttf", import.meta.url).href;
  const face = new document.defaultView.FontFace("Soundthing Monocraft", `url("${source}")`);
  document.fonts.add(face);
  fontLoads.set(document, face.load().catch(error => {
    document.defaultView.console.warn("[Soundthing] Caption font unavailable:", error);
  }));
}

export class CaptionController {
  constructor({ window, document, send, playbackPolicy = PLAYBACK_POLICY }) {
    if (!["keep-playing", "pause-until-ready"].includes(playbackPolicy)) throw new Error("Unknown playback policy.");
    Object.assign(this, { window, document, send, playbackPolicy });
    this.preferences = { enabled: false, server: "" };
    this.session = null;
    this.clearedVideoId = null;
    this.state = { status: "disabled" };
  }

  configure(preferences) {
    if (preferences.enabled !== this.preferences.enabled || preferences.server !== this.preferences.server) this.clearedVideoId = null;
    this.preferences = preferences;
    this.refresh();
  }

  refresh() {
    const id = watchVideoId(this.window.location.href);
    if (this.clearedVideoId !== id) this.clearedVideoId = null;
    const player = this.document.querySelector("#movie_player");
    const video = player?.querySelector("video");
    const old = this.session;
    if (!this.preferences.enabled || !id || !video) {
      this.disposeSession();
      this.state = { status: !this.preferences.enabled ? "disabled" : "unsupported", youtubeId: id };
      return;
    }
    if (this.clearedVideoId === id) {
      this.disposeSession();
      this.state = { status: "cache-cleared", youtubeId: id };
      return;
    }
    this.clearedVideoId = null;
    if (old && old.id === id && old.video === video && old.player === player && old.server === this.preferences.server && old.overlay.isConnected) { this.render(); return; }
    this.disposeSession();
    const overlay = this.document.createElement("div");
    overlay.id = "soundthing-caption-overlay";
    overlay.setAttribute("aria-label", "Soundthing captions and audio direction");
    overlay.style.cssText = "all:initial;position:absolute;right:48px;bottom:64px;width:max-content;z-index:45;pointer-events:none;";
    const shadow = overlay.attachShadow({ mode: "open" });
    const style = this.document.createElement("style");
    style.textContent = `
      :host([hidden]){display:none!important}
      *{box-sizing:border-box}
      section{position:relative;width:192px;height:64px;overflow:hidden;border:1px solid transparent;z-index:1;
        border-radius:12px;background:transparent;color:#e0e0e0;
        transition:width .24s cubic-bezier(.2,.8,.2,1),height .24s cubic-bezier(.2,.8,.2,1);
        font-family:"Soundthing Monocraft",monospace}
      article{position:absolute;left:0;top:0;width:max-content;min-width:190px;min-height:62px;
        padding:14px 16px;display:flex;align-items:center}
      .caption{opacity:.88;min-width:0;width:100%;font-size:clamp(14px,1.5vw,20px);font-weight:400;line-height:1.5;
        font-variant-ligatures:none;text-align:left;white-space:pre-line;overflow-wrap:anywhere;
        text-shadow:2px 2px 0 #383838}
      .caption[hidden]{display:none}
      @media(prefers-reduced-motion:reduce){section{transition:none}}
    `;
    const panel = this.document.createElement("section");
    const content = this.document.createElement("article");
    const text = this.document.createElement("div");
    text.className = "caption";
    content.append(text);
    loadCaptionFont(this.document);
    panel.append(content);
    shadow.append(style, panel);
    player.append(overlay);
    const session = { id, video, player, overlay, panel, content, text, server: this.preferences.server, cues: [], status: "queued", demo: false, error: "", ownedPause: false, submitted: false, inFlight: false, retryAttempts: 0 };
    session.resizePanel = () => {
      if (overlay.hidden || !content.isConnected) return;
      const playerWidth = player.clientWidth || this.document.defaultView?.innerWidth || 640;
      const maximum = Math.max(1, Math.min(480, playerWidth * .65, playerWidth - 64));
      content.style.minWidth = `${Math.max(0, Math.min(192, maximum) - 2)}px`;
      content.style.maxWidth = `${Math.max(0, maximum - 2)}px`;
      const bounds = content.getBoundingClientRect();
      panel.style.width = `${Math.ceil(bounds.width) + 2}px`;
      panel.style.height = `${Math.ceil(bounds.height) + 2}px`;
    };
    const Observer = this.document.defaultView?.ResizeObserver;
    if (Observer) {
      session.sizeObserver = new Observer(session.resizePanel);
      session.sizeObserver.observe(content);
      session.sizeObserver.observe(player);
    }
    session.waves = new DirectionWaves(panel, shadow, this.document);
    this.session = session;
    session.intervene = (event) => { if (event.isTrusted) session.ownedPause = false; };
    session.onPlay = () => { session.ownedPause = false; };
    player.addEventListener("pointerdown", session.intervene, true);
    player.addEventListener("keydown", session.intervene, true);
    video.addEventListener("play", session.onPlay);
    const frame = () => {
      if (this.session !== session) return;
      this.render();
      session.frame = this.window.requestAnimationFrame(frame);
    };
    session.frame = this.window.requestAnimationFrame(frame);
    this.render();
    this.poll(session);
    this.pollDirection(session);
  }

  isCurrent(session) { return this.session === session && this.preferences.enabled && session.id === watchVideoId(this.window.location.href) && session.server === this.preferences.server; }
  isAd(session) { return session.player.classList.contains("ad-showing") || session.player.classList.contains("ad-interrupting"); }

  releasePause(session) {
    const owned = session.ownedPause;
    session.ownedPause = false;
    if (owned && session.id === watchVideoId(this.window.location.href) && session.video.isConnected && session.video.paused && !session.video.ended) Promise.resolve(session.video.play()).catch(() => {});
  }

  render() {
    const session = this.session;
    if (!session) return;
    const current = this.isCurrent(session), ad = this.isAd(session);
    if (current && !ad && this.playbackPolicy === "pause-until-ready" && !["done", "error", "reconnecting"].includes(session.status) && !session.video.paused && !session.pauseAttempted) {
      session.pauseAttempted = true;
      session.ownedPause = true;
      session.video.pause();
    }
    let text = "";
    if (current && !ad && session.status === "done") text = activeCaption(session.cues, session.video.currentTime);
    let resized = false;
    if (session.text.textContent !== text) { session.text.textContent = text; resized = true; }
    session.text.hidden = !text;
    session.panel.toggleAttribute("data-idle", !text);
    session.text.toggleAttribute("data-status", session.status !== "done");
    session.overlay.hidden = !current || ad;
    if (resized || !session.sized) { session.resizePanel(); session.sized = true; }
    session.waves.render(session.video.currentTime, current && !ad);
    this.state = { status: session.status, youtubeId: session.id, error: session.error, demo: session.demo, playbackPolicy: this.playbackPolicy, directionMode: session.waves.track?.mode || "pending" };
  }

  async request(session, action) {
    const result = await this.send({ type: "soundthing:api", action, youtubeId: session.id });
    if (!result?.ok) {
      const error = new Error(result?.error || "Couldn't reach Soundthing. Check the local server.");
      error.retryable = result?.retryable === true;
      error.statusCode = result?.statusCode || 0;
      throw error;
    }
    return result;
  }

  async poll(session = this.session) {
    if (!session || !this.isCurrent(session) || session.inFlight || ["done", "error"].includes(session.status)) return;
    session.inFlight = true;
    try {
      if (!session.submitted) { await this.request(session, "ensure"); session.submitted = true; }
      if (!this.isCurrent(session)) return;
      const status = await this.request(session, "status");
      if (!this.isCurrent(session)) return;
      Object.assign(session, status);
      if (status.status === "done") {
        const { vtt } = await this.request(session, "captions");
        if (!this.isCurrent(session)) return;
        session.cues = parseVtt(vtt);
        this.logCompletion(session, vtt);
        this.releasePause(session);
      } else if (status.status === "error") this.releasePause(session);
      session.retryAttempts = 0;
    } catch (error) {
      if (!this.isCurrent(session)) return;
      // A restarted backend may have lost an unfinished job. Re-submit its ID;
      // completed files are reused by the server, and active work is deduplicated.
      if (error.statusCode === 404) session.submitted = false;
      const recoverable = error.retryable === true || error.statusCode === 404;
      session.status = recoverable ? "reconnecting" : "error";
      if (recoverable) session.retryAttempts++;
      session.error = error.message;
      this.releasePause(session);
    } finally {
      session.inFlight = false;
      if (this.isCurrent(session)) {
        this.render();
        if (!["done", "error"].includes(session.status)) {
          const delay = session.retryAttempts ? Math.min(POLL_INTERVAL_MS * 2 ** Math.min(session.retryAttempts - 1, 4), 15000) : POLL_INTERVAL_MS;
          session.timer = this.window.setTimeout(() => this.poll(session), delay);
        }
      }
    }
  }

  invalidateCache(youtubeId) {
    if (watchVideoId(this.window.location.href) !== youtubeId) return;
    this.clearedVideoId = youtubeId;
    this.disposeSession();
    this.state = { status: "cache-cleared", youtubeId };
  }

  retry() { this.clearedVideoId = null; this.disposeSession(); this.refresh(); }

  async logCompletion(session, vtt) {
    if (!this.isCurrent(session) || session.completionLogged) return;
    session.completionLogged = true;
    let scenes, sceneError;
    try { ({ scenes } = await this.request(session, "scenes")); }
    catch (error) { sceneError = error.message; }
    if (!this.isCurrent(session)) return;
    // Log in the YouTube tab's console, independently of caption/direction readiness.
    const logger = this.window.console || this.document.defaultView?.console;
    logger?.log(`[Soundthing] Video processing complete\n${JSON.stringify({
      youtubeId: session.id, demo: session.demo,
      vttUrl: `${session.server}/api/videos/youtube_${session.id}/captions/conditioned.vtt`,
      scenesUrl: `${session.server}/api/videos/youtube_${session.id}/env_segments`,
    }, null, 2)}`);
    logger?.log(`[Soundthing] Conditioned VTT:\n${vtt}`);
    if (sceneError) logger?.log(`[Soundthing] Scenes unavailable: ${sceneError}`);
    else logger?.log(`[Soundthing] Scenes (environment segments):\n${JSON.stringify(scenes, null, 2)}`);
  }

  async pollDirection(session) {
    if (!this.isCurrent(session) || session.directionInFlight) return;
    session.directionInFlight = true;
    try {
      const { direction } = await this.request(session, "direction");
      if (!this.isCurrent(session)) return;
      session.waves.setTrack(direction);
      this.render();
    } catch {
      // Direction readiness never changes caption status or restarts its job.
      if (this.isCurrent(session)) {
        session.directionAttempts = (session.directionAttempts || 0) + 1;
        session.directionTimer = this.window.setTimeout(() => this.pollDirection(session), Math.min(1500 * 2 ** Math.min(session.directionAttempts - 1, 5), 30000));
      }
    } finally { session.directionInFlight = false; }
  }

  disposeSession() {
    const session = this.session;
    if (!session) return;
    this.session = null;
    this.window.clearTimeout(session.timer);
    this.window.clearTimeout(session.directionTimer);
    this.window.cancelAnimationFrame(session.frame);
    session.sizeObserver?.disconnect();
    session.player.removeEventListener("pointerdown", session.intervene, true);
    session.player.removeEventListener("keydown", session.intervene, true);
    session.video.removeEventListener("play", session.onPlay);
    session.waves.dispose();
    session.overlay.remove();
    this.releasePause(session);
  }
}
