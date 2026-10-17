import { PLAYBACK_POLICY, POLL_INTERVAL_MS, watchVideoId } from "./settings.mjs";
import { activeCaption, parseVtt } from "./vtt.mjs";
import { DirectionWaves } from "./direction.mjs";

export class CaptionController {
  constructor({ window, document, send, playbackPolicy = PLAYBACK_POLICY }) {
    if (!["keep-playing", "pause-until-ready"].includes(playbackPolicy)) throw new Error("Unknown playback policy.");
    Object.assign(this, { window, document, send, playbackPolicy });
    this.preferences = { enabled: false, server: "" };
    this.session = null;
    this.state = { status: "disabled" };
  }

  configure(preferences) { this.preferences = preferences; this.refresh(); }

  refresh() {
    const id = watchVideoId(this.window.location.href);
    const player = this.document.querySelector("#movie_player");
    const video = player?.querySelector("video");
    const old = this.session;
    if (!this.preferences.enabled || !id || !video) {
      this.disposeSession();
      this.state = { status: !this.preferences.enabled ? "disabled" : "unsupported", youtubeId: id };
      return;
    }
    if (old && old.id === id && old.video === video && old.player === player && old.server === this.preferences.server && old.overlay.isConnected) { this.render(); return; }
    this.disposeSession();
    const overlay = this.document.createElement("div");
    overlay.id = "soundthing-caption-overlay";
    overlay.setAttribute("aria-label", "Soundthing non-speech captions");
    overlay.style.cssText = "all:initial;position:absolute;right:16px;bottom:64px;max-width:min(55%,480px);z-index:45;pointer-events:none;";
    const shadow = overlay.attachShadow({ mode: "open" });
    const style = this.document.createElement("style");
    style.textContent = ":host([hidden]){display:none!important}div{box-sizing:border-box;color:white;background:rgba(15,12,22,.88);border-radius:6px;padding:9px 13px;font:500 clamp(14px,1.5vw,22px)/1.45 Arial,sans-serif;text-align:right;white-space:pre-line;overflow-wrap:anywhere;box-shadow:0 2px 8px #0004}div[data-status]{font-size:12px;color:#eee3ff;}";
    const text = this.document.createElement("div");
    shadow.append(style, text);
    player.append(overlay);
    const session = { id, video, player, overlay, text, server: this.preferences.server, cues: [], status: "queued", demo: false, error: "", ownedPause: false, submitted: false, inFlight: false, retryAttempts: 0 };
    session.waves = new DirectionWaves(text, shadow, this.document);
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
    if (current && !ad) text = session.status === "done" ? activeCaption(session.cues, session.video.currentTime) : session.status === "error" ? "Soundthing: open the popup to retry" : `Soundthing · ${session.status}…`;
    if (session.text.textContent !== text) session.text.textContent = text;
    session.text.toggleAttribute("data-status", session.status !== "done");
    session.overlay.hidden = !text;
    session.waves.render(session.video.currentTime, !!text && session.status === "done" && current && !ad);
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
        this.pollDirection(session);
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

  retry() { this.disposeSession(); this.refresh(); }

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
    session.player.removeEventListener("pointerdown", session.intervene, true);
    session.player.removeEventListener("keydown", session.intervene, true);
    session.video.removeEventListener("play", session.onPlay);
    session.waves.dispose();
    session.overlay.remove();
    this.releasePause(session);
  }
}
