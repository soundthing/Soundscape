// Consumer validation for the documented DirectionTrack wire format.
export function validateDirection(track) {
  const layouts = { stereo: ["FL", "FR"], surround51: ["FL", "FR", "FC", "SL", "SR"], unavailable: [] };
  const channels = track && layouts[track.mode];
  if (!channels || !Number.isFinite(track.duration) || track.duration < 0 ||
      JSON.stringify(track.channels) !== JSON.stringify(channels) || !Array.isArray(track.frames)) throw Error("Invalid direction track.");
  if (track.mode === "unavailable") {
    if (track.frames.length) throw Error("Invalid unavailable direction track.");
    return track;
  }
  if (!track.frames.length || track.frames[0].time !== 0) throw Error("Direction track must start at zero.");
  let previous = -1;
  for (const frame of track.frames) {
    if (!Number.isFinite(frame.time) || frame.time <= previous || frame.time >= track.duration ||
        !frame.levels || Object.keys(frame.levels).length !== channels.length ||
        !channels.every(channel => Number.isFinite(frame.levels[channel]) && frame.levels[channel] >= 0 && frame.levels[channel] <= 1)) throw Error("Invalid direction frame.");
    previous = frame.time;
  }
  return track;
}

export function levelsAt(track, time) {
  if (!track || track.mode === "unavailable" || !Number.isFinite(time) || time < 0 || time >= track.duration) return {};
  let lo = 0, hi = track.frames.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (track.frames[mid].time <= time) lo = mid + 1; else hi = mid; }
  const a = track.frames[Math.max(0, lo - 1)], b = track.frames[lo];
  const ratio = b ? (time - a.time) / (b.time - a.time) : 0;
  return Object.fromEntries(track.channels.map(channel => [channel, a.levels[channel] + ((b?.levels[channel] ?? a.levels[channel]) - a.levels[channel]) * ratio]));
}

export function channelAngle(mode, channel) {
  return mode === "stereo" ? { FL: -90, FR: 90 }[channel] : { FL: -30, FR: 30, FC: 0, SL: -110, SR: 110 }[channel];
}

// Angles are horizontal: forward=top, rear=bottom, never elevation.
export function borderPoint(width, height, degrees) {
  const angle = degrees * Math.PI / 180, dx = Math.sin(angle), dy = -Math.cos(angle);
  const distance = Math.min(width / 2 / Math.max(Math.abs(dx), 1e-12), height / 2 / Math.max(Math.abs(dy), 1e-12));
  return { x: width / 2 + dx * distance, y: height / 2 + dy * distance, dx, dy };
}

export class DirectionWaves {
  constructor(text, shadow, document) {
    this.text = text;
    this.width = this.height = 0;
    this.paths = new Map();
    this.document = document;
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("aria-hidden", "true");
    this.svg.setAttribute("data-direction", "unavailable");
    this.svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none";
    shadow.append(this.svg);
    this.motion = document.defaultView?.matchMedia?.("(prefers-reduced-motion: reduce)");
    const Observer = document.defaultView?.ResizeObserver;
    if (Observer) {
      this.observer = new Observer(() => this.measure());
      this.observer.observe(text);
    }
    this.measure();
  }

  measure() {
    const bounds = this.text.getBoundingClientRect();
    this.width = bounds.width; this.height = bounds.height;
  }

  setTrack(track) {
    this.track = validateDirection(track);
    this.svg.replaceChildren();
    this.paths.clear();
    this.svg.setAttribute("data-direction", track.mode);
    for (const channel of track.channels) {
      const path = this.document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("data-channel", channel);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#7dd3fc");
      path.setAttribute("stroke-width", "2.25");
      path.setAttribute("stroke-linecap", "round");
      this.paths.set(channel, path); this.svg.append(path);
    }
  }

  render(time, visible) {
    const levels = visible ? levelsAt(this.track, time) : {};
    this.svg.style.visibility = visible ? "visible" : "hidden";
    if (!this.width || !this.height) this.measure();
    for (const [channel, path] of this.paths) {
      const strength = levels[channel] || 0;
      path.style.display = strength > .04 ? "" : "none";
      if (strength <= .04) continue;
      const { x, y, dx, dy } = borderPoint(this.width, this.height, channelAngle(this.track.mode, channel));
      const vertical = Math.abs(x) < .01 || Math.abs(x - this.width) < .01;
      const half = Math.min(24, (vertical ? this.height : this.width) * .35);
      const points = [];
      for (let i = 0; i <= 24; i++) {
        const along = (i / 24 * 2 - 1) * half;
        const wave = this.motion?.matches ? 0 : Math.sin(i / 24 * Math.PI * 4 - time * 8) * Math.sin(i / 24 * Math.PI) * strength * 9;
        const px = x + (vertical ? 0 : along) + dx * wave;
        const py = y + (vertical ? along : 0) + dy * wave;
        points.push(`${i ? "L" : "M"}${px.toFixed(2)},${py.toFixed(2)}`);
      }
      path.setAttribute("d", points.join(" "));
      path.setAttribute("opacity", String(.25 + .75 * strength));
    }
  }

  dispose() { this.observer?.disconnect(); this.svg.remove(); }
}
