import { DOA_VISUAL } from "./settings.mjs";

// Backend levels are compressed dB values. Expand visual contrast without
// normalizing a quiet soundtrack into loud-looking activity.
export function directionActivity(levels, settings = DOA_VISUAL) {
  const drive = Object.fromEntries(Object.entries(levels).map(([channel, level]) => [channel,
    (Math.max(0, Math.min(1, (level - settings.noiseFloor) / (1 - settings.noiseFloor)))) ** settings.loudnessExponent]));
  const peak = Math.max(0, ...Object.values(drive));
  return Object.fromEntries(Object.entries(drive).map(([channel, value]) => [channel,
    peak ? value * (1 - settings.directionBias + settings.directionBias * value / peak) : 0]));
}

export function waveDisplacement(delta, span, activity, time, reducedMotion = false, settings = DOA_VISUAL) {
  // Small tangential sway, capped for closely spaced speakers and narrow panels.
  const sway = reducedMotion ? 0 : Math.min(settings.swayPx, span * .08) * Math.sin(time * Math.PI * 2 * settings.swayHz);
  const t = (delta - sway) / span + .5;
  if (t <= 0 || t >= 1 || !activity) return 0;
  const envelope = Math.sin(t * Math.PI) ** 2;
  const profile = .8 + .2 * Math.cos((t - .5) * Math.PI * 4);
  // Outward pulsing and limited sway keep motion close to the sound's direction.
  const pulse = reducedMotion ? 1 : 1 - settings.pulseDepth * (.5 - .5 * Math.cos(time * Math.PI * 2 * settings.pulseHz));
  return settings.maxDisplacementPx * activity * envelope * profile * pulse;
}

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

// Clockwise distance around a rounded border, starting at the top-left tangent.
export function roundedBorder(width, height, radius, distance) {
  const r = Math.min(radius, width / 2, height / 2), a = width - 2 * r, b = height - 2 * r, q = Math.PI * r / 2;
  const perimeter = 2 * (a + b) + 4 * q;
  let d = ((distance % perimeter) + perimeter) % perimeter;
  if (d <= a) return { x: r + d, y: 0, nx: 0, ny: -1 };
  d -= a;
  const arc = (cx, cy, angle) => ({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle), nx: Math.cos(angle), ny: Math.sin(angle) });
  if (d <= q) return arc(width - r, r, d / r - Math.PI / 2);
  d -= q;
  if (d <= b) return { x: width, y: r + d, nx: 1, ny: 0 };
  d -= b;
  if (d <= q) return arc(width - r, height - r, d / r);
  d -= q;
  if (d <= a) return { x: width - r - d, y: height, nx: 0, ny: 1 };
  d -= a;
  if (d <= q) return arc(r, height - r, d / r + Math.PI / 2);
  d -= q;
  if (d <= b) return { x: 0, y: height - r - d, nx: -1, ny: 0 };
  d -= b;
  return arc(r, r, d / r + Math.PI);
}

export function directionAnchor(width, height, degrees, radius = 12) {
  const r = Math.min(radius, width / 2, height / 2);
  const perimeter = 2 * (width + height - 4 * r) + 2 * Math.PI * r;
  const origin = width / 2 - r;
  const desired = (degrees + 360) % 360;
  let low = 0, high = perimeter;
  for (let i = 0; i < 24; i++) {
    const mid = (low + high) / 2, point = roundedBorder(width, height, r, origin + mid);
    const angle = (Math.atan2(point.x - width / 2, -(point.y - height / 2)) * 180 / Math.PI + 360) % 360;
    if (angle < desired) low = mid; else high = mid;
  }
  return { distance: origin + (low + high) / 2, perimeter, radius: r };
}

export class DirectionWaves {
  constructor(text, shadow, document) {
    this.text = text;
    this.width = this.height = 0;
    this.document = document;
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("aria-hidden", "true");
    this.svg.setAttribute("data-direction", "unavailable");
    this.svg.style.cssText = "position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none;z-index:0";
    this.path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    this.path.setAttribute("data-wave", "border");
    this.path.setAttribute("fill", "rgb(12 16 20 / 32%)");
    this.path.setAttribute("stroke", "#c4c4c4");
    this.path.setAttribute("stroke-width", "1");
    this.path.setAttribute("stroke-linejoin", "round");
    this.svg.append(this.path);
    shadow.append(this.svg);
    this.motion = document.defaultView?.matchMedia?.("(prefers-reduced-motion: reduce)");
    const Observer = document.defaultView?.ResizeObserver;
    if (Observer) {
      this.observer = new Observer(() => {
        this.measure();
        // Rebuild geometry before paint, including every step of a size transition.
        if (this.lastTime !== undefined) this.render(this.lastTime, this.lastVisible);
      });
      this.observer.observe(text);
    }
    this.measure();
  }

  measure() {
    const bounds = this.text.getBoundingClientRect();
    if (Math.abs(bounds.width - this.width) < .01 && Math.abs(bounds.height - this.height) < .01) return;
    this.width = bounds.width; this.height = bounds.height;
    this.anchors = null;
    if (this.width && this.height) this.svg.setAttribute("viewBox", `0 0 ${this.width} ${this.height}`);
  }

  setTrack(track) {
    this.track = validateDirection(track);
    this.anchors = null;
    this.svg.setAttribute("data-direction", track.mode);
  }

  render(time, visible) {
    this.lastTime = time; this.lastVisible = visible;
    const levels = visible ? levelsAt(this.track, time) : {};
    const activity = directionActivity(levels);
    this.svg.style.visibility = visible ? "visible" : "hidden";
    if (!this.width || !this.height) this.measure();
    if (!this.width || !this.height) return;
    if (!this.anchors) {
      this.anchors = new Map((this.track?.channels || []).map(channel => [channel,
        directionAnchor(this.width, this.height, channelAngle(this.track.mode, channel))]));
      for (const [channel, anchor] of this.anchors) {
        anchor.span = Math.min(220, Math.max(96, this.height * 1.6));
        for (const [other, neighbor] of this.anchors) {
          if (other === channel) continue;
          const gap = Math.abs(anchor.distance - neighbor.distance);
          anchor.span = Math.min(anchor.span, Math.min(gap, anchor.perimeter - gap) * 1.15);
        }
        anchor.span *= DOA_VISUAL.spanScale;
      }
    }
    const radius = Math.min(12, this.width / 2, this.height / 2);
    const perimeter = 2 * (this.width + this.height - 4 * radius) + 2 * Math.PI * radius;
    const samples = Math.max(160, Math.ceil(perimeter / 3)), points = [];
    // All speakers deform one shared silhouette, including the panel's fill.
    // Silence and unavailable direction leave the same border at rest.
    for (let i = 0; i < samples; i++) {
      const distance = i / samples * perimeter;
      const edge = roundedBorder(this.width, this.height, radius, distance);
      let outward = 0;
      for (const [channel, anchor] of this.anchors) {
        const strength = activity[channel] || 0;
        if (!strength) continue;
        const delta = ((distance - anchor.distance + perimeter * 1.5) % perimeter) - perimeter / 2;
        outward += waveDisplacement(delta, anchor.span, strength, time, this.motion?.matches);
      }
      outward = Math.min(DOA_VISUAL.maxDisplacementPx, outward);
      points.push(`${i ? "L" : "M"}${(edge.x + edge.nx * outward).toFixed(2)},${(edge.y + edge.ny * outward).toFixed(2)}`);
    }
    const geometry = points.join(" ") + " Z";
    if (this.path.getAttribute("d") !== geometry) this.path.setAttribute("d", geometry);
    const strength = Math.max(0, ...Object.values(activity));
    this.path.setAttribute("stroke-opacity", String(.18 + .28 * strength));
  }

  dispose() { this.observer?.disconnect(); this.svg.remove(); }
}
