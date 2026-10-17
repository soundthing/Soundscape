function timestamp(value) {
  const match = /^(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})$/.exec(value);
  if (!match || +match[2] > 59 || +match[3] > 59) throw new Error("Invalid VTT timestamp.");
  return +(match[1] || 0) * 3600 + +match[2] * 60 + +match[3] + +match[4] / 1000;
}

export function parseVtt(value) {
  const text = value.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
  if (!/^WEBVTT(?:[ \t].*)?\n/.test(text)) throw new Error("Invalid VTT header.");
  const cues = [];
  for (const block of text.split(/\n[ \t]*\n/).slice(1)) {
    const lines = block.trim().split("\n");
    if (!block.trim() || /^(NOTE|STYLE|REGION)(?:\s|$)/.test(lines[0])) continue;
    const index = lines[0].includes("-->") ? 0 : 1;
    const match = /^(\S+)\s+-->\s+(\S+)(?:\s+.*)?$/.exec(lines[index] || "");
    if (!match) throw new Error("Invalid VTT cue timing.");
    const start = timestamp(match[1]), end = timestamp(match[2]);
    if (end <= start) throw new Error("Invalid VTT cue duration.");
    const caption = lines.slice(index + 1).join("\n");
    if (caption) cues.push({ start, end, text: caption });
  }
  return cues.sort((a, b) => a.start - b.start);
}

export function activeCaption(cues, time) {
  const lines = [];
  for (const cue of cues) {
    if (cue.start > time) break;
    if (time < cue.end) lines.push(cue.text);
  }
  return lines.join("\n");
}
