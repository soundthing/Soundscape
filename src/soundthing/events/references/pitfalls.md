# Pitfalls

Traps that aren't obvious from the code. Add one when something bites; remove it when the
cause is gone.

- **Sample rates differ per model.** Audio is 16 kHz; PANNs CNN14 wants 32 kHz and CLAP
  48 kHz. Feeding the wrong rate gives plausible-looking but wrong scores, not an error.
- **AudioSet names must match exactly.** `vocabulary.yaml` names are matched against the
  tagger's label list character for character. Mismatches fail at model load. Check with
  `soundthing check-config --audioset <csv>`.
- **CLAP-only classes are silent without CLAP.** Several classes have no AudioSet labels
  (e.g. kettle whistle, turn signal, seatbelt), so they never fire until CLAP is in.
- **CLAP scores aren't probabilities.** Raw cosine similarities sit in a narrow band and
  must be calibrated before they're compared with tagger probabilities or thresholds.
- **Onsets can come out early.** A 1 s window starts scoring when a sound enters its
  trailing edge, so window-centre timing can lead the real onset by up to ~0.5 s — the
  whole eval tolerance. Measure the offset on labeled clips before correcting it.
- **Over-smoothing kills impulsive sounds.** Gunshots and door slams span only a few
  windows; a wide median filter erases them.
- **Short events stay on screen longer than their span.** The caption writer extends cues
  to a minimum duration, which matters when capping captions on screen.
- **Heavy imports break the stub path.** torch / transformers at module top level fail
  wherever the `events` extra isn't installed, including CI.
- **CLAP calibration center/scale are untuned placeholders.** Default values
  (center: 0.2, scale: 20.0) in pipeline config are placeholders; until tuned on
  labeled validation clips, CLAP zero-shot scores may under- or over-trigger
  relative to AudioSet tagger probabilities.
- **Silero VAD expects 16 kHz mono float32.** Multi-channel audio or other sample rates must
  be averaged to mono and resampled; feeding raw multi-channel audio leads to silent failures
  or shape mismatches.
- **Coarse group ids are not in `cfg.classes`.** Calling `cfg.sound_class()` or
  `cfg.coarse_of()` with a coarse group id raises `KeyError`. Coarse groups are neither
  safety classes nor sustained.
- **Clip median for loudness must exclude digital silence.** Pure-silence frames
  skew the whole-clip median downwards, falsely making baseline-level sounds appear
  loud. Filtering out frames below 1e-6 RMS avoids this.
- **Windowed events can overhang clip duration.** Sliding windows extending past audio
  boundaries or at the clip edge can produce detections ending after `inputs.duration`
  or starting at/after it. Dropping start >= duration and clipping ends to duration
  ensures all events satisfy contract validation.
- **AudioSet scores flatter the tagger.** PANNs was trained on AudioSet (a disjoint split, but the same sources and label style), so AudioSet F0.5 likely overstates performance on real videos.
- **FSD50K labels are hierarchy-propagated.** Parent ontology labels (e.g. `Animal`,
  `Domestic_animals_and_pets`) appear alongside specific leaf labels on every clip.
  Filtering for clips that reach exactly one distinct mapped class prevents multi-source
  contamination while allowing clips with multiple ontology tags that map to the same
  target class. So never map a parent label (`Alarm`, `Door`, `Glass`) whose children
  map to other classes: every siren/knock clip would turn two-class and be dropped.
- **VAD hysteresis state across chunk boundaries.** Silero VAD per-chunk probabilities
  require maintaining hysteresis state (on at threshold, off below threshold - 0.15)
  and resetting model states between files.
- **Overlapping events under priority filter.** Overlapping events of different priorities
  (e.g. safety vs non-safety) will have the non-safety event suppressed by `priority_filter`.
  When synthesizing multi-event test clips, space activations across non-overlapping time
  windows so both events are preserved.
