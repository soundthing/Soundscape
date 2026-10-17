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
- **Small eval set overfits fast.** 10–20 clips: tune on part, check on the rest, and
  prefer raising a threshold to adding a special case.
- **Heavy imports break the stub path.** torch / transformers at module top level fail
  wherever the `events` extra isn't installed, including CI.
- **CLAP calibration center/scale are untuned placeholders.** Default values
  (center: 0.2, scale: 20.0) in pipeline config are placeholders; until tuned on
  labeled validation clips, CLAP zero-shot scores may under- or over-trigger
  relative to AudioSet tagger probabilities.

