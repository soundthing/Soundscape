# Pitfalls

Traps that aren't obvious from the code. Add one when something bites; remove it when the
cause is gone.

- **Sample rates differ per model.** Audio is 16 kHz; PANNs CNN14 wants 32 kHz. Feeding the wrong rate gives plausible-looking but wrong scores, not an error.
- **AudioSet names must match exactly.** `vocabulary.yaml` names are matched against the
  tagger's label list character for character. Mismatches fail at model load. Check with
  `soundthing check-config --audioset <csv>`.
- **Retired specific labels must not become generic aliases.** The five former CLAP-only classes have no AudioSet mappings and were removed. Do not map a generic click or whistle to a specific object merely to restore their captions. Version-1 stored runs must be regenerated, not relabeled as version 2.
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

- **The full label set contains speech.** Speech-label outputs must be exempt from
  VAD suppression; otherwise the speech detector suppresses speech's own captions.
- **Ontology nodes and model outputs differ.** Unscored intermediate nodes may connect
  scored ancestors and children. Project through them, and retain multiple parents.
- **AudioSet label absence is not a verified negative.** Derived environment
  coefficients describe annotated co-occurrence in a selected corpus, not calibrated
  real-world event rates. Generated context captions can also be wrong.

- **Stored estimates and effective weights now differ.** In the optional unfloored
  format, a value below the configured minimum is valid evidence. Apply the bound
  before mixing environments; clamping only the final mixture changes uncertain
  scene behavior. Keep legacy bounded validation strict.
