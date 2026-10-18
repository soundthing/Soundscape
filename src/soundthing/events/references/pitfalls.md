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
- **Minimum duration can still remove isolated detections.** Removing score smoothing
  does not remove the 0.3-second minimum event duration. A lone 0.25-second detection
  is still dropped unless gap merging joins it with another run.
- **Short events stay on screen longer than their span.** The caption writer extends cues
  to a minimum duration and then trims ancestor overlap, which can make parents shorter.
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
- **Demucs removes dialogue well but music only partly.** Demucs separates into drums,
  bass, other, and vocals. The effects stem is mix minus vocals, so drums, bass, and
  other instruments are kept in "effects" and can still mask sound events.
- **Extraction is 16 kHz so separator input is band-limited.** Demucs expects 44.1 kHz
  stereo. Upsampling 16 kHz mono to 44.1 kHz cannot restore high-frequency harmonics, so
  separation quality above 8 kHz is unavailable.
- **Separated audio is out of distribution for PANNs.** Models were pretrained on
  unseparated natural audio. Artifacts and altered stem balances from separation mean
  thresholds tuned on raw audio may not transfer cleanly.
- **Windows HF cache symlink warning is harmless.** On Windows without developer mode
  enabled, huggingface_hub logs a warning about symlinks and falls back to file copies.
  This is harmless and does not affect model loading or caching.

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
