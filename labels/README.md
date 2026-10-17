# Hand labels

One file per eval clip: `labels/<video_id>.json`, in the `clip_labels` format
(see [docs/interfaces.md](../docs/interfaces.md) and
[docs/examples/labels.json](../docs/examples/labels.json)).

- `video_id` must match the clip's file stem (`clip01.mp4` → `clip01`).
- `class_id` uses fine vocabulary ids from `config/vocabulary.yaml`, never coarse groups.
- `env` uses ids from `config/environments.yaml`; use `general` only if no environment fits.
- Label every caption-worthy non-speech sound, including off-screen ones. Mark the onset
  as precisely as you can (eval tolerance is 0.5 s); the end can be approximate.
- Put clips themselves in `data/clips/` (gitignored); share them out of band.

| clip | labeler | environments |
|------|---------|--------------|
|      |         |              |
