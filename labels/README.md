# Eval labels

We don't hand-label clips. The eval set is the **AudioSet strong-label eval split**
(Google's `audioset_eval_strong.tsv`), converted to the `clip_labels` format
(see [docs/interfaces.md](../docs/interfaces.md)) and mapped to `config/vocabulary.yaml`
through each class's `audioset:` names. About 6k 10 s clips have at least one mapped event.
They have no environment labels, so they score baseline mode only.

It lives in the evaluation harness, not here: about 80% goes to `data/harness/dev/` and 20% to
`data/harness/holdout/` (gitignored, split by a hash of the YouTube id). To rebuild it:

1. Download `audioset_eval_strong.tsv` from
   `http://storage.googleapis.com/us_audioset/youtube_corpus/strong/`.
2. Get the audio as `<segment_id>.wav` (16 kHz mono). It comes from the Hugging Face
   dataset `agkphysics/AudioSet`, `data/eval/*.parquet`.
3. Run:
   `python -m soundthing.harness import-audioset --tsv audioset_eval_strong.tsv --class-csv ~/panns_data/class_labels_indices.csv --audio <wav dir> [--build-caches]`

This folder can still hold `clip_labels` JSON for `soundthing eval` on full pipeline
runs. `video_id` must match the clip's file stem, and `class_id`/`env` must use the ids
in `config/`.
