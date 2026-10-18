# Kaggle cache runner

Builds harness feature caches on a free Kaggle GPU (T4/P100). The kernel runs the code from
your working tree (embedded in the script as a tarball), so nothing needs committing or pushing,
and no GitHub token is needed.

## Setup (once)

- Kaggle CLI >= 1.8 in its own Python 3.11+ env: `uv tool install --python 3.12 kaggle`.
  The harness finds it on `PATH`, in `~/.local/bin`, or via `SOUNDTHING_KAGGLE_CLI`.
- API token (kaggle.com → Settings → API) saved as `~/.kaggle/access_token`; phone-verify the
  account (GPU and internet need it).
- `KAGGLE_USERNAME` set, or override slugs with `SOUNDTHING_KAGGLE_KERNEL`,
  `SOUNDTHING_KAGGLE_DATASET`, `SOUNDTHING_KAGGLE_AUDIOSET`, `SOUNDTHING_KAGGLE_ESC50`.
- AudioSet audio uploaded as the private dataset `<user>/soundthing-audioset-eval`:
  `audio/<segment_id>.flac`, `audioset_eval_strong.tsv`, `class_labels_indices.csv`
  (see `labels/README.md` for sources). Upload from inside the folder:
  `kaggle datasets create -p . -r zip`.

## Loop

1. `python -m soundthing.harness recache --window 1.0 --hop 0.25 --model panns_cnn14`
   pushes the kernel.
2. Poll `kernel_status()` until `complete` (about an hour for the full AudioSet set).
3. `download_cache(from_kernel=True)`, then `install_cache(download_dir, cfg)`: checks the
   cache key, refuses dev ids that are in the holdout, and installs caches + dev labels.

Holdout labels never leave the kernel. Locally, the holdout split comes from running
`import-audioset` (same deterministic split).
