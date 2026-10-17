# Kaggle Tuning Cache Runner

Builds development audio feature caches on a free Kaggle GPU (T4/P100) for sound event tuning.

## Environment Variables and Credentials

- `KAGGLE_USERNAME` & `KAGGLE_KEY`: Kaggle API credentials (or configure `~/.kaggle/kaggle.json`).
- `SOUNDTHING_REPO_URL`: Git remote URL of the repository (required for cloning on Kaggle).
- `SOUNDTHING_KAGGLE_KERNEL`: Optional kernel slug override (defaults to `${KAGGLE_USERNAME}/soundthing-tuning-cache`).
- `SOUNDTHING_KAGGLE_DATASET`: Optional cache dataset slug override (defaults to `${KAGGLE_USERNAME}/soundthing-tuning-cache-data`).
- `SOUNDTHING_KAGGLE_ESC50`: Optional Kaggle ESC-50 dataset slug mirror.
- Kaggle secret `GITHUB_TOKEN`: Add as a Kaggle User Secret (`GITHUB_TOKEN`) if cloning a private GitHub repository.

## Execution Loop

1. **Request Recache**: Call `request_recache(window, hop, model)` to push the kernel script at the current git SHA.
2. **Poll Status**: Wait until `kernel_status()` returns `"complete"`.
3. **Save Dataset or Download Output**: Save the completed kernel run output as a private Kaggle dataset version, or download directly via `download_cache(from_kernel=True)`.
4. **Install Cache**: Call `install_cache(download_dir, cfg)` to verify the cache key against the configuration hash, ensure disjoint dev/holdout IDs, and populate `data/tuning/cache/` and `data/tuning/dev/labels/`.

## Important Note

The kernel executes `git clone` and checkouts `git_sha`. Only committed and pushed code is built on Kaggle. Uncommitted local working tree changes will not be included in the remote build.
