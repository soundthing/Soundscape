# Stack

- **`harness` extra:** mcp (2.x server SDK), optuna (TPE for sweeps), kaggle CLI. Imported
  lazily so the base package and pipeline run without them.
- **Kaggle cache runner:** kaggle CLI (>=1.8, own Python 3.12 env) on a free T4/P100 builds
  model caches; the kernel installs a tarball of the pusher's working tree embedded in the script.
- **State:** `data/harness/` in the main checkout, shared by all worktrees
  (`SOUNDTHING_HARNESS_HOME` overrides; a pre-existing `data/tuning/` is used until renamed).
