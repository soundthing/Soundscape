# Stack

- **`harness` extra:** mcp (2.x server SDK), optuna (TPE for sweeps), kaggle CLI. Imported
  lazily so the base package and pipeline run without them.
- **Kaggle cache runner:** kaggle 1.7 CLI/API on free T4/P100 builds model caches; the
  kernel installs the repo at a pushed git sha.
- **State:** `data/harness/` in the main checkout, shared by all worktrees
  (`SOUNDTHING_HARNESS_HOME` overrides; a pre-existing `data/tuning/` is used until renamed).
