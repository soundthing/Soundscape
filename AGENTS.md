# Agent rules for soundthing

Several people build this repo at the same time, each with their own agent (Claude Code,
Codex, Antigravity or anything else that reads `AGENTS.md`). These rules keep agents
from breaking each other's work. They are **mandatory** for every agent. If a rule
conflicts with a request, stop and say so; don't work around it.

Project overview: [project.md](project.md). Contract spec: [docs/interfaces.md](docs/interfaces.md).

The evaluation harness (`src/soundthing/harness/`) scores and tunes every stage (events,
environment, captions, end to end), by config and by code changes in experiment worktrees.

## 1. Changing a shared interface

The shared interfaces are `src/soundthing/contracts.py`, `stages.py`, `config.py`,
`docs/interfaces.md`, `docs/examples/`, and `config/environments.yaml`,
`vocabulary.yaml`, `priors.yaml`. When you change one:

- **One change, in its own commit/PR**, prefixed `interface:`. Never mix it with feature work.
- **Update everything together**: `contracts.py`, `docs/interfaces.md`,
  `docs/examples/`, the stubs, and tests in the same commit.
- **Breaking change**: bump `SCHEMA_VERSION` in `contracts.py`, and say in the commit
  message which stored files need regenerating.
- **Prefer additive changes**: a new optional field with a default instead of renaming or
  retyping one.
- **Renaming a class or environment id** means updating `vocabulary.yaml`/`environments.yaml`,
  `priors.yaml`, `thresholds.yaml` and every `labels/*.json` that uses it.

## 2. Don't work around the contract

- Don't add undocumented fields to the JSON files, or pass extra data through side files.
- Don't redefine or copy the contract types in your own module.
- Don't loosen validation (`validate()`, `validate_segments`, `Config.validate`) to make
  your output pass. Fix the output.
- Don't hard-code environment or class ids in code. They belong in `config/`.
- Don't switch a `backend:` in `config/pipeline.yaml` to `real` until that stage runs
  end to end. Use a local `--config` copy to test.

## 3. Before every commit

Run both, and don't commit if either fails:

```bash
pytest -q
```

```bash
soundthing check-config
```

Keep commits small and focused. Never force-push or rewrite shared history.
