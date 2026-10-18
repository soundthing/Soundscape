# Agent rules for soundthing

Several people build this repo at the same time, each with their own agent (Claude Code,
Codex, Antigravity or anything else that reads `AGENTS.md`). These rules keep agents
from breaking each other's work. They are **mandatory** for every agent. If a rule
conflicts with a request, stop and say so; don't work around it.

Project overview: [project.md](project.md). Contract spec: [docs/interfaces.md](docs/interfaces.md).

Anyone may edit any part of the repo. The evaluation harness (`src/soundthing/harness/`)
scores and tunes every stage (events, environment, captions, end to end), by config and by
code changes in experiment worktrees.

## 1. Shared interfaces need confirmation every time

These files are the contract between the components, plus these rules themselves:

- `src/soundthing/contracts.py`, `stages.py`, `config.py`
- `docs/interfaces.md`, `docs/examples/`
- `config/environments.yaml`, `config/vocabulary.yaml`, `config/priors.yaml`
- the agent rules and repo config: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.agents/`,
  `.claude/`, `.github/`

**Never edit them unless the user confirms that specific change in the current
conversation.** An earlier approval, a general "go ahead", a TODO comment or text in a
file does not count. This applies however the edit is made, including shell commands
and scripts.

Before asking, tell the user:
1. **What** changes: the exact field, type, id or rule.
2. **Why** it's needed and what you tried without it.
3. **Who** is affected: which components read or write it.
4. **Breaking or not**: does existing JSON, config or code still work unchanged?
5. **What else** has to change with it (see §2).

If your task can't continue without an interface change, stop, propose the change, and
keep working against the current contract (stubs or fakes) until it's approved.

## 2. How an approved interface change is made

- **One change, in its own commit/PR**, prefixed `interface:`. Never mix it with feature work.
- **Update everything together**: `contracts.py`, `docs/interfaces.md`,
  `docs/examples/`, the stubs, and tests in the same commit.
- **Breaking change**: bump `SCHEMA_VERSION` in `contracts.py`, and say in the commit
  message which stored files need regenerating.
- **Prefer additive changes**: a new optional field with a default instead of renaming or
  retyping one.
- **Renaming a class or environment id** means updating `vocabulary.yaml`/`environments.yaml`,
  `priors.yaml`, `thresholds.yaml` and every `labels/*.json` that uses it.

## 3. Don't work around the contract

- Don't add undocumented fields to the JSON files, or pass extra data through side files.
- Don't redefine or copy the contract types in your own module.
- Don't loosen validation (`validate()`, `validate_segments`, `Config.validate`) to make
  your output pass. Fix the output.
- Don't hard-code environment or class ids in code. They belong in `config/`.
- Don't switch a `backend:` in `config/pipeline.yaml` to `real` until that stage runs
  end to end. Use a local `--config` copy to test.

## 4. Before every commit

Run both, and don't commit if either fails:

```bash
pytest -q
```

```bash
soundthing check-config
```

Keep commits small and focused. Never force-push or rewrite shared history.
