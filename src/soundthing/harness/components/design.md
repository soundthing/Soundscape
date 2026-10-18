# Harness component design contract (shared by all 4 parallel tracks)

Four tracks run in parallel from the same base commit. Each codes against THIS contract.
Where you need another track's code that does not exist yet in your worktree, use a
lazy import inside the function plus a fake in your tests (monkeypatch BOTH
`sys.modules[...]` AND the package attribute, e.g. `soundthing.harness.components.environment`,
because once the real module exists the package attribute wins).

Already committed on the base: `src/soundthing/harness/components/base.py` (Component
protocol, Clip, Evaluation, SECTION_OWNERS, key_owner) and
`src/soundthing/harness/components/__init__.py` (REGISTRY, get(name), names()).
Read both first. Do not change their public names/signatures; you may add to them only
if your track owns that (nobody does: report needed changes instead).

## Track A (core): generic score / ledger / bootstrap / sweep / promote / space / CLI / MCP

### `soundthing.harness.score`
```python
@dataclass
class ScoreResult:
    run_id: str              # uuid4 hex[:12]
    component: str
    objective: float
    metrics: dict[str, float]  # includes "objective"
    per_clip: dict[str, dict]  # each has "stats": list[float]
    diagnostics: dict
    limits: dict[str, bool]
    passed: bool
    overrides: dict
    split: str
    def to_dict(self) -> dict    # asdict + every metrics key ALSO copied to top level
    def __getattr__(self, name)  # falls back to metrics[name] (so result.baseline works for events)

def score(overrides=None, split="dev", cfg=None, best=None, labels_dir=None,
          component: str = "events") -> ScoreResult
```
`score` = `get(component)`; apply_overrides; check every override key starts with one of
`comp.sections` (else ValueError); resolve labels dir as today; `comp.load_clips`;
`comp.evaluate`; `best = ledger.best(component)` if None; `comp.limits(metrics, best)`.
Does not write the ledger.

### `soundthing.harness.ledger`
- Every entry written by score/sweep logging carries `"component"`; a missing key means "events".
- `best(component="events")`, `set_best(entry, component="events")`: events uses
  `paths.best_path()` (unchanged file `best.json`); others use `paths.best_path(component)`
  -> `best-<component>.json` next to it. `paths.best_path(component: str = "events")`.
- `read(component: str | None = None)` filters when given.

### `soundthing.harness.bootstrap`
`paired_bootstrap(cand_per_clip, best_per_clip, n_boot=1000, seed=0, component="events")`
uses `per_clip[vid]["stats"]` and `get(component).stat_objective`. Back-compat: an events
per_clip entry without "stats" falls back to its "conditioned" counts.

### CLI (`python -m soundthing.harness ...`)
- Every existing command that scores/tunes (`score`, `sweep`, `promote`, `space`, `best`,
  `ledger`) takes `--component NAME` (default events).
- `score` also takes `--json-out PATH`: writes `ScoreResult.to_dict()` JSON there
  (Track D's trusted scorer runs this in a subprocess) and `--no-log` (skip ledger write).
- `list-components`: prints `[{name, owner, sections, code_paths}]` as JSON.

### MCP (`TuningTools` in mcp_server.py)
Existing tools gain an optional `component: str = "events"` arg. New tool `list_components`.

### Promotion ownership
`harness_role()` in `soundthing.harness.paths`: env `SOUNDTHING_ROLE` (default "person1").
`promote(run_id, ...)`: keys whose `key_owner(k) == harness_role()` are written as today;
keys owned by another role are not written: they go into a proposal via
`proposals.propose_shared_change(...)` (one proposal, note says which owner). Shared keys
still refuse as today. Tuning-log entry goes to `comp.tuning_log`. Holdout check runs only
for components that have a holdout (events); others skip it with `"holdout": "n/a"`.

### `components/events.py` (Track A)
`COMPONENT` wrapping today's events scoring exactly (objective F0.5 conditioned, metrics
baseline/wrong_env/safety_recall/fp_per_min/grid_f1, limits identical to today, per_clip
modes as today plus "stats" = conditioned [tp_fine, tp_coarse, fp, fn], n_stats=4).
sections ("events.", "thresholds."), code_paths ("src/soundthing/events",),
tuning_log "src/soundthing/events/references/tuning-log.md", owner person1.
Also exports, for Tracks C (pipeline/captions):
```python
def load_event_caches(cfg, labels) -> list[ClipCache]       # raises FileNotFoundError like today
def replay_events(cache, segments, cfg, text_emb) -> list[CaptionEvent]   # = events.replay.replay
def text_emb_for(cfg, caches) -> np.ndarray | None         # today's step 4 in score.py
def labeled_env_segments(labels, cfg, confidence=0.7) -> list[EnvSegment]  # moved from score.py (re-export there)
def event_counts(matches: list[ClipMatch]) -> dict          # safety_recall, fp_per_min helpers as needed
```

## Track B: `components/environment.py`
```python
@dataclass
class EnvCache:
    video_id: str; duration: float; key: str
    spans: list[tuple[float, float]]
    env_ids: list[str]                     # non-general envs, config order
    audio_logits: np.ndarray | None        # (S, E) raw prompt similarities, pre-temperature
    visual_logits: np.ndarray | None       # (S, E) or None (audio-only clips)
def predict_segments(cache: EnvCache, cfg: Config) -> list[EnvSegment]   # replay; Track C uses it
def load_env_cache(video_id: str, cfg: Config) -> EnvCache                # FileNotFoundError if missing
COMPONENT  # name "environment", owner person2, sections ("environment.",)
```

## Track C: `components/captions.py`, `components/pipeline.py`
Use Track A's events helpers and Track B's `load_env_cache` / `predict_segments`.

## Track D: trusted scorer, runner, code promotion
```python
# soundthing.harness.trusted
def trusted_score(worktree: Path, component: str, overrides: dict, *, pinned: str | None = None,
                  split="dev", timeout: float = 3600) -> dict    # ScoreResult.to_dict()
# soundthing.harness.runner
def run_experiment(component: str, hypothesis: str, *, patch: str | None = None,
                   overrides: dict | None = None, base: str | None = None,
                   keep_worktree: bool = False) -> dict
def promote_code(run_id: str) -> dict
```
Trusted scorer runs `python -m soundthing.harness score --component C --overrides JSON
--json-out F --no-log` in a subprocess against an overlay tree (see Track D brief).
