"""Hillclimb control: workers, experiments, verification and the orchestrator loop.

Run with the project venv from anywhere:
    .venv/Scripts/python .claude/skills/hillclimb/hc.py <command> ...

State lives in data/hillclimb/ of the main checkout (gitignored):
    state.json        run settings, base branch, workers, applied wins
    journal.md        orchestrator notes, one line per tick
    workers/<id>/     brief.md, out.json, err.txt, results.jsonl
    driver.log        orchestrator loop log
    STOP              present = the loop stops and ends every worker
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

SKILL_DIR = Path(__file__).resolve().parent
MAIN = SKILL_DIR.parents[2]
sys.path.insert(0, str(MAIN / "src"))

from soundthing.harness import paths  # noqa: E402

HOME = MAIN / "data" / "hillclimb"
STATE = HOME / "state.json"
JOURNAL = HOME / "journal.md"
STOP = HOME / "STOP"
REVIEWED = HOME / "reviewed.json"
MIN_HOLDOUT = 100  # holdout clips cached before wins can land
VENV_PY = MAIN / ".venv" / "Scripts" / "python.exe"
WT_ROOT = paths.experiments_root()
COMPONENT = "events"

DEFAULTS = {
    "agy": {"model": "gemini-3.8-flash-high", "effort": None},
    "claude": {"model": "claude-opus-5-5", "effort": "medium"},
}
NO_WINDOW = 0x08000000
NEW_GROUP = 0x00000200


# ------------------------------------------------------------------ state


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"workers": {}, "applied": [], "base_branch": None}


def save_state(st: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def git(*args: str, cwd: Path = MAIN, check: bool = True) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {res.stderr.strip()}")
    return res.stdout.strip()


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def kill_tree(pid: int | None) -> None:
    if pid and pid_alive(pid):
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)


def journal(line: str) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(f"- {now()} {line}\n")


def out(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ------------------------------------------------------------------ workers


def _reviewed() -> dict:
    return json.loads(REVIEWED.read_text(encoding="utf-8")) if REVIEWED.is_file() else {}


def results(wid: str) -> list[dict]:
    f = HOME / "workers" / wid / "results.jsonl"
    if not f.is_file():
        return []
    rev = _reviewed()
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        if f"{wid}/{r.get('run_id')}" in rev:
            r["reviewed"] = rev[f"{wid}/{r.get('run_id')}"]
    return rows


def worker_cmd(w: dict, prompt: str) -> list[str]:
    if w["kind"] == "agy":
        cmd = ["agy", "-p", prompt, "--model", w["model"], "--dangerously-skip-permissions",
               "--output-format", "json"]
    else:
        cmd = ["claude", "-p", prompt, "--model", w["model"], "--dangerously-skip-permissions",
               "--output-format", "json"]
    if w.get("effort"):
        cmd += ["--effort", w["effort"]]
    return cmd


def spawn(kind: str, model: str | None, effort: str | None, focus: str) -> dict:
    st = load_state()
    base = st.get("base_branch")
    if not base:
        raise SystemExit("not started: run `hc.py start` first")
    d = DEFAULTS[kind]
    wid = f"{kind[0]}{uuid.uuid4().hex[:5]}"
    wt = WT_ROOT / f"hc-{wid}"
    git("worktree", "add", "--detach", str(wt), base)
    wdir = HOME / "workers" / wid
    wdir.mkdir(parents=True, exist_ok=True)
    w = {
        "id": wid, "kind": kind, "model": model or d["model"],
        "effort": effort if effort is not None else d["effort"],
        "focus": focus, "worktree": str(wt), "started": now(), "status": "running",
    }
    brief = (SKILL_DIR / "worker.md").read_text(encoding="utf-8")
    for k, v in {
        "WORKER_ID": wid, "WORKTREE": wt.as_posix(), "MAIN": MAIN.as_posix(), "PY": VENV_PY.as_posix(),
        "HC": (SKILL_DIR / "hc.py").as_posix(), "BASE_BRANCH": base, "FOCUS": focus,
        "SHELL_NOTE": "Your shell tool is PowerShell." if kind == "agy" else "Your shell tool is bash (Git Bash).",
    }.items():
        brief = brief.replace("{{" + k + "}}", v)
    (wdir / "brief.md").write_text(brief, encoding="utf-8")
    prompt = f"You are hillclimb worker {wid}. Read the file {(wdir / 'brief.md').as_posix()} and follow it exactly."
    proc = subprocess.Popen(
        worker_cmd(w, prompt), cwd=wt, stdin=subprocess.DEVNULL,
        stdout=open(wdir / "out.json", "wb"), stderr=open(wdir / "err.txt", "wb"),
        creationflags=NO_WINDOW | NEW_GROUP,
    )
    w["pid"] = proc.pid
    st = load_state()
    st["workers"][wid] = w
    save_state(st)
    journal(f"spawned {wid} ({kind} {w['model']} {w['effort'] or ''}) focus: {focus}")
    return w


def kill(wid: str, reason: str) -> dict:
    st = load_state()
    w = st["workers"].get(wid)
    if not w:
        raise SystemExit(f"unknown worker {wid}")
    kill_tree(w.get("pid"))
    wt = Path(w["worktree"])
    git("worktree", "remove", "--force", str(wt), check=False)
    if wt.exists():
        shutil.rmtree(wt, ignore_errors=True)
    w["status"] = "ended"
    w["ended"] = now()
    w["end_reason"] = reason
    save_state(st)
    journal(f"ended {wid}: {reason}")
    return w


def status(brief: bool = False) -> dict:
    from soundthing.harness import clipset, ledger

    st = load_state()
    workers = []
    for w in st["workers"].values():
        if w["status"] != "running" and brief:
            continue
        rs = results(w["id"])
        workers.append({
            "id": w["id"], "kind": w["kind"], "model": w["model"], "effort": w.get("effort"),
            "focus": w["focus"], "alive": pid_alive(w.get("pid")), "status": w["status"],
            "experiments": len(rs),
            "last": rs[-1] if rs else None,
            "unreviewed_wins": [r for r in rs if r.get("accept") and not r.get("reviewed")],
        })
    best = ledger.best(COMPONENT) or {}
    try:
        cs = clipset.status(COMPONENT)
    except Exception as exc:
        cs = {"error": str(exc)}
    return {
        "base_branch": st.get("base_branch"),
        "head": git("rev-parse", "--short", "HEAD"),
        "best": {k: best.get(k) for k in ("objective", "clipset", "kind", "git_sha", "ts")},
        "best_metrics": best.get("metrics"),
        "clipset": cs,
        "workers": workers,
        "applied": st.get("applied", [])[-10:],
        "stop_requested": STOP.exists(),
    }


# ------------------------------------------------------------------ experiments


def _read_patch(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")
    text = text.replace("\r\n", "\n")
    return text if text.endswith("\n") else text + "\n"


def _parse_sets(items: list[str]) -> dict:
    import yaml

    ovr = {}
    for item in items:
        k, _, v = item.partition("=")
        ovr[k.strip()] = yaml.safe_load(v)
    return ovr


def run_params(overrides: dict, hypothesis: str) -> dict:
    from soundthing.harness.mcp_server import TuningTools

    r = TuningTools().score(overrides, hypothesis, component=COMPONENT)
    boot = r.get("bootstrap")
    accept = bool(r.get("passed")) and (boot is None or bool(boot.get("accept")))
    return {"run_id": r.get("run_id"), "objective": r.get("objective"), "passed": r.get("passed"),
            "bootstrap": boot, "limits": r.get("limits"), "accept": accept}


def run_code(patch: str, overrides: dict, hypothesis: str) -> dict:
    from soundthing.harness.runner import run_experiment

    e = run_experiment(COMPONENT, hypothesis, patch=patch, overrides=overrides)
    return {"run_id": e.get("run_id"), "objective": e.get("objective"), "passed": e.get("passed"),
            "bootstrap": e.get("bootstrap"), "limits": e.get("limits"), "accept": bool(e.get("accept")),
            "branch": e.get("branch")}


def exp(wid: str, sets: list[str], patch_file: str | None, hypothesis: str) -> dict:
    st = load_state()
    if wid not in st["workers"] or st["workers"][wid]["status"] != "running":
        raise SystemExit(f"worker {wid} is not running; stop now")
    os.chdir(MAIN)
    overrides = _parse_sets(sets)
    tag = f"[{wid}] {hypothesis}"
    rec = {"ts": now(), "worker": wid, "hypothesis": hypothesis, "overrides": overrides,
           "kind": "code" if patch_file else "params"}
    try:
        if patch_file:
            patch = _read_patch(Path(patch_file))
            if not patch.strip():
                raise ValueError("patch file is empty")
            saved = HOME / "workers" / wid / f"patch-{int(time.time())}.patch"
            saved.write_text(patch, encoding="utf-8", newline="\n")
            rec["patch_file"] = str(saved)
            rec.update(run_code(patch, overrides, tag))
        else:
            if not overrides:
                raise ValueError("give --set KEY=VALUE overrides or --patch FILE")
            rec.update(run_params(overrides, tag))
    except Exception as exc:
        rec.update({"accept": False, "error": f"{type(exc).__name__}: {exc}"})
    rec["verdict"] = "ACCEPTED" if rec.get("accept") else "FAILED"
    with open(HOME / "workers" / wid / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


# ------------------------------------------------------------------ verify + apply


def _mark_reviewed(wid: str, run_id: str, outcome: str) -> None:
    # Workers append to results.jsonl concurrently, so review outcomes live in their own file.
    rev = _reviewed()
    rev[f"{wid}/{run_id}"] = outcome
    tmp = REVIEWED.with_suffix(".tmp")
    tmp.write_text(json.dumps(rev, indent=2), encoding="utf-8")
    os.replace(tmp, REVIEWED)


def _find(wid: str, run_id: str) -> dict:
    for r in results(wid):
        if r.get("run_id") == run_id:
            return r
    raise SystemExit(f"no result {run_id} for worker {wid}")


def _rebaseline() -> dict:
    from soundthing.harness import clipset

    return clipset.rebaseline(COMPONENT, force=True)


def _holdout_ready() -> tuple[bool, str]:
    from soundthing.config import load_config
    from soundthing.harness import clipset

    done, total = clipset.cached_holdout_ids(load_config(MAIN / "config"), COMPONENT)
    if len(done) >= min(MIN_HOLDOUT, total):
        return True, f"{len(done)}/{total}"
    return False, f"only {len(done)}/{total} holdout clips cached (need {MIN_HOLDOUT})"


def _clean(*pathspecs: str) -> bool:
    return not git("status", "--porcelain", "--untracked-files=no", "--", *pathspecs)


def _land(paths_: list[str], msg: str) -> str:
    """Commit only these paths, so other people's uncommitted work in the checkout is untouched."""
    git("add", "-A", "--", *paths_)
    git("-c", "user.name=hillclimb", "-c", "user.email=hillclimb@soundthing.local", "commit", "-q",
        "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", "--", *paths_)
    return git("rev-parse", "--short", "HEAD")


def _fail(wid: str, run_id: str, r: dict, stage: str, v: dict) -> dict:
    if v.get("branch"):
        git("branch", "-D", v["branch"], check=False)
    _mark_reviewed(wid, run_id, f"rejected-on-{stage}")
    journal(f"{stage} FAILED {wid}/{run_id}: {r['hypothesis']}")
    return {"applied": False, "failed": stage, "verify": v}


def apply(wid: str, run_id: str) -> dict:
    """Land a win on the base branch (main) if it holds up: re-run it against the current best
    on the current base (dev), then pass/fail it on the cached holdout clips. Then commit only
    the files it changes and rebaseline. Returns deferred (and leaves the win unreviewed) when
    it can't be checked yet."""
    os.chdir(MAIN)
    st = load_state()
    base = st["base_branch"]
    if git("branch", "--show-current") != base:
        raise SystemExit(f"main checkout is not on {base}")
    r = _find(wid, run_id)
    ready, why = _holdout_ready()
    if not ready:
        return {"applied": False, "deferred": why}
    touched = ["config"] if r["kind"] == "params" else ["src/soundthing/events", "config"]
    if not _clean(*touched):
        return {"applied": False, "deferred": f"uncommitted changes in {touched} of the main checkout"}
    hyp = f"[verify {wid}/{run_id}] {r['hypothesis']}"
    overrides = r.get("overrides") or {}
    if r["kind"] == "params":
        from soundthing.harness.promote import holdout_check, write_overrides

        v = run_params(overrides, hyp)
        if not v["accept"]:
            return _fail(wid, run_id, r, "verify", v)
        if not holdout_check(overrides):
            return _fail(wid, run_id, r, "holdout", v)
        write_overrides(overrides, MAIN / "config")
        msg = (f"hillclimb: {r['hypothesis']}\n\nParameters {json.dumps(overrides)} from worker {wid}, "
               f"run {run_id}; verified on dev as {v['run_id']}, holdout pass ({why} clips cached).")
    else:
        from soundthing.harness.runner import _code_holdout_check

        patch = Path(r["patch_file"]).read_text(encoding="utf-8")
        v = run_code(patch, overrides, hyp)
        if not v["accept"]:
            return _fail(wid, run_id, r, "verify", v)
        if not _code_holdout_check(MAIN, {"branch": v["branch"], "run_id": v["run_id"], "overrides": overrides}):
            return _fail(wid, run_id, r, "holdout", v)
        git("apply", "--whitespace=nowarn", r["patch_file"])
        if overrides:
            from soundthing.harness.promote import write_overrides

            write_overrides(overrides, MAIN / "config")
        msg = (f"hillclimb: {r['hypothesis']}\n\nCode change from worker {wid}, run {run_id}; verified on dev "
               f"as {v['run_id']}, holdout pass ({why} clips cached).")
        for b in (r.get("branch"), v.get("branch")):
            if b:
                git("branch", "-D", b, check=False)
    sha = _land(touched, msg)
    rb = _rebaseline()
    _mark_reviewed(wid, run_id, "applied")
    st = load_state()
    st.setdefault("applied", []).append({"ts": now(), "worker": wid, "run_id": run_id, "sha": sha,
                                         "hypothesis": r["hypothesis"], "objective": rb.get("objective")})
    save_state(st)
    journal(f"APPLIED {wid}/{run_id} as {sha} on {base}: {r['hypothesis']} -> best {rb.get('objective')}")
    return {"applied": True, "sha": sha, "verify": v, "holdout": "pass", "rebaseline": rb}


def reject(wid: str, run_id: str, reason: str) -> dict:
    r = _find(wid, run_id)
    if r.get("branch"):
        git("branch", "-D", r["branch"], check=False)
    _mark_reviewed(wid, run_id, f"rejected: {reason}")
    journal(f"rejected {wid}/{run_id}: {reason}")
    return {"rejected": True}


# ------------------------------------------------------------------ cleanup


def gc() -> dict:
    st = load_state()
    live = {Path(w["worktree"]).name for w in st["workers"].values() if w["status"] == "running"}
    removed = []
    for wt in WT_ROOT.glob("hc-*"):
        if wt.name not in live:
            git("worktree", "remove", "--force", str(wt), check=False)
            shutil.rmtree(wt, ignore_errors=True)
            removed.append(wt.name)
    cutoff = time.time() - 2 * 3600
    for wt in WT_ROOT.glob("soundthing-*"):
        if wt.stat().st_mtime < cutoff:
            git("worktree", "remove", "--force", str(wt), check=False)
            shutil.rmtree(wt, ignore_errors=True)
            removed.append(wt.name)
    git("worktree", "prune", check=False)
    pending = {r.get("branch") for w in st["workers"] for r in results(w)
               if r.get("branch") and r.get("accept") and not r.get("reviewed")}
    branches = [b.strip(" *+") for b in git("branch", "--list", "harness/exp-*").splitlines()]
    deleted = [b for b in branches if b not in pending]
    for b in deleted:
        git("branch", "-D", b, check=False)
    for wid, w in st["workers"].items():
        if w["status"] == "running" and not pid_alive(w.get("pid")) and not Path(w["worktree"]).exists():
            w["status"] = "ended"
    save_state(st)
    return {"removed_worktrees": removed, "deleted_branches": deleted}


# ------------------------------------------------------------------ start / drive / stop


def start(roster: list[dict], orch_model: str, orch_effort: str, interval: int,
          allow_branch: bool = False) -> dict:
    HOME.mkdir(parents=True, exist_ok=True)
    STOP.unlink(missing_ok=True)
    st = load_state()
    cur = git("branch", "--show-current")
    if cur != "main" and not allow_branch:
        raise SystemExit(f"main checkout is on {cur!r}; wins land on the checked-out branch, so switch to main "
                         "(or pass --allow-branch)")
    st.update({"base_branch": cur, "started_from": git("rev-parse", "HEAD"), "roster": roster,
               "orchestrator": {"model": orch_model, "effort": orch_effort}, "interval": interval})
    save_state(st)
    from soundthing.harness import clipset

    rb = clipset.rebaseline(COMPONENT, force=not paths.best_path(COMPONENT).is_file())
    log = open(HOME / "driver.log", "ab")
    proc = subprocess.Popen([str(VENV_PY), str(Path(__file__).resolve()), "drive"], cwd=MAIN,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            creationflags=NO_WINDOW | NEW_GROUP | 0x00000008)  # DETACHED_PROCESS
    st = load_state()
    st["driver_pid"] = proc.pid
    save_state(st)
    journal(f"started on {cur}; roster {json.dumps(roster)}; orchestrator {orch_model}/{orch_effort}")
    return {"base_branch": cur, "driver_pid": proc.pid, "rebaseline": rb, "log": str(HOME / "driver.log")}


def _digest(s: dict) -> str:
    parts = [f"{w['id']}:{w['alive']}:{w['experiments']}:{len(w['unreviewed_wins'])}" for w in s["workers"]]
    return "|".join(sorted(parts))


def drive() -> None:
    st = load_state()
    orch = st["orchestrator"]
    interval = int(st.get("interval", 60))
    last_digest, last_tick, tick = None, 0.0, 0
    while not STOP.exists():
        try:
            s = status(brief=True)
            running = len([w for w in s["workers"] if w["alive"]])
            want = sum(int(r["count"]) for r in st["roster"])
            digest = _digest(s)
            idle = digest == last_digest and running >= want and time.time() - last_tick < 900
            if not idle:
                tick += 1
                prompt = (f"Hillclimb orchestrator tick {tick}. Read {SKILL_DIR / 'orchestrator.md'} and do "
                          f"one tick as it describes, then end your turn.")
                cmd = ["claude", "-p", prompt, "--model", orch["model"], "--effort", orch["effort"],
                       "--dangerously-skip-permissions", "--output-format", "json"]
                print(f"{now()} tick {tick} ({running}/{want} workers running)", flush=True)
                res = subprocess.run(cmd, cwd=MAIN, stdin=subprocess.DEVNULL, capture_output=True,
                                     text=True, encoding="utf-8", errors="replace", timeout=3600)
                try:
                    body = json.loads(res.stdout)
                    print(f"{now()} tick {tick} done: {str(body.get('result', ''))[:400]}", flush=True)
                except Exception:
                    print(f"{now()} tick {tick} exit {res.returncode}: {res.stdout[-400:]} {res.stderr[-400:]}", flush=True)
                last_tick = time.time()
                last_digest = _digest(status(brief=True))
        except Exception as exc:
            print(f"{now()} driver error: {exc}", flush=True)
        for _ in range(interval):
            if STOP.exists():
                break
            time.sleep(1)
    print(f"{now()} stop requested; ending workers", flush=True)
    _end_all("stopped")


def _end_all(reason: str) -> None:
    st = load_state()
    for wid, w in st["workers"].items():
        if w["status"] == "running":
            kill(wid, reason)
    gc()
    journal("loop stopped")


def stop() -> dict:
    HOME.mkdir(parents=True, exist_ok=True)
    STOP.write_text(now(), encoding="utf-8")
    st = load_state()
    kill_tree(st.get("driver_pid"))
    _end_all("stopped by user")
    return {"stopped": True, "base_branch": st.get("base_branch"), "applied": st.get("applied", [])}


# ------------------------------------------------------------------ cli


def main() -> None:
    p = argparse.ArgumentParser(prog="hc.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--roster", required=True, help='JSON list: [{"kind":"agy","count":5,"model":null,"effort":null}]')
    s.add_argument("--orchestrator-model", default="claude-opus-5-5")
    s.add_argument("--orchestrator-effort", default="medium")
    s.add_argument("--interval", type=int, default=60)
    s.add_argument("--allow-branch", action="store_true", help="land wins on a branch other than main")
    sub.add_parser("drive")
    sub.add_parser("stop")
    s = sub.add_parser("status")
    s.add_argument("--all", action="store_true", help="include ended workers")
    s = sub.add_parser("spawn")
    s.add_argument("--kind", required=True, choices=["agy", "claude"])
    s.add_argument("--model")
    s.add_argument("--effort")
    s.add_argument("--focus", required=True)
    s = sub.add_parser("kill")
    s.add_argument("worker")
    s.add_argument("--reason", default="ended by orchestrator")
    s = sub.add_parser("exp")
    s.add_argument("--worker", required=True)
    s.add_argument("--set", dest="sets", action="append", default=[])
    s.add_argument("--patch")
    s.add_argument("--hypothesis", required=True)
    s = sub.add_parser("apply")
    s.add_argument("worker")
    s.add_argument("run_id")
    s = sub.add_parser("reject")
    s.add_argument("worker")
    s.add_argument("run_id")
    s.add_argument("--reason", required=True)
    sub.add_parser("gc")
    s = sub.add_parser("results")
    s.add_argument("--last", type=int, default=40)

    a = p.parse_args()
    if a.cmd == "start":
        out(start(json.loads(a.roster), a.orchestrator_model, a.orchestrator_effort, a.interval, a.allow_branch))
    elif a.cmd == "drive":
        drive()
    elif a.cmd == "stop":
        out(stop())
    elif a.cmd == "status":
        out(status(brief=not a.all))
    elif a.cmd == "spawn":
        out(spawn(a.kind, a.model, a.effort, a.focus))
    elif a.cmd == "kill":
        out(kill(a.worker, a.reason))
    elif a.cmd == "exp":
        rec = exp(a.worker, a.sets, a.patch, a.hypothesis)
        out(rec)
        sys.exit(0 if rec.get("accept") else 3)
    elif a.cmd == "apply":
        out(apply(a.worker, a.run_id))
    elif a.cmd == "reject":
        out(reject(a.worker, a.run_id, a.reason))
    elif a.cmd == "gc":
        out(gc())
    elif a.cmd == "results":
        st = load_state()
        rows = [r for wid in st["workers"] for r in results(wid)]
        rows.sort(key=lambda r: r.get("ts", ""))
        out([{k: r.get(k) for k in ("ts", "worker", "kind", "hypothesis", "overrides", "objective",
                                     "verdict", "reviewed", "error")} for r in rows[-a.last:]])


if __name__ == "__main__":
    main()
