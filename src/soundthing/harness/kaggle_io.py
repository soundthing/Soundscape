"""Kaggle client and helpers for remote GPU cache builds."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

from soundthing.config import Config, load_config
from soundthing.harness.paths import cache_root, dev_labels_dir, harness_home


def _kaggle_cli() -> str:
    env = os.environ.get("SOUNDTHING_KAGGLE_CLI")
    if env:
        return env
    found = shutil.which("kaggle")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / ("kaggle.exe" if os.name == "nt" else "kaggle")
    if local.is_file():
        return str(local)
    raise RuntimeError("kaggle CLI not found; install it (`uv tool install kaggle`) or set SOUNDTHING_KAGGLE_CLI")


class CliApi:
    """The subset of KaggleApi we use, backed by the `kaggle` CLI.

    The CLI can live in its own Python: new-style `KGAT_` API tokens need kaggle>=1.8,
    which requires Python 3.11+.
    """

    def _run(self, *args: str) -> str:
        res = subprocess.run([_kaggle_cli(), *args], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"kaggle {' '.join(args)} failed: {res.stderr.strip() or res.stdout.strip()}")
        return res.stdout

    def kernels_push(self, folder: str) -> dict[str, Any]:
        return {"output": self._run("kernels", "push", "-p", folder).strip()}

    def kernels_status(self, slug: str) -> str:
        out = self._run("kernels", "status", slug)
        m = re.search(r'status "?(?:KernelWorkerStatus\.)?(\w+)"?', out)
        return m.group(1) if m else out.strip()

    def kernels_output(self, slug: str, path: str) -> None:
        self._run("kernels", "output", slug, "-p", path, "-o")

    def dataset_download_files(self, slug: str, path: str, unzip: bool) -> None:
        args = ["datasets", "download", slug, "-p", path]
        if unzip:
            args.append("--unzip")
        self._run(*args)


class KaggleClient:
    """Thin wrapper around Kaggle API calls."""

    def __init__(self, api: Any | None = None) -> None:
        self._api = api

    def _get_api(self) -> Any:
        if self._api is None:
            self._api = CliApi()
        return self._api

    def push_kernel(self, folder: Path) -> dict[str, Any]:
        """Push a kernel folder via the Kaggle API."""
        res = self._get_api().kernels_push(str(folder))
        if isinstance(res, dict):
            return res
        if hasattr(res, "__dict__"):
            return vars(res)
        return {"result": res} if res is not None else {}

    def kernel_status(self, slug: str) -> str:
        """Return normalized lower-case status string."""
        res = self._get_api().kernels_status(slug)
        if isinstance(res, dict):
            status = res.get("status", "")
        elif hasattr(res, "status"):
            status = getattr(res, "status")
        elif isinstance(res, str):
            status = res
        else:
            status = str(res)
        return str(status).strip().lower()

    def download_kernel_output(self, slug: str, dest: Path) -> Path:
        """Download output files from a kernel run."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self._get_api().kernels_output(slug, str(dest))
        return dest

    def download_dataset(self, slug: str, dest: Path) -> Path:
        """Download and unzip a dataset."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self._get_api().dataset_download_files(slug, path=str(dest), unzip=True)
        return dest


def kernel_slug() -> str:
    """Return the configured kernel slug."""
    env = os.environ.get("SOUNDTHING_KAGGLE_KERNEL")
    if env:
        return env
    user = os.environ.get("KAGGLE_USERNAME")
    if user:
        return f"{user}/soundthing-tuning-cache"
    raise RuntimeError(
        "Kaggle kernel slug is not configured. Set SOUNDTHING_KAGGLE_KERNEL or KAGGLE_USERNAME."
    )


def _default_dataset_slug() -> str:
    env = os.environ.get("SOUNDTHING_KAGGLE_DATASET")
    if env:
        return env
    user = os.environ.get("KAGGLE_USERNAME")
    if user:
        return f"{user}/soundthing-tuning-cache-data"
    raise RuntimeError(
        "Kaggle dataset slug is not configured. Set SOUNDTHING_KAGGLE_DATASET or KAGGLE_USERNAME."
    )


dataset_slug = _default_dataset_slug


def audioset_dataset_slug() -> str | None:
    """Kaggle dataset holding the AudioSet eval audio (`audio/*.flac`) and label files."""
    env = os.environ.get("SOUNDTHING_KAGGLE_AUDIOSET")
    if env:
        return env
    user = os.environ.get("KAGGLE_USERNAME")
    return f"{user}/soundthing-audioset-eval" if user else None


BUNDLE_PATHS = ("src/soundthing", "config", "pyproject.toml")


def code_bundle(root: Path | None = None) -> tuple[str, str]:
    """Tar+gzip the package and config from the working tree; return (base64, sha256)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]

    def skip_cache(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in BUNDLE_PATHS:
            tar.add(root / rel, arcname=rel, filter=skip_cache)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def push_cache_kernel(
    window: float,
    hop: float,
    model: str,
    *,
    git_sha: str | None = None,
    n_per_env: int = 20,
    seed: int = 0,
    holdout_frac: float = 0.2,
    client: KaggleClient | None = None,
) -> dict[str, Any]:
    """Render kernel files (with the working tree's code embedded) and push to Kaggle."""
    if git_sha is None:
        import soundthing.harness.ledger as ledger_module

        git_sha = ledger_module.git_state()["sha"]

    slug = kernel_slug()
    title = slug.split("/")[-1]

    kaggle_dir = Path(__file__).resolve().parent / "kaggle"
    template_path = kaggle_dir / "kernel-metadata.template.json"
    template_data = json.loads(template_path.read_text(encoding="utf-8"))

    default_esc50 = template_data.pop("_soundthing_default_esc50_dataset", None)
    esc50_slug = os.environ.get("SOUNDTHING_KAGGLE_ESC50") or default_esc50

    meta = {k: v for k, v in template_data.items() if not k.startswith("_soundthing")}
    meta["id"] = slug
    meta["title"] = title
    audioset_slug = audioset_dataset_slug()
    meta["dataset_sources"] = [d for d in (audioset_slug, esc50_slug) if d]
    meta["enable_gpu"] = True
    meta["enable_internet"] = True
    meta["is_private"] = True

    params = {
        "window_sec": window,
        "hop_sec": hop,
        "tagger_model": model,
        "git_sha": git_sha,
        "n_per_env": n_per_env,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "esc50_dataset": esc50_slug,
        "audioset_dataset": audioset_slug,
    }
    bundle_b64, bundle_sha = code_bundle()
    params["bundle_sha256"] = bundle_sha

    kernel_src = (kaggle_dir / "cache_kernel.py").read_text(encoding="utf-8")
    params_line = f"PARAMS = {json.dumps(params)}  # filled in by kaggle_io.push_cache_kernel"
    kernel_code = re.sub(r"^PARAMS\s*=.*$", lambda _: params_line, kernel_src, count=1, flags=re.MULTILINE)
    bundle_line = f'BUNDLE = "{bundle_b64}"  # filled in by kaggle_io.push_cache_kernel'
    kernel_code = re.sub(r"^BUNDLE\s*=.*$", lambda _: bundle_line, kernel_code, count=1, flags=re.MULTILINE)

    if client is None:
        client = KaggleClient()

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (folder / "cache_kernel.py").write_text(kernel_code, encoding="utf-8")
        push_res = client.push_kernel(folder)

    return {"kernel": slug, "params": params, "push": push_res}


def kernel_status(slug: str | None = None, client: KaggleClient | None = None) -> str:
    """Return normalized status of a Kaggle kernel."""
    if slug is None:
        slug = kernel_slug()
    if client is None:
        client = KaggleClient()
    return client.kernel_status(slug)


def download_cache(
    dataset_slug: str | None = None,
    dest: Path | None = None,
    *,
    from_kernel: bool = False,
    client: KaggleClient | None = None,
) -> Path:
    """Download cache dataset or kernel output."""
    if client is None:
        client = KaggleClient()
    if from_kernel:
        target_slug = dataset_slug if dataset_slug is not None else kernel_slug()
    else:
        target_slug = dataset_slug if dataset_slug is not None else _default_dataset_slug()

    if dest is None:
        slug_name = target_slug.split("/")[-1]
        dest = harness_home() / "downloads" / slug_name

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if from_kernel:
        return client.download_kernel_output(target_slug, dest)
    return client.download_dataset(target_slug, dest)


def verify_cache_key(cache_dir: Path, cfg: Config) -> str:
    """Verify that cache_dir contains meta.json matching cfg's cache key."""
    import soundthing.events.cache as cache_module

    expected_key = cache_module.cache_key(cfg)

    meta_file = Path(cache_dir) / "meta.json"
    if not meta_file.exists():
        meta_file = Path(cache_dir) / "tuning" / "meta.json"

    if not meta_file.exists():
        raise ValueError(
            f"meta.json not found in {cache_dir} or {Path(cache_dir) / 'tuning'}; "
            f"missing key, expected {expected_key!r}"
        )

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    actual_key = meta.get("cache_key")
    if actual_key != expected_key:
        raise ValueError(
            f"Cache key mismatch: found {actual_key!r}, expected {expected_key!r}"
        )
    return str(actual_key)


def install_cache(src_dir: Path, cfg: Config) -> Path:
    """Install downloaded cache and dev labels into the tuning home."""
    src_dir = Path(src_dir)
    key = verify_cache_key(src_dir, cfg)

    tuning_dir = src_dir / "tuning" if (src_dir / "tuning").is_dir() else src_dir

    if (tuning_dir / "cache" / key).exists():
        npz_dir = tuning_dir / "cache" / key
    elif (src_dir / "cache" / key).exists():
        npz_dir = src_dir / "cache" / key
    elif (src_dir / key).exists():
        npz_dir = src_dir / key
    else:
        npz_dir = tuning_dir / "cache" / key

    if (tuning_dir / "dev" / "labels").exists():
        labels_dir = tuning_dir / "dev" / "labels"
        dev_dir = tuning_dir / "dev"
    elif (src_dir / "dev" / "labels").exists():
        labels_dir = src_dir / "dev" / "labels"
        dev_dir = src_dir / "dev"
    elif (src_dir / "labels").exists():
        labels_dir = src_dir / "labels"
        dev_dir = src_dir
    else:
        labels_dir = tuning_dir / "dev" / "labels"
        dev_dir = tuning_dir / "dev"

    label_files = sorted(labels_dir.glob("*.json")) if labels_dir.exists() else []
    label_ids = [p.stem for p in label_files]

    import soundthing.harness.holdout as holdout_module

    holdout_module.assert_disjoint(label_ids)

    target_cache_dir = cache_root() / key
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    if npz_dir.exists():
        for npz in npz_dir.glob("*.npz"):
            shutil.copy2(npz, target_cache_dir / npz.name)

    target_labels_dir = dev_labels_dir()
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    for lf in label_files:
        shutil.copy2(lf, target_labels_dir / lf.name)

    manifest_file = dev_dir / "manifest.json"
    if manifest_file.exists():
        target_labels_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_file, target_labels_dir.parent / "manifest.json")

    return target_cache_dir


def request_recache(
    window: float,
    hop: float,
    model: str,
    *,
    client: KaggleClient | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """MCP entrypoint: push a cache kernel with the requested parameters."""
    import soundthing.events.cache as cache_module
    import soundthing.harness.ledger as ledger_module
    import soundthing.harness.overrides as overrides_module

    cfg = load_config()
    overrides = {
        "events.window_sec": window,
        "events.hop_sec": hop,
        "events.tagger.model": model,
    }
    tuned_cfg = overrides_module.apply_overrides(cfg, overrides)
    expected_cache_key = cache_module.cache_key(tuned_cfg)

    state = ledger_module.git_state()
    git_sha = state["sha"]
    dirty = bool(state.get("dirty", False))

    note = (
        "The kernel builds the working tree as bundled (see bundle_sha256), "
        + ("including uncommitted changes." if dirty else "which matches the commit sha.")
    )

    push_result = push_cache_kernel(
        window=window,
        hop=hop,
        model=model,
        git_sha=git_sha,
        client=client,
        **kw,
    )

    k_slug = push_result["kernel"]
    d_slug = dataset_slug()

    return {
        "kernel": k_slug,
        "dataset": d_slug,
        "expected_cache_key": expected_cache_key,
        "git_sha": git_sha,
        "dirty": dirty,
        "window_sec": window,
        "hop_sec": hop,
        "tagger_model": model,
        "bundle_sha256": push_result["params"]["bundle_sha256"],
        "status": "pushed",
        "note": note,
    }
