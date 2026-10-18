"""Local startup diagnostics and extension installation — OWNER: Person 3."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def prepare_local_path() -> None:
    """Find tools installed beside this Python without requiring activation."""
    binary_dir = str(Path(sys.executable).parent)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if binary_dir not in entries:
        os.environ["PATH"] = os.pathsep.join([binary_dir, *entries])


def doctor() -> int:
    prepare_local_path()
    ready = True
    for module, installation in (
        ("fastapi", "python -m pip install -e '.[server]'"),
        ("uvicorn", "python -m pip install -e '.[server]'"),
        ("multipart", "python -m pip install -e '.[server]'"),
        ("yt_dlp", "python -m pip install 'yt-dlp[default]'"),
    ):
        available = importlib.util.find_spec(module) is not None
        ready = ready and available
        print(f"{'OK' if available else 'MISSING'} {module}" + ("" if available else f" — {installation}"))
    for tool in ("ffmpeg", "ffprobe", "node"):
        available = shutil.which(tool) is not None
        ready = ready and available
        print(f"{'OK' if available else 'MISSING'} {tool}" + ("" if available else " — install it on the server PATH; see extension/README.md"))
    if os.environ.get("SOUNDTHING_YOUTUBE_COOKIES"):
        try:
            from .youtube import _cookie_arguments
            with _cookie_arguments():
                pass
            print("OK configured YouTube cookie export (contents not displayed)")
        except (OSError, RuntimeError) as exc:
            ready = False
            print(f"INVALID cookie configuration — {exc}")
    else:
        print("INFO no cookie export configured; use serve --cookies PATH if YouTube requests authentication")
    print("Local setup is ready." if ready else "Resolve the missing/invalid items before processing videos.")
    return 0 if ready else 1


def windows_extension_destination() -> Path:
    """Ask Windows for its user profile; don't assume its name matches WSL."""
    if os.name == "nt" and os.environ.get("USERPROFILE"):
        return Path(os.environ["USERPROFILE"]) / "Downloads" / "soundthing-extension"
    try:
        result = subprocess.run(["cmd.exe", "/c", "echo", "%USERPROFILE%"], capture_output=True, text=True, check=True, timeout=10)
        windows_profile = result.stdout.strip()
        if not windows_profile or "%USERPROFILE%" in windows_profile:
            raise ValueError("Windows user profile is unavailable")
        result = subprocess.run(["wslpath", "-u", windows_profile], capture_output=True, text=True, check=True, timeout=10)
        return Path(result.stdout.strip()) / "Downloads" / "soundthing-extension"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Pass --destination with the Chrome extension folder, for example /mnt/c/Users/YOUR_NAME/Downloads/soundthing-extension") from exc


def sync_extension(destination: str | Path | None, *, source: Path | None = None) -> Path:
    source = (source or Path(__file__).resolve().parents[3] / "extension").resolve()
    if not (source / "manifest.json").is_file():
        raise ValueError("Extension source files are unavailable; run this command from an editable source checkout")
    target = (Path(destination).expanduser() if destination is not None else windows_extension_destination()).resolve()
    if target == source or target in source.parents or source in target.parents:
        raise ValueError("The destination must be separate from the extension source and its parent folders")
    if target.exists() and any(target.iterdir()):
        try:
            installed = json.loads((target / "manifest.json").read_text())
            original = json.loads((source / "manifest.json").read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("Choose an empty folder or an existing Soundthing extension installation") from exc
        if installed.get("name") != original.get("name"):
            raise ValueError("Destination contains a different extension; choose a Soundthing installation folder")
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if any(part.startswith(".") or part in {"tests", "__pycache__"} for part in relative.parts):
            continue
        if path.is_file() and path.suffix != ".pyc" and path.name != "dev.sh":
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    # Retired library assets must also disappear from existing unpacked installs.
    # Limit cleanup to these known files; preserve users' other destination files.
    for name in ("index.html", "app.js", "styles.css", "tokens.css"):
        retired = target / name
        if retired.is_file() or retired.is_symlink():
            retired.unlink()
    print("Extension files synced. In chrome://extensions click Reload, then refresh your YouTube tabs.")
    return target
