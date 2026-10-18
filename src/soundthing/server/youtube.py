"""Local YouTube acquisition and serialized jobs — OWNER: Person 3.

Only video IDs enter this module. yt-dlp receives a canonical watch URL, never a
caller-supplied URL, command option, cookie source, or output path.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from ..pipeline import MODES, run_video
from .. import contracts as C
from ..direction import ensure_direction

VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
COOKIE_ENV = "SOUNDTHING_YOUTUBE_COOKIES"


class CacheBusyError(RuntimeError):
    pass


def youtube_id_from_url(value: str) -> str:
    """Parse a supported YouTube link or ID without fetching any remote URL."""
    value = value.strip()
    if VIDEO_ID.fullmatch(value):
        return value
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or parsed.username or parsed.password
            or parsed.port is not None):
        raise ValueError("Provide a YouTube video URL or its 11-character ID.")
    parts = parsed.path.strip("/").split("/")
    youtube_id = None
    if parsed.hostname == "youtu.be":
        youtube_id = parts[0]
    elif parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path.rstrip("/") == "/watch":
            youtube_id = parse_qs(parsed.query).get("v", [None])[0]
        elif len(parts) == 2 and parts[0] in {"shorts", "live", "embed"}:
            youtube_id = parts[1]
    elif parsed.hostname == "studio.youtube.com" and len(parts) >= 2 and parts[0] == "video":
        youtube_id = parts[1]
    if not youtube_id or not VIDEO_ID.fullmatch(youtube_id):
        raise ValueError("Provide a YouTube video URL or its 11-character ID.")
    return youtube_id


def audio_format(metadata: dict) -> str:
    """Prefer six channels only within the selected/default soundtrack language."""
    selected = metadata.get("requested_formats", [])
    default = next((f for f in selected if f.get("acodec") not in {None, "none"}), metadata)
    language = default.get("language")
    audio_languages = {f.get("language") for f in metadata.get("formats", [])
                       if f.get("acodec") not in {None, "none"}}
    if language is None and len(audio_languages) == 1:
        language = next(iter(audio_languages))
    if language is not None or len(audio_languages) <= 1:
        surround = [f for f in metadata.get("formats", [])
                    if f.get("audio_channels") == 6 and f.get("language") == language
                    and f.get("vcodec") == "none" and f.get("acodec") not in {None, "none"}
                    and re.fullmatch(r"[A-Za-z0-9_-]+", f.get("format_id", ""))]
        if surround:
            best = max(surround, key=lambda f: f.get("abr") or f.get("tbr") or 0)
            return f"bv[height<=480]+{best['format_id']}/bv+{best['format_id']}"
    return "bv[height<=480]+ba[audio_channels<=?2]/b[height<=480]/b"


@contextmanager
def _cookie_arguments():
    """Use a private, disposable jar; yt-dlp may update it during a download.

    Only the server operator can configure the source. The exported original is
    never modified, and neither its contents nor its path enter HTTP responses.
    """
    configured = os.environ.get(COOKIE_ENV)
    if not configured:
        yield []
        return
    try:
        source = Path(configured).expanduser().open("rb")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{COOKIE_ENV} must point to an existing, readable cookie file. See extension/README.md.") from exc
    with source:
        header = source.readline(4096)
        if not header.rstrip(b"\r\n").startswith((b"# Netscape HTTP Cookie File", b"# HTTP Cookie File")):
            raise RuntimeError(f"{COOKIE_ENV} must be an exported Netscape-format cookie file. See extension/README.md.")
        with tempfile.TemporaryDirectory(prefix="soundthing-youtube-auth-") as folder:
            working = Path(folder) / "cookies.txt"
            with working.open("xb") as output:
                working.chmod(0o600)
                output.write(header)
                shutil.copyfileobj(source, output)
            yield ["--cookies", str(working)]


def complete_run(run: Path) -> bool:
    if not all((run / f"captions.{mode}.vtt").is_file() for mode in MODES):
        return False
    try:
        C.load_inputs(run / "inputs.json")
        for mode in MODES:
            C.load_events(run / f"events.{mode}.json")
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return True


def _invoke(arguments: list[str], timeout: int) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist",
             "--no-progress", "--js-runtimes", "node", *arguments],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        return result.stdout
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("YouTube download timed out. Retry this video later.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or "YouTube rejected the download."
        if "not a bot" in stderr.lower():
            if "--cookies" in arguments:
                raise RuntimeError("YouTube still requires browser verification. Refresh your exported YouTube cookies and retry. See extension/README.md (YouTube authentication).") from exc
            raise RuntimeError(f"YouTube requires browser authentication. Export YouTube cookies locally and set {COOKIE_ENV} before restarting the server. See extension/README.md (YouTube authentication).") from exc
        if "--cookies" in arguments:
            # Cookie-jar parse errors can quote whole cookie rows in stderr.
            # Never pass authenticated downloader stderr through to the browser.
            raise RuntimeError("yt-dlp couldn't download this video using the configured cookies. Check video access, refresh the exported YouTube cookies, and update yt-dlp if needed. See extension/README.md.") from exc
        detail = stderr.strip()[-1200:]
        raise RuntimeError(f"yt-dlp: {detail}") from exc


def download_video(youtube_id: str, uploads: Path) -> Path:
    if not VIDEO_ID.fullmatch(youtube_id):
        raise ValueError("Invalid YouTube video ID")
    if importlib.util.find_spec("yt_dlp") is None:
        raise RuntimeError("Install the downloader in the server environment: python -m pip install 'yt-dlp[default]'")
    for tool in ("ffmpeg", "ffprobe", "node"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is missing from the server PATH. See extension/README.md for setup.")
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    with _cookie_arguments() as auth:
        metadata = json.loads(_invoke([*auth, "--skip-download", "--dump-single-json", "--", url], 60))
        if metadata.get("is_live") or metadata.get("is_upcoming") or metadata.get("live_status") in {"is_live", "is_upcoming", "post_live"}:
            raise RuntimeError("Live and upcoming streams aren't supported. Choose a published video.")
        if metadata.get("id") != youtube_id or not metadata.get("duration"):
            raise RuntimeError("YouTube didn't return a playable, finite video.")
        # New acquisition location avoids reusing older downloads that may have
        # selected stereo despite an available 5.1 track. Old originals stay intact.
        selection = audio_format(metadata)
        selection_key = hashlib.sha256(selection.encode()).hexdigest()[:12]
        folder = (uploads / "youtube" / youtube_id / "channel-preserved" / selection_key).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        output = _invoke([
            *auth, "--format", selection,
            "--merge-output-format", "mkv", "--no-overwrites",
            "--output", str(folder / f"youtube_{youtube_id}.%(ext)s"),
            "--print", "after_move:filepath", "--no-simulate", "--", url,
        ], 900)
    lines = output.strip().splitlines()
    path = Path(lines[-1]).resolve() if lines else folder
    if path.parent != folder or path.stem != f"youtube_{youtube_id}" or not path.is_file():
        raise RuntimeError("The downloader didn't produce the expected video file.")
    return path


class YouTubeJobs:
    def __init__(self, runs, uploads, cfg, status, *, downloader=download_video, processor=run_video):
        self.runs, self.uploads, self.cfg, self.status = runs, uploads, cfg, status
        self.downloader, self.processor = downloader, processor
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="soundthing-youtube")
        self.lock = threading.Lock()
        self.direction_pending = set()

    def clear_cache(self, youtube_id: str) -> dict:
        """Clear one idle video's results/downloads, serialized against submission."""
        if not VIDEO_ID.fullmatch(youtube_id):
            raise ValueError("Invalid YouTube video ID")
        video_id = f"youtube_{youtube_id}"
        with self.lock:
            active = self.status.get(video_id, {}).get("status")
            if active in {"queued", "downloading", "processing"} or video_id in self.direction_pending:
                raise CacheBusyError("This video is still processing. Wait for it to finish, then clear its cache.")
            targets = [(self.runs / video_id, self.runs, "results"),
                       (self.uploads / "youtube" / youtube_id, self.uploads, "downloads")]
            # Older acquisition paths may hold the same video's original directly.
            targets.extend((self.uploads / f"{video_id}.{ext}", self.uploads, "downloads")
                           for ext in ("mp4", "mkv", "webm", "mov"))
            for path, root, _ in targets:
                if not path.parent.resolve().is_relative_to(root.resolve()):
                    raise ValueError("The video cache is outside the configured storage directory.")
            removed = []
            self.status.pop(video_id, None)
            for path, _, kind in targets:
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                else:
                    continue
                if kind not in removed:
                    removed.append(kind)
            return {"video_id": video_id, "cleared": True, "removed": removed}

    def queue_direction(self, video_id: str) -> None:
        """Backfill older runs on the same serialized executor without blocking captions."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            return
        run = self.runs / video_id
        with self.lock:
            if video_id in self.direction_pending or (run / "direction.json").is_file() or not complete_run(run):
                return
            self.direction_pending.add(video_id)
            try:
                self.executor.submit(self._direction_run, video_id)
            except RuntimeError:
                self.direction_pending.discard(video_id)

    def _direction_run(self, video_id: str) -> None:
        try:
            run = self.runs / video_id
            inputs = C.load_inputs(run / "inputs.json")
            source = inputs.video_path
            youtube_id = video_id.removeprefix("youtube_")
            if video_id.startswith("youtube_") and VIDEO_ID.fullmatch(youtube_id):
                try:
                    source = self.downloader(youtube_id, self.uploads)
                except Exception:
                    logging.getLogger(__name__).warning("Could not reacquire direction audio; analyzing cached original")
            ensure_direction(source, run, inputs.duration)
        except Exception:
            logging.getLogger(__name__).exception("Could not backfill direction for %s", video_id)
        finally:
            with self.lock:
                self.direction_pending.discard(video_id)

    def submit(self, youtube_id: str) -> str:
        if not VIDEO_ID.fullmatch(youtube_id):
            raise ValueError("Invalid YouTube video ID")
        video_id = f"youtube_{youtube_id}"
        with self.lock:
            current = self.status.get(video_id, {}).get("status")
            if current in {"queued", "downloading", "processing"}:
                return video_id
            # An errored job must be retried, even if a previous run left artifacts.
            if current != "error" and complete_run(self.runs / video_id):
                self.status[video_id] = {"status": "done"}
                return video_id
            self.status[video_id] = {"status": "queued"}
            try:
                self.executor.submit(self._run, youtube_id, video_id)
            except RuntimeError:
                self.status[video_id] = {"status": "error", "error": "The server is shutting down. Restart it and retry."}
                raise
        return video_id

    def _run(self, youtube_id: str, video_id: str) -> None:
        try:
            self.status[video_id] = {"status": "downloading"}
            source = self.downloader(youtube_id, self.uploads)
            self.status[video_id] = {"status": "processing"}
            self.processor(source, self.runs, self.cfg)
            if not complete_run(self.runs / video_id):
                raise RuntimeError("Processing finished without both caption tracks. Retry the video.")
            self.status[video_id] = {"status": "done"}
        except Exception as exc:
            self.status[video_id] = {"status": "error", "error": str(exc)}

    def close(self) -> None:
        self.executor.shutdown(wait=True)
