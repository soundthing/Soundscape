"""FastAPI demo backend — OWNER: Person 3.

POST /api/videos                    upload a video; processing starts in background -> {video_id}
POST /api/youtube                   download/process a public or unlisted video -> {video_id}
GET  /api/videos/{id}               {status, demo, error?}; includes downloading
GET  /api/videos/{id}/video         the uploaded video
GET  /api/videos/{id}/captions/{mode}.vtt   mode = baseline | conditioned
GET  /api/videos/{id}/env_segments  env_segments.json payload (for an env timeline overlay)
GET  /                              demo player
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict
import shutil
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import contracts as C
from ..config import load_config
from ..extract import video_id_for
from ..pipeline import MODES, run_video
from .youtube import YouTubeJobs, complete_run

RUNS = Path(os.environ.get("SOUNDTHING_RUNS", "runs"))
UPLOADS = Path(os.environ.get("SOUNDTHING_UPLOADS", "uploads"))
STATIC = Path(__file__).parent / "static"
ALLOWED = {".mp4", ".mov", ".webm"}

cfg = load_config(os.environ.get("SOUNDTHING_CONFIG"))
status: dict[str, dict] = {}
youtube_jobs = YouTubeJobs(RUNS, UPLOADS, cfg, status)


@asynccontextmanager
async def lifespan(app):
    yield
    youtube_jobs.close()


app = FastAPI(title="soundthing demo", lifespan=lifespan)


class YouTubeRequest(BaseModel):
    youtube_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")


@app.post("/api/youtube", status_code=202)
def submit_youtube(request: YouTubeRequest):
    return {"video_id": youtube_jobs.submit(request.youtube_id)}


def _process(video_path: Path, video_id: str) -> None:
    try:
        status[video_id] = {"status": "processing"}
        run_video(video_path, RUNS, cfg)
        status[video_id] = {"status": "done"}
    except Exception as e:  # surfaced to the UI
        traceback.print_exc()
        status[video_id] = {"status": "error", "error": str(e)}


@app.post("/api/videos")
async def upload(file: UploadFile, background: BackgroundTasks):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"unsupported file type {suffix!r}")
    video_id = video_id_for(file.filename)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / f"{video_id}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    status[video_id] = {"status": "queued"}
    background.add_task(_process, dest, video_id)
    return {"video_id": video_id}


@app.get("/api/videos/{video_id}")
def get_status(video_id: str):
    demo = any(cfg.pipeline[stage]["backend"] == "stub" for stage in ("events", "environment"))
    if video_id in status:
        return {**status[video_id], "demo": demo}
    if complete_run(RUNS / video_id):
        return {"status": "done", "demo": demo}
    raise HTTPException(404)


@app.get("/api/videos/{video_id}/video")
def get_video(video_id: str):
    path = RUNS / video_id / "inputs.json"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(C.load_inputs(path).video_path)


@app.get("/api/videos/{video_id}/direction")
def get_direction(video_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise HTTPException(404)
    path = RUNS / video_id / "direction.json"
    if not path.is_file():
        youtube_jobs.queue_direction(video_id)
        raise HTTPException(404)
    try:
        return asdict(C.load_direction(path))
    except (OSError, ValueError, TypeError, KeyError):
        raise HTTPException(500, "Direction track is invalid")


@app.get("/api/videos/{video_id}/captions/{mode}.vtt")
def get_captions(video_id: str, mode: str):
    path = RUNS / video_id / f"captions.{mode}.vtt"
    if mode not in MODES or not path.exists() or not complete_run(RUNS / video_id):
        raise HTTPException(404)
    return FileResponse(path, media_type="text/vtt")


@app.get("/api/videos/{video_id}/env_segments")
def get_env_segments(video_id: str):
    path = RUNS / video_id / "env_segments.json"
    if not path.exists():
        raise HTTPException(404)
    return JSONResponse([s.__dict__ for s in C.load_segments(path)])


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
