"""FastAPI demo backend — OWNER: Person 3.

POST /api/videos                    upload a video; processing starts in background -> {video_id}
GET  /api/videos/{id}               {status, error?}
GET  /api/videos/{id}/video         the uploaded video
GET  /api/videos/{id}/captions/{mode}.vtt   mode = baseline | conditioned
GET  /api/videos/{id}/env_segments  env_segments.json payload (for an env timeline overlay)
GET  /                              demo player
"""

from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .. import contracts as C
from ..config import load_config
from ..extract import video_id_for
from ..pipeline import MODES, run_video

RUNS = Path(os.environ.get("SOUNDTHING_RUNS", "runs"))
UPLOADS = Path(os.environ.get("SOUNDTHING_UPLOADS", "uploads"))
STATIC = Path(__file__).parent / "static"
ALLOWED = {".mp4", ".mov", ".webm"}

app = FastAPI(title="soundthing demo")
cfg = load_config(os.environ.get("SOUNDTHING_CONFIG"))
status: dict[str, dict] = {}


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
    if video_id in status:
        return status[video_id]
    if (RUNS / video_id / "inputs.json").exists():
        return {"status": "done"}
    raise HTTPException(404)


@app.get("/api/videos/{video_id}/video")
def get_video(video_id: str):
    path = RUNS / video_id / "inputs.json"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(C.load_inputs(path).video_path)


@app.get("/api/videos/{video_id}/captions/{mode}.vtt")
def get_captions(video_id: str, mode: str):
    path = RUNS / video_id / f"captions.{mode}.vtt"
    if mode not in MODES or not path.exists():
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
