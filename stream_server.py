"""Streaming transcription server for the tsumugi UI.

Unlike ``app.py`` (batch: upload everything, then poll a status list), this
server answers ``POST /api/transcribe`` with a Server-Sent Events stream of the
transcription as it happens: pipeline stages, per-stem window progress, and the
notes themselves as the windowed decoder commits them.

Run it with the project's Python, then open the built frontend:

    python stream_server.py            # http://127.0.0.1:8000

The frontend lives in ``tsumugi-ui/`` (Vite + React + TypeScript). Build it with
``npm install && npm run build``; ``tsumugi-ui/dist`` is served from ``/``.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

# Keep the CUDA allocator from fragmenting before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from instrument_agnostic_amt.streaming.pipeline import (
    StreamOptions,
    format_error,
    run_streaming_transcription,
)

OUTPUT_ROOT = Path(os.environ.get("TSUMUGI_OUTPUT_ROOT", "stream_outputs"))
UPLOAD_ROOT = OUTPUT_ROOT / "uploads"
WEB_DIST = Path(__file__).parent / "tsumugi-ui" / "dist"
ALLOWED_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".webm",
}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

app = FastAPI(title="tsumugi streaming transcription")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_job_lock = threading.Lock()
_current_job: str | None = None


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _try_acquire_job(job_id: str) -> bool:
    global _current_job
    with _job_lock:
        if _current_job is not None:
            return False
        _current_job = job_id
        return True


def _release_job(job_id: str) -> None:
    global _current_job
    with _job_lock:
        if _current_job == job_id:
            _current_job = None


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "busy": _current_job is not None,
        "output_root": str(OUTPUT_ROOT.resolve()),
        "frontend_built": WEB_DIST.exists(),
    }


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                handle.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="音频文件过大（上限 512 MB）")
            handle.write(chunk)
    return written


def _download_youtube(url: str, destination_dir: Path, emit) -> Path:
    import yt_dlp

    destination_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(destination_dir / "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        path = Path(downloader.prepare_filename(info))
    for candidate in (path, path.with_suffix(".wav")):
        if candidate.exists():
            emit(
                {
                    "type": "stage",
                    "key": "download",
                    "label": "获取云端音源",
                    "status": "done",
                    "detail": candidate.name,
                }
            )
            return candidate
    raise RuntimeError("下载完成但没有找到音频文件")


@app.post("/api/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    window_batch_size: int = Form(default=1),
) -> StreamingResponse:
    if file is None and not url:
        raise HTTPException(status_code=400, detail="请上传音频文件或提供视频链接")
    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix and suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"不支持的音频格式: {suffix}")

    job_id = uuid.uuid4().hex[:12]
    if not _try_acquire_job(job_id):
        raise HTTPException(
            status_code=503,
            detail="服务器正在处理另一个转录任务，请稍后重试",
        )

    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        if file is not None:
            destination = UPLOAD_ROOT / f"{job_id}_{Path(file.filename or 'audio').name}"
            await _save_upload(file, destination)
            audio_path = destination
        else:
            audio_path = None  # resolved inside the worker after the download stage
    except Exception:
        _release_job(job_id)
        raise

    events: queue.Queue[dict | None] = queue.Queue()
    stop = threading.Event()

    def worker() -> None:
        try:
            source = audio_path
            if source is None:
                events.put(
                    {
                        "type": "stage",
                        "key": "download",
                        "label": "获取云端音源",
                        "status": "running",
                        "detail": url,
                    }
                )
                source = _download_youtube(str(url), job_dir / "download", events.put)
            events.put(
                {
                    "type": "accepted",
                    "job_id": job_id,
                    "audio": Path(source).name,
                    "midi_url": f"/api/jobs/{job_id}/midi",
                }
            )
            run_streaming_transcription(
                source,
                events.put,
                job_dir=job_dir,
                options=StreamOptions(window_batch_size=max(1, int(window_batch_size))),
                should_cancel=stop.is_set,
            )
        except Exception as exc:  # pragma: no cover - surfaced to the client
            events.put(format_error(exc))
        finally:
            events.put(None)

    threading.Thread(target=worker, name=f"transcribe-{job_id}", daemon=True).start()

    async def event_stream():
        try:
            while True:
                if await request.is_disconnected():
                    stop.set()
                    break
                try:
                    item = await asyncio.to_thread(events.get, True, 1.0)
                except queue.Empty:
                    # Keeps the loop responsive so a disconnect is noticed early.
                    continue
                if item is None:
                    break
                yield _sse(item)
        except asyncio.CancelledError:
            stop.set()
            raise
        finally:
            stop.set()
            _release_job(job_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/midi")
async def download_midi(job_id: str) -> FileResponse:
    export_dir = OUTPUT_ROOT / job_id / "export"
    if not export_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在或尚未完成")
    files = sorted(export_dir.glob("*.mid"))
    if not files:
        raise HTTPException(status_code=404, detail="MIDI 尚未生成")
    path = files[0]
    return FileResponse(
        path,
        filename=path.name,
        media_type="audio/midi",
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> JSONResponse:
    job_dir = OUTPUT_ROOT / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    return JSONResponse({"deleted": job_id})


if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
else:

    @app.get("/")
    async def missing_frontend() -> JSONResponse:
        return JSONResponse(
            {
                "detail": "前端尚未构建",
                "hint": "cd web && npm install && npm run build",
            },
            status_code=503,
        )


if __name__ == "__main__":
    import uvicorn

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Streaming server on http://127.0.0.1:8000 (outputs in {OUTPUT_ROOT.resolve()})")
    uvicorn.run(app, host="127.0.0.1", port=8000)
