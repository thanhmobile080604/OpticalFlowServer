import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import hashlib
import shutil
import os
import uuid
import tempfile
import threading
import time
import math
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
from src.inference import NormalizedPoint, NormalizedRoi, OpticalFlowProcessor, ProcessingCancelled

LOG_LEVEL = os.getenv("OPTICAL_FLOW_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("optical_flow.server")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logging.getLogger("optical_flow").setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI(title="Optical Flow Server")

# Allow CORS for potential web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model processor
# Models are stored in the repository-level `AI_model` directory
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "AI_model"
DEFAULT_MODEL = MODEL_DIR / "optical_flow_estimation_raft_2023aug_int8bq.onnx"
DEQUANT_MODEL = MODEL_DIR / "optical_flow_estimation_raft_2023aug_dequant.onnx"
ALT_MODEL = MODEL_DIR / "optical_flow_estimation_raft_2023aug.onnx"
# Prefer dequantized float model, then alternate float model, then default int8 model
if DEQUANT_MODEL.exists():
    MODEL_PATH = str(DEQUANT_MODEL)
elif ALT_MODEL.exists():
    MODEL_PATH = str(ALT_MODEL)
else:
    MODEL_PATH = str(DEFAULT_MODEL)
try:
    logger.info("Loading optical-flow model path=%s exists=%s", MODEL_PATH, os.path.exists(MODEL_PATH))
    processor = OpticalFlowProcessor(MODEL_PATH)
    providers = processor.session.get_providers() if getattr(processor, "session", None) else []
    logger.info("Successfully loaded model path=%s providers=%s", MODEL_PATH, providers)
except Exception as e:
    logger.exception("Failed to load model path=%s error=%s", MODEL_PATH, e)
    processor = None

@dataclass
class VideoJob:
    job_id: str
    status: str
    input_path: str
    output_path: str
    mode: str
    is_moving: bool
    roi: Optional[NormalizedRoi] = None
    progress: int = 0
    error: Optional[str] = None
    cancel_requested: bool = False


@dataclass
class VideoUpload:
    upload_id: str
    upload_dir: str
    file_name: str
    file_size: int
    chunk_size: int
    total_chunks: int
    created_at: float
    received_chunks: set


video_jobs = {}
video_jobs_lock = threading.Lock()
MAX_CONCURRENT_VIDEO_JOBS = max(1, int(os.getenv("OPTICAL_FLOW_MAX_CONCURRENT_VIDEO_JOBS", "1")))
MAX_PENDING_VIDEO_JOBS = max(
    MAX_CONCURRENT_VIDEO_JOBS,
    int(os.getenv("OPTICAL_FLOW_MAX_PENDING_VIDEO_JOBS", "8")),
)
video_job_slots = threading.Semaphore(MAX_CONCURRENT_VIDEO_JOBS)

video_uploads = {}
video_uploads_lock = threading.Lock()
MAX_UPLOAD_CHUNK_BYTES = max(
    1 * 1024 * 1024,
    int(os.getenv("OPTICAL_FLOW_MAX_UPLOAD_CHUNK_BYTES", str(90 * 1024 * 1024))),
)
MAX_PENDING_VIDEO_UPLOADS = max(1, int(os.getenv("OPTICAL_FLOW_MAX_PENDING_VIDEO_UPLOADS", "8")))
VIDEO_UPLOAD_TTL_SECONDS = max(60, int(os.getenv("OPTICAL_FLOW_UPLOAD_TTL_SECONDS", "3600")))
RESULT_DOWNLOAD_CHUNK_BYTES = max(
    1 * 1024 * 1024,
    int(os.getenv("OPTICAL_FLOW_RESULT_CHUNK_BYTES", str(32 * 1024 * 1024))),
)


def cleanup_files(file_paths):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug("Cleaned up temp file path=%s", path)
        except Exception as e:
            logger.warning("Failed to cleanup temp file path=%s error=%s", path, e)



def cleanup_directory(path):
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            logger.debug("Cleaned up temp directory path=%s", path)
    except Exception as e:
        logger.warning("Failed to cleanup temp directory path=%s error=%s", path, e)


def set_video_job(job_id: str, **updates):
    with video_jobs_lock:
        job = video_jobs.get(job_id)
        if not job:
            return
        for key, value in updates.items():
            setattr(job, key, value)


def get_video_job(job_id: str) -> Optional[VideoJob]:
    with video_jobs_lock:
        return video_jobs.get(job_id)


def remove_video_job(job_id: str):
    with video_jobs_lock:
        video_jobs.pop(job_id, None)


def video_job_counts():
    with video_jobs_lock:
        jobs = list(video_jobs.values())
    queued = sum(1 for job in jobs if job.status == "queued")
    processing = sum(1 for job in jobs if job.status in ("processing", "cancelling"))
    completed = sum(1 for job in jobs if job.status == "completed")
    failed = sum(1 for job in jobs if job.status == "failed")
    cancelled = sum(1 for job in jobs if job.status == "cancelled")
    pending = queued + processing
    return {
        "queued": queued,
        "processing": processing,
        "pending": pending,
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "total": len(jobs),
        "max_concurrent": MAX_CONCURRENT_VIDEO_JOBS,
        "max_pending": MAX_PENDING_VIDEO_JOBS,
    }
def get_video_upload(upload_id: str) -> Optional[VideoUpload]:
    with video_uploads_lock:
        return video_uploads.get(upload_id)


def remove_video_upload(upload_id: str, cleanup: bool = True) -> Optional[VideoUpload]:
    with video_uploads_lock:
        upload = video_uploads.pop(upload_id, None)
    if cleanup and upload:
        cleanup_directory(upload.upload_dir)
    return upload


def video_upload_counts():
    with video_uploads_lock:
        uploads = list(video_uploads.values())
    return {
        "pending": len(uploads),
        "max_pending": MAX_PENDING_VIDEO_UPLOADS,
        "max_chunk_bytes": MAX_UPLOAD_CHUNK_BYTES,
        "ttl_seconds": VIDEO_UPLOAD_TTL_SECONDS,
    }


def cleanup_stale_video_uploads():
    now = time.time()
    stale_uploads = []
    with video_uploads_lock:
        for upload_id, upload in list(video_uploads.items()):
            if now - upload.created_at > VIDEO_UPLOAD_TTL_SECONDS:
                stale_uploads.append(video_uploads.pop(upload_id))
    for upload in stale_uploads:
        logger.info("Cleaning stale video upload upload_id=%s upload_dir=%s", upload.upload_id, upload.upload_dir)
        cleanup_directory(upload.upload_dir)


def chunk_path(upload: VideoUpload, chunk_index: int) -> str:
    return os.path.join(upload.upload_dir, f"chunk_{chunk_index:06d}.part")


def upload_payload(upload: VideoUpload):
    return {
        "upload_id": upload.upload_id,
        "status": "uploading",
        "file_name": upload.file_name,
        "file_size": upload.file_size,
        "chunk_size": upload.chunk_size,
        "received_chunks": len(upload.received_chunks),
        "total_chunks": upload.total_chunks,
    }


def completed_video_job_or_error(job_id: str) -> VideoJob:
    job = get_video_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status == "failed":
        logger.warning("Video job result requested but job failed job_id=%s error=%s", job_id, job.error)
        raise HTTPException(status_code=500, detail=job.error or "Video processing failed.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is not completed yet: {job.status}.")
    if not os.path.exists(job.output_path):
        logger.warning("Video job result missing output file job_id=%s output_path=%s", job_id, job.output_path)
        raise HTTPException(status_code=410, detail="Processed video is no longer available.")
    return job


def video_result_info(job: VideoJob):
    output_size = os.path.getsize(job.output_path)
    total_chunks = (output_size + RESULT_DOWNLOAD_CHUNK_BYTES - 1) // RESULT_DOWNLOAD_CHUNK_BYTES
    return {
        "job_id": job.job_id,
        "file_name": os.path.basename(job.output_path),
        "file_size": output_size,
        "chunk_size": RESULT_DOWNLOAD_CHUNK_BYTES,
        "total_chunks": total_chunks,
    }


def iter_file_chunk(path: str, offset: int, byte_count: int):
    with open(path, "rb") as file_obj:
        file_obj.seek(offset)
        remaining = byte_count
        while remaining > 0:
            data = file_obj.read(min(1024 * 1024, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def cleanup_video_job_result(job_id: str) -> Optional[VideoJob]:
    job = get_video_job(job_id)
    if not job:
        return None
    cleanup_files([job.input_path, job.output_path])
    remove_video_job(job_id)
    return job


def reject_if_video_job_queue_full(filename: str):
    counts = video_job_counts()
    if counts["pending"] >= MAX_PENDING_VIDEO_JOBS:
        logger.warning(
            "Async process-video job rejected because queue is full filename=%s pending=%s max_pending=%s",
            filename,
            counts["pending"],
            MAX_PENDING_VIDEO_JOBS,
        )
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many video jobs queued or processing "
                f"({counts['pending']}/{MAX_PENDING_VIDEO_JOBS})."
            ),
        )


def enqueue_video_job(
    background_tasks: BackgroundTasks,
    input_path: str,
    filename: str,
    mode_name: str,
    resolved_is_moving: bool,
    roi: Optional[NormalizedRoi] = None,
):
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_temp.name
    output_temp.close()

    job_id = uuid.uuid4().hex
    job = VideoJob(
        job_id=job_id,
        status="queued",
        input_path=input_path,
        output_path=output_path,
        mode=mode_name,
        is_moving=resolved_is_moving,
        roi=roi,
        progress=0,
    )
    with video_jobs_lock:
        video_jobs[job_id] = job

    input_size = os.path.getsize(input_path) if os.path.exists(input_path) else -1
    logger.info(
        "Video job queued job_id=%s filename=%s mode=%s resolved_is_moving=%s roi=%s input_path=%s input_size_bytes=%s output_path=%s",
        job_id,
        filename,
        mode_name,
        resolved_is_moving,
        roi_log_payload(roi),
        input_path,
        input_size,
        output_path,
    )
    background_tasks.add_task(run_video_job, job_id)
    return {
        "job_id": job_id,
        "status": job.status,
        "queue": video_job_counts(),
    }


def is_video_job_cancel_requested(job_id: str) -> bool:
    with video_jobs_lock:
        job = video_jobs.get(job_id)
        return bool(job and job.cancel_requested)


def request_video_job_cancel(job_id: str) -> Optional[VideoJob]:
    with video_jobs_lock:
        job = video_jobs.get(job_id)
        if not job:
            return None
        job.cancel_requested = True
        if job.status in ("queued", "completed"):
            job.status = "cancelled"
            job.error = "Cancelled by client."
        elif job.status == "processing":
            job.status = "cancelling"
            job.error = "Cancellation requested by client."
        return job


def resolve_is_moving(is_moving: Optional[bool], isMoving: Optional[bool]) -> bool:
    if is_moving is not None:
        return bool(is_moving)
    if isMoving is not None:
        return bool(isMoving)
    return False


def vector_direction_sign_for_motion(is_moving: bool) -> float:
    return 1.0 if is_moving else -1.0


def finite_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_roi_path_points(raw_path_points: Optional[str]):
    points = []
    if not raw_path_points:
        return points
    for raw_point in raw_path_points.split(";"):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        parts = raw_point.split(",", 1)
        if len(parts) != 2:
            continue
        x = finite_float(parts[0])
        y = finite_float(parts[1])
        if x is None or y is None:
            continue
        points.append(NormalizedPoint(x=clamp_unit(x), y=clamp_unit(y)))
    return points


def resolve_roi(
    roi_enabled: Optional[bool],
    roi_left: Optional[float],
    roi_top: Optional[float],
    roi_right: Optional[float],
    roi_bottom: Optional[float],
    roi_view_aspect_ratio: Optional[float],
    roi_path_points: Optional[str],
    roi_selected_position_ms: Optional[int] = None,
) -> Optional[NormalizedRoi]:
    if roi_enabled is False:
        return None

    raw_values = (roi_left, roi_top, roi_right, roi_bottom)
    has_any_roi_value = any(value is not None for value in raw_values)
    if not roi_enabled and not has_any_roi_value:
        return None
    if any(value is None for value in raw_values):
        if roi_enabled:
            raise HTTPException(status_code=400, detail="ROI is enabled but rectangle coordinates are missing.")
        return None

    left = finite_float(roi_left)
    top = finite_float(roi_top)
    right = finite_float(roi_right)
    bottom = finite_float(roi_bottom)
    if left is None or top is None or right is None or bottom is None:
        raise HTTPException(status_code=400, detail="ROI rectangle coordinates must be finite numbers.")

    left, right = sorted((clamp_unit(left), clamp_unit(right)))
    top, bottom = sorted((clamp_unit(top), clamp_unit(bottom)))
    if right - left <= 1e-6 or bottom - top <= 1e-6:
        raise HTTPException(status_code=400, detail="ROI rectangle must have positive width and height.")

    view_aspect_ratio = finite_float(roi_view_aspect_ratio)
    if view_aspect_ratio is None or view_aspect_ratio <= 0.0:
        view_aspect_ratio = 1.0

    return NormalizedRoi(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        view_aspect_ratio=view_aspect_ratio,
        path_points=parse_roi_path_points(roi_path_points),
        selected_position_ms=max(0, int(roi_selected_position_ms or 0)),
    )


def roi_log_payload(roi: Optional[NormalizedRoi]):
    if roi is None:
        return None
    return {
        "left": round(roi.left, 4),
        "top": round(roi.top, 4),
        "right": round(roi.right, 4),
        "bottom": round(roi.bottom, 4),
        "view_aspect_ratio": round(roi.view_aspect_ratio, 4),
        "path_points": len(roi.path_points),
        "selected_position_ms": getattr(roi, "selected_position_ms", 0),
    }


def acquire_video_job_slot(job_id: str) -> bool:
    while True:
        if is_video_job_cancel_requested(job_id):
            set_video_job(job_id, status="cancelled", error="Cancelled by client.")
            return False
        if video_job_slots.acquire(timeout=1.0):
            return True


def run_video_job(job_id: str):
    job = get_video_job(job_id)
    if not job:
        logger.warning("Video job missing before processing job_id=%s", job_id)
        return
    if not processor:
        set_video_job(job_id, status="failed", error="Model not loaded on server.")
        logger.error("Video job cannot start because model is not loaded job_id=%s", job_id)
        return
    if job.cancel_requested or job.status == "cancelled":
        set_video_job(job_id, status="cancelled", error="Cancelled by client.")
        cleanup_files([job.input_path, job.output_path])
        logger.info("Video job cancelled before processing job_id=%s", job_id)
        return

    slot_acquired = False
    try:
        logger.info(
            "Video job waiting for processing slot job_id=%s pending=%s max_concurrent=%s",
            job_id,
            video_job_counts()["pending"],
            MAX_CONCURRENT_VIDEO_JOBS,
        )
        slot_acquired = acquire_video_job_slot(job_id)
        if not slot_acquired:
            cleanup_files([job.input_path, job.output_path])
            logger.info("Video job cancelled while queued job_id=%s", job_id)
            return

        set_video_job(job_id, status="processing", progress=0)
        vector_direction_sign = vector_direction_sign_for_motion(job.is_moving)

        def update_progress(percent):
            set_video_job(job_id, progress=max(0, min(100, int(percent))))

        input_size = os.path.getsize(job.input_path) if os.path.exists(job.input_path) else -1
        logger.info(
            "Video job started job_id=%s mode=%s is_moving=%s vector_direction_sign=%.1f roi=%s input_path=%s input_size_bytes=%s output_path=%s",
            job_id,
            job.mode,
            job.is_moving,
            vector_direction_sign,
            roi_log_payload(job.roi),
            job.input_path,
            input_size,
            job.output_path,
        )
        processor.process_video(
            job.input_path,
            job.output_path,
            mode=job.mode.upper(),
            vector_direction_sign=vector_direction_sign,
            roi=job.roi,
            req_id=job_id,
            progress_callback=update_progress,
            cancel_callback=lambda: is_video_job_cancel_requested(job_id),
        )
        if is_video_job_cancel_requested(job_id):
            cleanup_files([job.output_path])
            set_video_job(job_id, status="cancelled", error="Cancelled by client.")
            logger.info("Video job cancelled after processing returned job_id=%s", job_id)
            return
        set_video_job(job_id, status="completed", progress=100)
        output_size = os.path.getsize(job.output_path) if os.path.exists(job.output_path) else -1
        logger.info("Video job completed job_id=%s output_path=%s output_size_bytes=%s", job_id, job.output_path, output_size)
    except ProcessingCancelled as e:
        cleanup_files([job.output_path])
        set_video_job(job_id, status="cancelled", error="Cancelled by client.")
        logger.info("Video job cancelled job_id=%s mode=%s message=%s", job_id, job.mode, e)
    except Exception as e:
        cleanup_files([job.output_path])
        set_video_job(job_id, status="failed", error=str(e))
        logger.exception("Video job failed job_id=%s mode=%s error=%s", job_id, job.mode, e)
    finally:
        if slot_acquired:
            video_job_slots.release()
        cleanup_files([job.input_path])


@app.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("VECTORS"),
    is_moving: Optional[bool] = Form(None),
    isMoving: Optional[bool] = Form(None),
    roi_enabled: Optional[bool] = Form(None),
    roi_left: Optional[float] = Form(None),
    roi_top: Optional[float] = Form(None),
    roi_right: Optional[float] = Form(None),
    roi_bottom: Optional[float] = Form(None),
    roi_view_aspect_ratio: Optional[float] = Form(None),
    roi_path_points: Optional[str] = Form(None),
    roi_selected_position_ms: Optional[int] = Form(None),
):
    """
    Process a video using the RAFT Optical Flow model.
    mode: "VECTORS" or "HEATMAP"
    is_moving: true if the camera is moving forward, false if moving backward/stationary (affects vector direction)
    """
    if not processor:
        logger.error("Synchronous process-video rejected because model is not loaded filename=%s", file.filename)
        return {"error": "Model not loaded on server."}

    roi = resolve_roi(
        roi_enabled,
        roi_left,
        roi_top,
        roi_right,
        roi_bottom,
        roi_view_aspect_ratio,
        roi_path_points,
        roi_selected_position_ms,
    )

    # Save uploaded video to system temporary files
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path = input_temp.name
    input_temp.close()
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_temp.name
    output_temp.close()

    resolved_is_moving = resolve_is_moving(is_moving, isMoving)
    vector_direction_sign = vector_direction_sign_for_motion(resolved_is_moving)
    mode_name = mode.upper()
    input_size = os.path.getsize(input_path) if os.path.exists(input_path) else -1
    logger.info(
        "Synchronous video processing started filename=%s mode=%s raw_is_moving=%s raw_isMoving=%s resolved_is_moving=%s vector_direction_sign=%.1f roi=%s input_path=%s input_size_bytes=%s output_path=%s",
        file.filename,
        mode_name,
        is_moving,
        isMoving,
        resolved_is_moving,
        vector_direction_sign,
        roi_log_payload(roi),
        input_path,
        input_size,
        output_path,
    )

    try:
        # Process the video (no status file)
        processor.process_video(input_path, output_path, mode=mode_name, vector_direction_sign=vector_direction_sign, roi=roi)

        # Schedule cleanup after sending response
        background_tasks.add_task(cleanup_files, [input_path, output_path])
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else -1
        logger.info("Synchronous video processing completed output_path=%s output_size_bytes=%s", output_path, output_size)

        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=os.path.basename(output_path)
        )
    except Exception as e:
        # cleanup temp files on error
        cleanup_files([input_path, output_path])
        logger.exception("Synchronous video processing failed filename=%s mode=%s error=%s", file.filename, mode_name, e)
        return {"error": str(e)}


@app.post("/process-video/jobs", status_code=202)
async def create_process_video_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("VECTORS"),
    is_moving: Optional[bool] = Form(None),
    isMoving: Optional[bool] = Form(None),
    roi_enabled: Optional[bool] = Form(None),
    roi_left: Optional[float] = Form(None),
    roi_top: Optional[float] = Form(None),
    roi_right: Optional[float] = Form(None),
    roi_bottom: Optional[float] = Form(None),
    roi_view_aspect_ratio: Optional[float] = Form(None),
    roi_path_points: Optional[str] = Form(None),
    roi_selected_position_ms: Optional[int] = Form(None),
):
    """
    Create an async video-processing job for Cloudflare Tunnel clients.
    This avoids holding a long HTTP request open while RAFT processing runs.
    """
    if not processor:
        logger.error("Async process-video job rejected because model is not loaded filename=%s", file.filename)
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    reject_if_video_job_queue_full(file.filename)

    roi = resolve_roi(
        roi_enabled,
        roi_left,
        roi_top,
        roi_right,
        roi_bottom,
        roi_view_aspect_ratio,
        roi_path_points,
        roi_selected_position_ms,
    )

    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path = input_temp.name
    input_temp.close()
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    mode_name = mode.upper()
    resolved_is_moving = resolve_is_moving(is_moving, isMoving)
    return enqueue_video_job(
        background_tasks=background_tasks,
        input_path=input_path,
        filename=file.filename or "input.mp4",
        mode_name=mode_name,
        resolved_is_moving=resolved_is_moving,
        roi=roi,
    )


@app.post("/process-video/uploads", status_code=201)
def create_process_video_upload(
    file_name: str = Form("input.mp4"),
    file_size: int = Form(...),
    chunk_size: int = Form(...),
    total_chunks: int = Form(...),
):
    cleanup_stale_video_uploads()
    if file_size <= 0:
        raise HTTPException(status_code=400, detail="file_size must be positive.")
    if chunk_size <= 0 or chunk_size > MAX_UPLOAD_CHUNK_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"chunk_size must be between 1 and {MAX_UPLOAD_CHUNK_BYTES} bytes.",
        )
    if total_chunks <= 0:
        raise HTTPException(status_code=400, detail="total_chunks must be positive.")
    if total_chunks != ((file_size + chunk_size - 1) // chunk_size):
        raise HTTPException(status_code=400, detail="total_chunks does not match file_size and chunk_size.")

    with video_uploads_lock:
        if len(video_uploads) >= MAX_PENDING_VIDEO_UPLOADS:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Too many video uploads in progress "
                    f"({len(video_uploads)}/{MAX_PENDING_VIDEO_UPLOADS})."
                ),
            )

    upload_id = uuid.uuid4().hex
    upload_root = os.path.join(tempfile.gettempdir(), "optical_flow_uploads")
    os.makedirs(upload_root, exist_ok=True)
    upload_dir = tempfile.mkdtemp(prefix=f"upload_{upload_id}_", dir=upload_root)
    upload = VideoUpload(
        upload_id=upload_id,
        upload_dir=upload_dir,
        file_name=file_name or "input.mp4",
        file_size=int(file_size),
        chunk_size=int(chunk_size),
        total_chunks=int(total_chunks),
        created_at=time.time(),
        received_chunks=set(),
    )
    with video_uploads_lock:
        video_uploads[upload_id] = upload
    logger.info(
        "Video upload created upload_id=%s file_name=%s file_size=%s chunk_size=%s total_chunks=%s upload_dir=%s",
        upload_id,
        upload.file_name,
        upload.file_size,
        upload.chunk_size,
        upload.total_chunks,
        upload.upload_dir,
    )
    payload = upload_payload(upload)
    payload["limits"] = video_upload_counts()
    return payload


@app.post("/process-video/uploads/{upload_id}/chunks")
async def upload_process_video_chunk(
    upload_id: str,
    chunk: UploadFile = File(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    chunk_sha256: Optional[str] = Form(None),
):
    upload = get_video_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found.")
    if total_chunks != upload.total_chunks:
        raise HTTPException(status_code=400, detail="total_chunks does not match upload session.")
    if chunk_index < 0 or chunk_index >= upload.total_chunks:
        raise HTTPException(status_code=400, detail="chunk_index is out of range.")

    temp_path = os.path.join(upload.upload_dir, f"chunk_{chunk_index:06d}.uploading")
    final_path = chunk_path(upload, chunk_index)
    bytes_written = 0
    hasher = hashlib.sha256() if chunk_sha256 else None
    try:
        with open(temp_path, "wb") as buffer:
            while True:
                data = await chunk.read(1024 * 1024)
                if not data:
                    break
                bytes_written += len(data)
                if bytes_written > MAX_UPLOAD_CHUNK_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Chunk exceeds max size {MAX_UPLOAD_CHUNK_BYTES} bytes.",
                    )
                if hasher is not None:
                    hasher.update(data)
                buffer.write(data)
        if bytes_written <= 0:
            raise HTTPException(status_code=400, detail="Chunk is empty.")
        expected_bytes = min(upload.chunk_size, upload.file_size - (chunk_index * upload.chunk_size))
        if bytes_written != expected_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"Chunk size mismatch: {bytes_written} != {expected_bytes}.",
            )
        if hasher is not None and hasher.hexdigest().lower() != chunk_sha256.lower():
            raise HTTPException(status_code=400, detail="Chunk checksum mismatch.")
        os.replace(temp_path, final_path)
        with video_uploads_lock:
            active_upload = video_uploads.get(upload_id)
            if not active_upload:
                raise HTTPException(status_code=404, detail="Upload not found.")
            active_upload.received_chunks.add(chunk_index)
            payload = upload_payload(active_upload)
        logger.info(
            "Video upload chunk stored upload_id=%s chunk_index=%s bytes=%s received=%s/%s",
            upload_id,
            chunk_index,
            bytes_written,
            payload["received_chunks"],
            payload["total_chunks"],
        )
        payload["chunk_index"] = chunk_index
        return payload
    except HTTPException:
        cleanup_files([temp_path])
        raise
    except Exception as e:
        cleanup_files([temp_path])
        logger.exception("Video upload chunk failed upload_id=%s chunk_index=%s error=%s", upload_id, chunk_index, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-video/uploads/{upload_id}/complete", status_code=202)
def complete_process_video_upload(
    upload_id: str,
    background_tasks: BackgroundTasks,
    mode: str = Form("VECTORS"),
    is_moving: Optional[bool] = Form(None),
    isMoving: Optional[bool] = Form(None),
    roi_enabled: Optional[bool] = Form(None),
    roi_left: Optional[float] = Form(None),
    roi_top: Optional[float] = Form(None),
    roi_right: Optional[float] = Form(None),
    roi_bottom: Optional[float] = Form(None),
    roi_view_aspect_ratio: Optional[float] = Form(None),
    roi_path_points: Optional[str] = Form(None),
    roi_selected_position_ms: Optional[int] = Form(None),
):
    if not processor:
        logger.error("Chunked process-video job rejected because model is not loaded upload_id=%s", upload_id)
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    reject_if_video_job_queue_full(upload_id)
    upload = get_video_upload(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found.")

    missing = [index for index in range(upload.total_chunks) if index not in upload.received_chunks]
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"Upload is missing chunks: {missing[:10]}{'...' if len(missing) > 10 else ''}",
        )

    roi = resolve_roi(
        roi_enabled,
        roi_left,
        roi_top,
        roi_right,
        roi_bottom,
        roi_view_aspect_ratio,
        roi_path_points,
        roi_selected_position_ms,
    )

    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path = input_temp.name
    input_temp.close()
    try:
        with open(input_path, "wb") as output:
            for index in range(upload.total_chunks):
                part_path = chunk_path(upload, index)
                with open(part_path, "rb") as part:
                    shutil.copyfileobj(part, output)
        actual_size = os.path.getsize(input_path)
        if actual_size != upload.file_size:
            cleanup_files([input_path])
            raise HTTPException(
                status_code=400,
                detail=f"Assembled file size mismatch: {actual_size} != {upload.file_size}.",
            )

        logger.info(
            "Video upload assembled upload_id=%s file_name=%s input_path=%s size=%s",
            upload_id,
            upload.file_name,
            input_path,
            actual_size,
        )
        remove_video_upload(upload_id, cleanup=True)
        return enqueue_video_job(
            background_tasks=background_tasks,
            input_path=input_path,
            filename=upload.file_name,
            mode_name=mode.upper(),
            resolved_is_moving=resolve_is_moving(is_moving, isMoving),
            roi=roi,
        )
    except HTTPException:
        raise
    except Exception as e:
        cleanup_files([input_path])
        logger.exception("Video upload complete failed upload_id=%s error=%s", upload_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/process-video/uploads/{upload_id}")
def cancel_process_video_upload(upload_id: str):
    upload = remove_video_upload(upload_id, cleanup=True)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found.")
    logger.info("Video upload cancelled upload_id=%s upload_dir=%s", upload_id, upload.upload_dir)
    return {"upload_id": upload_id, "status": "cancelled"}


@app.post("/process-video/jobs/{job_id}/cancel")
def cancel_process_video_job(job_id: str):
    job = request_video_job_cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    logger.info(
        "Video job cancel requested job_id=%s status=%s input_path=%s output_path=%s",
        job_id,
        job.status,
        job.input_path,
        job.output_path,
    )
    if job.status == "cancelled":
        cleanup_files([job.input_path, job.output_path])
    return {"job_id": job_id, "status": job.status}


@app.get("/process-video/jobs/{job_id}")
def get_process_video_job(job_id: str):
    job = get_video_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    payload = asdict(job)
    payload.pop("input_path", None)
    payload.pop("output_path", None)
    payload.pop("cancel_requested", None)
    return payload


@app.get("/process-video/jobs/{job_id}/result/info")
def get_process_video_job_result_info(job_id: str):
    job = completed_video_job_or_error(job_id)
    payload = video_result_info(job)
    logger.info(
        "Video job result info returned job_id=%s output_path=%s output_size_bytes=%s total_chunks=%s",
        job_id,
        job.output_path,
        payload["file_size"],
        payload["total_chunks"],
    )
    return payload


@app.get("/process-video/jobs/{job_id}/result/chunks/{chunk_index}")
def get_process_video_job_result_chunk(job_id: str, chunk_index: int):
    job = completed_video_job_or_error(job_id)
    payload = video_result_info(job)
    total_chunks = payload["total_chunks"]
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(status_code=400, detail="chunk_index is out of range.")

    output_size = payload["file_size"]
    offset = chunk_index * RESULT_DOWNLOAD_CHUNK_BYTES
    byte_count = min(RESULT_DOWNLOAD_CHUNK_BYTES, output_size - offset)
    logger.info(
        "Video job result chunk returned job_id=%s chunk_index=%s bytes=%s total_chunks=%s",
        job_id,
        chunk_index,
        byte_count,
        total_chunks,
    )
    headers = {
        "Content-Length": str(byte_count),
        "Content-Disposition": f'attachment; filename="{os.path.basename(job.output_path)}.part{chunk_index + 1}"',
        "X-Chunk-Index": str(chunk_index),
        "X-Total-Chunks": str(total_chunks),
        "X-File-Size": str(output_size),
    }
    return StreamingResponse(
        iter_file_chunk(job.output_path, offset, byte_count),
        media_type="application/octet-stream",
        headers=headers,
    )


@app.delete("/process-video/jobs/{job_id}/result")
def cleanup_process_video_job_result(job_id: str):
    job = get_video_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status in ("queued", "processing", "cancelling"):
        raise HTTPException(status_code=409, detail=f"Job is not finished yet: {job.status}.")
    cleanup_video_job_result(job_id)
    logger.info("Video job result cleaned job_id=%s output_path=%s", job_id, job.output_path)
    return {"job_id": job_id, "status": "cleaned"}


@app.get("/process-video/jobs/{job_id}/result")
def get_process_video_job_result(job_id: str, background_tasks: BackgroundTasks):
    job = completed_video_job_or_error(job_id)
    output_size = os.path.getsize(job.output_path)
    logger.info("Video job result returned job_id=%s output_path=%s output_size_bytes=%s", job_id, job.output_path, output_size)
    background_tasks.add_task(cleanup_files, [job.output_path])
    background_tasks.add_task(remove_video_job, job_id)
    return FileResponse(
        path=job.output_path,
        media_type="video/mp4",
        filename=os.path.basename(job.output_path)
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": processor is not None,
        "runtime": processor.runtime_info() if processor else None,
        "video_jobs": video_job_counts(),
        "video_uploads": video_upload_counts(),
    }
