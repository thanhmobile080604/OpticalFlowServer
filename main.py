import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import shutil
import os
import uuid
import tempfile
import threading
from dataclasses import dataclass, asdict
from typing import Optional
from inference import OpticalFlowProcessor, ProcessingCancelled

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
DEFAULT_MODEL = "optical_flow_estimation_raft_2023aug_int8bq.onnx"
DEQUANT_MODEL = "optical_flow_estimation_raft_2023aug_dequant.onnx"
ALT_MODEL = "optical_flow_estimation_raft_2023aug.onnx"
# Prefer dequantized float model, then alternate float model, then default int8 model
if os.path.exists(DEQUANT_MODEL):
    MODEL_PATH = DEQUANT_MODEL
elif os.path.exists(ALT_MODEL):
    MODEL_PATH = ALT_MODEL
else:
    MODEL_PATH = DEFAULT_MODEL
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
    progress: int = 0
    error: Optional[str] = None
    cancel_requested: bool = False


video_jobs = {}
video_jobs_lock = threading.Lock()


def cleanup_files(file_paths):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug("Cleaned up temp file path=%s", path)
        except Exception as e:
            logger.warning("Failed to cleanup temp file path=%s error=%s", path, e)


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

    set_video_job(job_id, status="processing", progress=0)
    vector_direction_sign = vector_direction_sign_for_motion(job.is_moving)
    def update_progress(percent):
        set_video_job(job_id, progress=max(0, min(100, int(percent))))

    try:
        input_size = os.path.getsize(job.input_path) if os.path.exists(job.input_path) else -1
        logger.info(
            "Video job started job_id=%s mode=%s is_moving=%s vector_direction_sign=%.1f input_path=%s input_size_bytes=%s output_path=%s",
            job_id,
            job.mode,
            job.is_moving,
            vector_direction_sign,
            job.input_path,
            input_size,
            job.output_path,
        )
        processor.process_video(
            job.input_path,
            job.output_path,
            mode=job.mode.upper(),
            vector_direction_sign=vector_direction_sign,
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
        cleanup_files([job.input_path])


@app.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("VECTORS"),
    is_moving: Optional[bool] = Form(None),
    isMoving: Optional[bool] = Form(None)
):
    """
    Process a video using the RAFT Optical Flow model.
    mode: "VECTORS" or "HEATMAP"
    is_moving: true if the camera is moving forward, false if moving backward/stationary (affects vector direction)
    """
    if not processor:
        logger.error("Synchronous process-video rejected because model is not loaded filename=%s", file.filename)
        return {"error": "Model not loaded on server."}

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
        "Synchronous video processing started filename=%s mode=%s raw_is_moving=%s raw_isMoving=%s resolved_is_moving=%s vector_direction_sign=%.1f input_path=%s input_size_bytes=%s output_path=%s",
        file.filename,
        mode_name,
        is_moving,
        isMoving,
        resolved_is_moving,
        vector_direction_sign,
        input_path,
        input_size,
        output_path,
    )

    try:
        # Process the video (no status file)
        processor.process_video(input_path, output_path, mode=mode_name, vector_direction_sign=vector_direction_sign)

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
    isMoving: Optional[bool] = Form(None)
):
    """
    Create an async video-processing job for Cloudflare Tunnel clients.
    This avoids holding a long HTTP request open while RAFT processing runs.
    """
    if not processor:
        logger.error("Async process-video job rejected because model is not loaded filename=%s", file.filename)
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path = input_temp.name
    input_temp.close()
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_temp.name
    output_temp.close()

    job_id = uuid.uuid4().hex
    mode_name = mode.upper()
    resolved_is_moving = resolve_is_moving(is_moving, isMoving)
    job = VideoJob(
        job_id=job_id,
        status="queued",
        input_path=input_path,
        output_path=output_path,
        mode=mode_name,
        is_moving=resolved_is_moving,
        progress=0,
    )
    with video_jobs_lock:
        video_jobs[job_id] = job

    input_size = os.path.getsize(input_path) if os.path.exists(input_path) else -1
    logger.info(
        "Video job queued job_id=%s filename=%s mode=%s raw_is_moving=%s raw_isMoving=%s resolved_is_moving=%s input_path=%s input_size_bytes=%s output_path=%s",
        job_id,
        file.filename,
        mode_name,
        is_moving,
        isMoving,
        resolved_is_moving,
        input_path,
        input_size,
        output_path,
    )
    background_tasks.add_task(run_video_job, job_id)
    return {"job_id": job_id, "status": job.status}


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


@app.get("/process-video/jobs/{job_id}/result")
def get_process_video_job_result(job_id: str, background_tasks: BackgroundTasks):
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

    output_size = os.path.getsize(job.output_path) if os.path.exists(job.output_path) else -1
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
    return {"status": "ok", "model_loaded": processor is not None}
