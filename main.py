import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import tempfile
import threading
from dataclasses import dataclass, asdict
from typing import Optional
from inference import OpticalFlowProcessor

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
    processor = OpticalFlowProcessor(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Failed to load model {MODEL_PATH}. Error: {e}")
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


video_jobs = {}
video_jobs_lock = threading.Lock()


def cleanup_files(file_paths):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Failed to cleanup {path}: {e}")


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


def run_video_job(job_id: str):
    job = get_video_job(job_id)
    if not job:
        return

    set_video_job(job_id, status="processing", progress=0)
    vector_direction_sign = 1.0 if job.is_moving else -1.0
    def update_progress(percent):
        set_video_job(job_id, progress=max(0, min(100, int(percent))))

    try:
        processor.process_video(
            job.input_path,
            job.output_path,
            mode=job.mode.upper(),
            vector_direction_sign=vector_direction_sign,
            progress_callback=update_progress,
        )
        set_video_job(job_id, status="completed", progress=100)
    except Exception as e:
        cleanup_files([job.output_path])
        set_video_job(job_id, status="failed", error=str(e))
    finally:
        cleanup_files([job.input_path])


@app.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("VECTORS"),
    is_moving: bool = Form(False)
):
    """
    Process a video using the RAFT Optical Flow model.
    mode: "VECTORS" or "HEATMAP"
    is_moving: true if the camera is moving forward, false if moving backward/stationary (affects vector direction)
    """
    if not processor:
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

    vector_direction_sign = 1.0 if is_moving else -1.0

    try:
        # Process the video (no status file)
        processor.process_video(input_path, output_path, mode=mode.upper(), vector_direction_sign=vector_direction_sign)

        # Schedule cleanup after sending response
        background_tasks.add_task(cleanup_files, [input_path, output_path])

        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=os.path.basename(output_path)
        )
    except Exception as e:
        # cleanup temp files on error
        cleanup_files([input_path, output_path])
        return {"error": str(e)}


@app.post("/process-video/jobs", status_code=202)
async def create_process_video_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("VECTORS"),
    is_moving: bool = Form(False)
):
    """
    Create an async video-processing job for Cloudflare Tunnel clients.
    This avoids holding a long HTTP request open while RAFT processing runs.
    """
    if not processor:
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
    job = VideoJob(
        job_id=job_id,
        status="queued",
        input_path=input_path,
        output_path=output_path,
        mode=mode.upper(),
        is_moving=is_moving,
        progress=0,
    )
    with video_jobs_lock:
        video_jobs[job_id] = job

    background_tasks.add_task(run_video_job, job_id)
    return {"job_id": job_id, "status": job.status}


@app.get("/process-video/jobs/{job_id}")
def get_process_video_job(job_id: str):
    job = get_video_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    payload = asdict(job)
    payload.pop("input_path", None)
    payload.pop("output_path", None)
    return payload


@app.get("/process-video/jobs/{job_id}/result")
def get_process_video_job_result(job_id: str, background_tasks: BackgroundTasks):
    job = get_video_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.error or "Video processing failed.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is not completed yet: {job.status}.")
    if not os.path.exists(job.output_path):
        raise HTTPException(status_code=410, detail="Processed video is no longer available.")

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
