# OpticalFlowServer

FastAPI server for RAFT ONNX optical-flow video processing.

The server supports three ONNX Runtime backends:

- `CUDAExecutionProvider`: best for newer NVIDIA GPUs such as GTX/RTX 16xx/20xx/30xx/40xx.
- `DmlExecutionProvider`: Windows GPU backend through DirectML/DirectX. Use this for Quadro P400/Pascal.
- `CPUExecutionProvider`: safe fallback for any machine.

Do not install multiple ONNX Runtime backend packages together manually. They all import as `onnxruntime` and can overwrite each other.

## First Setup

From PowerShell (run the setup script in the repo's docs folder):

```powershell
cd C:\CODE\OpticalFlowServer
.\docs\docs_to_set_up\setup.ps1
```

`setup.ps1` (in `docs/docs_to_set_up`) detects the machine and installs one backend:

- Quadro P400/Pascal on Windows -> DirectML
- Newer NVIDIA GPU -> CUDA
- No suitable GPU -> CPU

Manual install options:

```powershell
python -m pip install -r docs/docs_to_set_up/requirements.txt
```

`requirements.txt` (in `docs/docs_to_set_up`) is CPU-safe. For a specific GPU backend:

```powershell
python -m pip install -r docs/docs_to_set_up/requirements-cuda.txt
python -m pip install -r docs/docs_to_set_up/requirements-directml.txt
```

## Run Server

Run uvicorn in the foreground. This writes logs to the terminal only and does not create `uvicorn.*.log` files.

```powershell
# From the repository root:
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or change into the `src` folder and run:
cd src
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Stop the server with `Ctrl+C`.

## Check Runtime

Open a second PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 6
```

Expected GPU examples:

```json
"execution_provider": "CUDAExecutionProvider",
"gpu_enabled": true
```

or, on Quadro P400/Windows:

```json
"execution_provider": "DmlExecutionProvider",
"gpu_enabled": true
```

If the server falls back to CPU, `/health` shows `execution_provider` as `CPUExecutionProvider` and includes `gpu_fallback_error` when a GPU provider failed at runtime.

## Cloudflare Tunnel

Install `cloudflared`:

https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

Run:

```powershell
cloudflared tunnel --url http://localhost:8000
```

Copy the generated URL into the Android app property:

```text
opticalFlowServerBaseUrl=https://your-tunnel-url.trycloudflare.com
```

## Useful Limits

Optional environment variables:

```powershell
$env:OPTICAL_FLOW_MAX_CONCURRENT_VIDEO_JOBS = "1"
$env:OPTICAL_FLOW_MAX_PENDING_VIDEO_JOBS = "8"
```

For manual provider override:

```powershell
$env:OPTICAL_FLOW_ONNX_PROVIDERS = "DmlExecutionProvider,CPUExecutionProvider"
$env:OPTICAL_FLOW_ONNX_PROVIDERS = "CUDAExecutionProvider,CPUExecutionProvider"
```

For Intel Iris Xe or other laptop iGPUs, ROI segmentation jobs can make DirectML stall the
desktop. The server defaults ROI jobs to CPU for stability. Keep this setting
unless you have a discrete GPU:

```powershell
$env:OPTICAL_FLOW_ROI_FORCE_CPU = "true"
$env:OPTICAL_FLOW_OPENCV_THREADS = "1"
$env:OPTICAL_FLOW_ROI_FRAME_OFFSET = "2"
$env:OPTICAL_FLOW_ROI_MIN_MOTION = "0.05"
```

## Cutie Object ROI

Video jobs with an ROI use Cutie to propagate the selected object mask across the
uploaded video. Optical-flow vectors/heatmap are then drawn only inside the Cutie
mask for that selected object.

Cutie source is vendored in `third_party/Cutie`. Install the server
requirements, then download the Cutie weights:

```powershell
python -m pip install -r docs/docs_to_set_up/requirements.txt
python third_party\Cutie\cutie\utils\download_models.py
```

Weights are intentionally ignored by git because the base checkpoint is larger
than GitHub's normal file limit.

Useful Cutie settings:

```powershell
$env:ROI_SEGMENTATION_BACKEND = "cutie"
$env:CUTIE_WEIGHTS = "third_party\Cutie\weights\cutie-base-mega.pth"
$env:CUTIE_DEVICE = "cpu"
$env:CUTIE_MAX_INTERNAL_SIZE = "1080"
$env:CUTIE_MEM_EVERY = "3"
```

Use `CUTIE_DEVICE=cuda` only when PyTorch CUDA is available. On machines without
NVIDIA CUDA, Cutie runs on CPU.

For a temporary non-tracking fallback:

```powershell
$env:ROI_SEGMENTATION_BACKEND = "static"
```
