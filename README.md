# OpticalFlowServer

FastAPI server for RAFT ONNX optical-flow video processing.

The server supports three ONNX Runtime backends:

- `CUDAExecutionProvider`: best for newer NVIDIA GPUs such as GTX/RTX 16xx/20xx/30xx/40xx.
- `DmlExecutionProvider`: Windows GPU backend through DirectML/DirectX. Use this for Quadro P400/Pascal.
- `CPUExecutionProvider`: safe fallback for any machine.

Do not install multiple ONNX Runtime backend packages together manually. They all import as `onnxruntime` and can overwrite each other.

## First Setup

From PowerShell:

```powershell
cd C:\CODE\OpticalFlowServer
.\setup.ps1
```

`setup.ps1` detects the machine and installs one backend:

- Quadro P400/Pascal on Windows -> DirectML
- Newer NVIDIA GPU -> CUDA
- No suitable GPU -> CPU

Manual install options:

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` is CPU-safe. For a specific GPU backend:

```powershell
python -m pip install -r requirements-cuda.txt
python -m pip install -r requirements-directml.txt
```

## Run Server

Run uvicorn in the foreground. This writes logs to the terminal only and does not create `uvicorn.*.log` files.

```powershell
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
$env:OPTICAL_FLOW_MAX_CONCURRENT_VIDEO_JOBS = "3"
$env:OPTICAL_FLOW_MAX_PENDING_VIDEO_JOBS = "8"
```

For manual provider override:

```powershell
$env:OPTICAL_FLOW_ONNX_PROVIDERS = "DmlExecutionProvider,CPUExecutionProvider"
$env:OPTICAL_FLOW_ONNX_PROVIDERS = "CUDAExecutionProvider,CPUExecutionProvider"
```
