# Optical Flow Server

FastAPI server for RAFT optical-flow video processing.

## Run Locally

```powershell
pip install -r requirements.txt
.\run_server.ps1
```

The server binds to `127.0.0.1:8000` by default. Keep it local and expose it
through Cloudflare Tunnel instead of opening an inbound firewall port.

## Cloudflare Tunnel

Install `cloudflared`, then create a locally managed tunnel:

```powershell
cloudflared tunnel login
cloudflared tunnel create optical-flow
cloudflared tunnel route dns optical-flow optical-flow.example.com
```

Copy `cloudflared.example.yml` to `%USERPROFILE%\.cloudflared\config.yml`,
then replace the tunnel ID, credentials path, and hostname.

Run the tunnel:

```powershell
.\run_tunnel.ps1
```

The Android app should use the public HTTPS base URL only:

```properties
opticalFlowServerBaseUrl=https://optical-flow.example.com
```

Set that property in the Android project's `local.properties`.

## API

- `GET /health` checks server status.
- `POST /process-video` keeps the original synchronous behavior.
- `POST /process-video/jobs` uploads a video and returns `job_id`.
- `GET /process-video/jobs/{job_id}` returns job status and `progress` percent.
- `GET /process-video/jobs/{job_id}/result` downloads the processed video.
