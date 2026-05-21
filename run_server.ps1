param(
    [string]$HostAddress = $(if ($env:OPTICAL_FLOW_SERVER_HOST) { $env:OPTICAL_FLOW_SERVER_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:OPTICAL_FLOW_SERVER_PORT) { [int]$env:OPTICAL_FLOW_SERVER_PORT } else { 8000 })
)

python -m uvicorn main:app --host $HostAddress --port $Port
