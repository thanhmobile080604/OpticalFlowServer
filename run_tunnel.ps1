param(
    [string]$ConfigPath = $(Join-Path $env:USERPROFILE ".cloudflared\config.yml"),
    [string]$Tunnel = $env:CLOUDFLARE_TUNNEL
)

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Error "cloudflared is not installed or is not on PATH."
    exit 1
}

if (-not (Test-Path -LiteralPath $ConfigPath)) {
    Write-Error "Cloudflare Tunnel config not found: $ConfigPath"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($Tunnel)) {
    cloudflared tunnel --config $ConfigPath run
} else {
    cloudflared tunnel --config $ConfigPath run $Tunnel
}
