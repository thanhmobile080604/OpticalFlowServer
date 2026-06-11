param(
    [ValidateSet("auto", "cpu", "cuda", "directml")]
    [string]$Backend = "auto"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$Python = "python"

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-NvidiaGpuNames {
    if (-not (Test-CommandExists "nvidia-smi")) {
        return @()
    }

    try {
        $names = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $names) {
            return @()
        }
        return @($names | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    } catch {
        return @()
    }
}

function Resolve-Backend {
    param([string]$Requested)

    if ($Requested -ne "auto") {
        return $Requested
    }

    $isWindows = $env:OS -eq "Windows_NT"
    $nvidiaNames = Get-NvidiaGpuNames
    $hasNvidia = $nvidiaNames.Count -gt 0
    $nvidiaText = ($nvidiaNames -join " ")
    $isPascalQuadro = $nvidiaText -match "Quadro P|P400|P600|P1000|P2000|Pascal"

    if ($isWindows -and $isPascalQuadro) {
        return "directml"
    }

    if ($hasNvidia) {
        return "cuda"
    }

    if ($isWindows) {
        return "directml"
    }

    return "cpu"
}

$SelectedBackend = Resolve-Backend $Backend

$requirementsByBackend = @{
    "cpu" = "requirements.txt"
    "cuda" = "requirements-cuda.txt"
    "directml" = "requirements-directml.txt"
}

$requirementsFile = $requirementsByBackend[$SelectedBackend]
if (-not $requirementsFile) {
    throw "Unknown backend: $SelectedBackend"
}

Write-Host "Selected ONNX Runtime backend: $SelectedBackend"
Write-Host "Installing from $requirementsFile"

$onnxPackages = @(
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-nvjitlink-cu12"
)

& $Python -m pip uninstall -y @onnxPackages
& $Python -m pip install -r $requirementsFile

Write-Host ""
Write-Host "Installed provider check:"
& $Python -c "import onnxruntime as ort; print('ONNX Runtime', ort.__version__); print('Providers', ort.get_available_providers())"

Write-Host ""
Write-Host "Run server:"
Write-Host "python -m uvicorn main:app --host 0.0.0.0 --port 8000"
