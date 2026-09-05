<#
.SYNOPSIS
    Fetch a prebuilt CUDA llama.cpp release into var/llama.cpp/.

.DESCRIPTION
    AURA's default backend is llama.cpp's `llama-server`, which speaks an
    OpenAI-compatible API. This grabs an official prebuilt Windows CUDA binary
    so nothing has to be compiled (no cmake, no Visual Studio).

    IMPORTANT for this machine: an RTX 5060 is Blackwell (compute capability
    12.0). It requires a CUDA 12.8-or-newer build. Older cu11/cu12.4 binaries
    load and then fail at inference with a "no kernel image is available"
    error. The script picks a cu12.8+ asset for that reason.

.PARAMETER Tag
    A specific llama.cpp release tag, e.g. b4800. Default: latest.
#>
[CmdletBinding()]
param(
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "var\llama.cpp"

Write-Host "AURA - llama.cpp setup" -ForegroundColor Cyan
Write-Host ""

# --- Verify the GPU and driver can actually run a CUDA build ----------------
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -eq $smi) {
    Write-Warning "nvidia-smi not found. Without an NVIDIA GPU, llama.cpp will run on CPU (slow)."
} else {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)
    Write-Host "  GPU: $gpu"
}

# --- Resolve the release asset ----------------------------------------------
$api = if ($Tag -eq "latest") {
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
} else {
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$Tag"
}

Write-Host "  Querying llama.cpp releases..." -ForegroundColor DarkGray
$release = Invoke-RestMethod -Uri $api -Headers @{ "User-Agent" = "aura-setup" }

# Prefer the newest CUDA toolkit available; Blackwell needs 12.8+.
$asset = $release.assets |
    Where-Object { $_.name -match "win.*cuda.*x64\.zip$" -or $_.name -match "cuda.*win.*x64\.zip$" } |
    Sort-Object { [int]($_.name -replace '.*cu(?:da)?-?(\d+)\.?(\d+).*', '$1$2') } -Descending |
    Select-Object -First 1

if ($null -eq $asset) {
    Write-Error @"
No Windows CUDA asset found in release $($release.tag_name).

Download one manually from:
  https://github.com/ggml-org/llama.cpp/releases

Pick the newest 'llama-<tag>-bin-win-cuda-<version>-x64.zip' (CUDA 12.8 or
newer for RTX 50-series), and extract it to:
  $dest
"@
}

Write-Host "  Release: $($release.tag_name)"
Write-Host "  Asset:   $($asset.name)"

if ($asset.name -match 'cu(?:da)?-?(\d+)\.(\d+)') {
    $major = [int]$Matches[1]; $minor = [int]$Matches[2]
    if ($major -lt 12 -or ($major -eq 12 -and $minor -lt 8)) {
        Write-Warning "This asset is CUDA $major.$minor. RTX 50-series (Blackwell) needs 12.8+; inference may fail."
    }
}

# --- Download and extract ----------------------------------------------------
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$zip = Join-Path $env:TEMP $asset.name

Write-Host "  Downloading $([math]::Round($asset.size / 1MB)) MB..." -ForegroundColor DarkGray
$ProgressPreference = "SilentlyContinue"   # ~10x faster Invoke-WebRequest
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
$ProgressPreference = "Continue"

Write-Host "  Extracting to var\llama.cpp ..." -ForegroundColor DarkGray
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip -Force

$server = Get-ChildItem -Path $dest -Filter "llama-server.exe" -Recurse |
    Select-Object -First 1
if ($null -eq $server) {
    Write-Error "llama-server.exe not found after extraction. Inspect $dest."
}

Write-Host ""
Write-Host "Ready: $($server.FullName)" -ForegroundColor Green
Write-Host "Next:  scripts\start_model_server.ps1" -ForegroundColor Green
