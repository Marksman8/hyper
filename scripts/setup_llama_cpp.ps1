<#
.SYNOPSIS
    Fetch prebuilt CUDA llama.cpp binaries into var/llama.cpp/.

.DESCRIPTION
    AURA's default backend is llama.cpp's `llama-server`, which speaks an
    OpenAI-compatible API. This grabs official prebuilt Windows CUDA binaries
    so nothing has to be compiled (no cmake, no Visual Studio).

    Two things this script gets right that are easy to get wrong:

    1. llama.cpp publishes its binary builds as GitHub *prereleases* tagged
       b<number>. The /releases/latest endpoint skips prereleases entirely and
       returns an unrelated stable tag with no binaries, so this walks the
       release list and takes the newest one that actually has a CUDA asset.

    2. The binaries do not bundle the CUDA runtime. There is a separate
       `cudart-llama-bin-win-cuda-<ver>-x64.zip` per release, and without it
       llama-server.exe fails to start with a missing-DLL error. Both are
       downloaded and extracted into the same directory.

    Hardware note: an RTX 5060 is Blackwell (compute capability 12.0) and needs
    a CUDA 12.8-or-newer build. Older cu12.4 binaries load and then fail at
    inference with "no kernel image is available". The script sorts by CUDA
    version and takes the highest.

.PARAMETER Tag
    A specific llama.cpp release tag, e.g. b10816. Default: newest with CUDA.
#>
[CmdletBinding()]
param(
    [string]$Tag = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "var\llama.cpp"
$headers = @{ "User-Agent" = "aura-setup" }

Write-Host "AURA - llama.cpp setup" -ForegroundColor Cyan
Write-Host ""

# --- Verify the GPU and driver can actually run a CUDA build ----------------
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -eq $smi) {
    Write-Warning "nvidia-smi not found. Without an NVIDIA GPU, llama.cpp will run on CPU (slow)."
} else {
    Write-Host "  GPU: $(& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
}

# --- Find the newest release that actually ships CUDA binaries ---------------
Write-Host "  Querying llama.cpp releases..." -ForegroundColor DarkGray

if ($Tag) {
    $releases = @(Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$Tag" -Headers $headers)
} else {
    # Deliberately NOT /releases/latest: binary builds are prereleases.
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15" -Headers $headers
}

$release = $null
$binAsset = $null
foreach ($r in $releases) {
    $candidate = $r.assets |
        Where-Object { $_.name -match '^llama-.*-bin-win-cuda-\d+\.\d+-x64\.zip$' } |
        Sort-Object { [version]([regex]::Match($_.name, 'cuda-(\d+\.\d+)').Groups[1].Value) } -Descending |
        Select-Object -First 1
    if ($null -ne $candidate) { $release = $r; $binAsset = $candidate; break }
}

if ($null -eq $binAsset) {
    Write-Error @"
No Windows CUDA asset found in the 15 most recent llama.cpp releases.

Download one manually from https://github.com/ggml-org/llama.cpp/releases
(newest 'llama-<tag>-bin-win-cuda-<ver>-x64.zip' plus the matching
'cudart-llama-bin-win-cuda-<ver>-x64.zip') and extract both to:
  $dest
"@
}

$cudaVer = [regex]::Match($binAsset.name, 'cuda-(\d+\.\d+)').Groups[1].Value
if ([version]$cudaVer -lt [version]"12.8") {
    Write-Warning "Newest available build is CUDA $cudaVer. RTX 50-series (Blackwell) needs 12.8+; inference will likely fail."
}

# The CUDA runtime DLLs, without which llama-server.exe will not start.
$cudartAsset = $release.assets | Where-Object { $_.name -eq "cudart-llama-bin-win-cuda-$cudaVer-x64.zip" }

Write-Host "  Release: $($release.tag_name)"
Write-Host "  CUDA:    $cudaVer"
Write-Host "  Assets:  $($binAsset.name)$(if ($cudartAsset) { ' + ' + $cudartAsset.name })"

# --- Download and extract ----------------------------------------------------
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$ProgressPreference = "SilentlyContinue"   # ~10x faster Invoke-WebRequest

foreach ($asset in @($binAsset, $cudartAsset)) {
    if ($null -eq $asset) { continue }
    $zip = Join-Path $env:TEMP $asset.name
    Write-Host "  Downloading $($asset.name) ($([math]::Round($asset.size / 1MB)) MB)..." -ForegroundColor DarkGray
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Remove-Item $zip -Force
}

$ProgressPreference = "Continue"

$server = Get-ChildItem -Path $dest -Filter "llama-server.exe" -Recurse | Select-Object -First 1
if ($null -eq $server) {
    Write-Error "llama-server.exe not found after extraction. Inspect $dest."
}

# Confirm the CUDA runtime landed next to the binary, since a missing DLL only
# surfaces as a cryptic failure at launch.
$cudart = Get-ChildItem -Path $server.Directory -Filter "cudart64*.dll" -ErrorAction SilentlyContinue
if ($null -eq $cudart) {
    Write-Warning "No cudart64*.dll beside llama-server.exe. If it fails to start, extract the cudart zip into $($server.Directory)."
}

Write-Host ""
Write-Host "Ready: $($server.FullName)" -ForegroundColor Green
Write-Host "Next:  scripts\start_model_server.ps1" -ForegroundColor Green
