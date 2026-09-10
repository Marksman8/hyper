<#
.SYNOPSIS
    Start a second llama-server serving the active vision model (10C.2).

.DESCRIPTION
    A VLM needs two files: the weights and an `mmproj` vision projector. Both
    are passed here. Without --mmproj the model loads happily and cannot see,
    which reads as a bad model rather than a missing flag, so this script
    refuses to start unless both files are present.

    **This does not run alongside the language model.** 3.3 GB of Qwen3.5 plus
    ~2.9 GB of vision does not fit on an 8 GB card that is also driving the
    display. Stop the model server first, or let 10C.3's ResourceGovernor do
    the eviction once it exists. The VRAM check below will tell you before CUDA
    does.

    A SEPARATE PROCESS on its own port, deliberately: evicting vision is then
    stopping a process, not unloading weights inside the server AURA is
    mid-conversation with.

.PARAMETER Model
    Vision catalogue key override, e.g. smolvlm2-2.2b.

.PARAMETER Port
    Listen port. Must match `vision.vlm.base_url` in config. Default 8082.

.PARAMETER Context
    Context window in tokens. Images cost tokens; a screenshot is not cheap.

.PARAMETER GpuLayers
    Layers to offload. -1 offloads everything. Lower it if you hit CUDA OOM.
#>
[CmdletBinding()]
param(
    [string]$Model = "",
    [int]$Port = 8082,
    [int]$Context = 0,
    [int]$GpuLayers = -999
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# --- Locate llama-server -----------------------------------------------------
$server = Get-ChildItem -Path (Join-Path $root "var\llama.cpp") -Filter "llama-server.exe" -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -eq $server) {
    Write-Error "llama-server.exe not found. Run: scripts\setup_llama_cpp.ps1"
}

# --- Read the active vision model from the catalogue -------------------------
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "No virtualenv. Run: uv sync" }

$modelArg = if ($Model) { $Model } else { "" }
$json = & $py -c @"
import json, sys
sys.path.insert(0, r'$root\src')
from aura.config import get_model_catalogue, models_dir
cat = get_model_catalogue()
key = '$modelArg' or cat.active_vision
e = cat.vision_catalogue[key]
out = models_dir() / 'vision'
print(json.dumps({
    'key': key, 'name': e.display_name,
    'file': str(out / e.filename), 'mmproj': str(out / e.mmproj),
    'ctx': e.context_length, 'ngl': e.gpu_layers, 'vram': e.vram_gb,
}))
"@
if ($LASTEXITCODE -ne 0) { Write-Error "Could not read config/models.yaml" }
$cfg = $json | ConvertFrom-Json

if (-not (Test-Path $cfg.file)) {
    Write-Error "Weights not found: $($cfg.file)`nRun: uv run python scripts\download_vlm.py"
}
if (-not (Test-Path $cfg.mmproj)) {
    # Worth its own error. Starting without this produces a model that answers
    # confidently about images it never saw.
    Write-Error "Vision projector not found: $($cfg.mmproj)`nRun: uv run python scripts\download_vlm.py"
}

$ctx = if ($Context -gt 0) { $Context } else { $cfg.ctx }
$ngl = if ($GpuLayers -ne -999) { $GpuLayers } else { $cfg.ngl }

# --- Warn before CUDA OOM rather than after ----------------------------------
#
# Advisory only, and wrapped because it is advisory. nvidia-smi is present but
# returns a permissions error on this machine, which left $mem null and took
# the whole script down on a .Trim() -- a VRAM *warning* that prevents the
# server starting is worse than no warning at all.
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    try {
        $raw = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        $mem = if ($raw) { $raw -split ',' } else { $null }
        if ($mem -and $mem.Count -ge 2) {
            $freeGb = ([double]$mem[1].Trim() - [double]$mem[0].Trim()) / 1024
            Write-Host ("  VRAM free: {0:N1} GB, vision model needs ~{1} GB" -f $freeGb, $cfg.vram)
            if ($freeGb -lt $cfg.vram) {
                Write-Warning "Not enough free VRAM. Stop the language model server (port 8081) first."
            }
        } else {
            Write-Host "  VRAM: nvidia-smi gave no reading; skipping the check." -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "  VRAM: could not be read ($($_.Exception.Message)); skipping the check." -ForegroundColor DarkGray
    }
}

# --- Shared secret -----------------------------------------------------------
# Its own token, not the language model's. Two processes, two secrets: leaking
# one should not hand over the other, and AURA reads this one via
# vision.vlm.api_key_file.
$secretDir = Join-Path $root "var\secrets"
$secretFile = Join-Path $secretDir "vlm_api_token"
New-Item -ItemType Directory -Force -Path $secretDir | Out-Null
if (-not (Test-Path $secretFile)) {
    $bytes = [byte[]]::new(32)
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    # Not Set-Content -Encoding utf8: PowerShell 5.1 writes a BOM, and BOM
    # bytes inside a token produce a 401 that looks like a wrong key.
    [System.IO.File]::WriteAllText($secretFile, $token, (New-Object System.Text.UTF8Encoding($false)))
    & "$env:SystemRoot\System32\icacls.exe" $secretFile /inheritance:r /grant:r "$($env:USERNAME):F" | Out-Null
    Write-Host "  Generated VLM server token: var\secrets\vlm_api_token" -ForegroundColor DarkGray
}
$apiKey = (Get-Content $secretFile -Raw).Trim()

Write-Host ""
Write-Host "AURA vision model server" -ForegroundColor Cyan
Write-Host "  Model:     $($cfg.name)  [$($cfg.key)]"
Write-Host "  Projector: $(Split-Path -Leaf $cfg.mmproj)"
Write-Host "  Context:   $ctx tokens"
Write-Host "  Layers:    $(if ($ngl -eq -1) { 'all on GPU' } else { "$ngl on GPU" })"
Write-Host "  API:       http://127.0.0.1:$Port/v1  (bearer token required)"
Write-Host ""

& $server.FullName `
    --model $cfg.file `
    --mmproj $cfg.mmproj `
    --host 127.0.0.1 `
    --port $Port `
    --api-key $apiKey `
    --ctx-size $ctx `
    --n-gpu-layers $ngl `
    --flash-attn on `
    --jinja `
    --alias aura-vision
