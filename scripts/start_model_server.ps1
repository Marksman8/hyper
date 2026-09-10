<#
.SYNOPSIS
    Start llama-server with the active model from config/models.yaml.

.DESCRIPTION
    This is AURA's brain, running as its own process. Keep it in its own
    terminal: AURA Core restarts in under a second, this takes ~20 s to load
    weights, and decoupling them is the point of the two-process design.

.PARAMETER Model
    Catalogue key override, e.g. qwen3-8b.

.PARAMETER Port
    Listen port. Must match `llm.base_url` in config. Default 8081.

.PARAMETER Context
    Context window in tokens. Larger costs VRAM in the KV cache.

.PARAMETER GpuLayers
    Layers to offload. -1 offloads everything (the default and what you want
    when the model fits). Lower it if you hit CUDA OOM.

.PARAMETER Reasoning
    off | on | auto. Qwen3.x and other reasoning models emit chain-of-thought
    into a separate `reasoning_content` field before answering. Left on with a
    small token budget, they can spend the entire budget thinking and return an
    empty answer. v0.1 conversation wants low latency, so the default is off;
    turn it on for hard problems, and expect the Phase 4 agent loop to want it.
#>
[CmdletBinding()]
param(
    [string]$Model = "",
    [int]$Port = 8081,
    [int]$Context = 0,
    [int]$GpuLayers = -999,
    [ValidateSet("off", "on", "auto")]
    [string]$Reasoning = "off"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# --- Locate llama-server -----------------------------------------------------
$server = Get-ChildItem -Path (Join-Path $root "var\llama.cpp") -Filter "llama-server.exe" -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -eq $server) {
    Write-Error "llama-server.exe not found. Run: scripts\setup_llama_cpp.ps1"
}

# --- Read the active model from the catalogue --------------------------------
# Parsed by AURA itself so there is exactly one source of truth.
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "No virtualenv. Run: uv sync" }

$modelArg = if ($Model) { $Model } else { "" }
$json = & $py -c @"
import json, sys
sys.path.insert(0, r'$root\src')
from aura.config import get_model_catalogue, models_dir
cat = get_model_catalogue()
key = '$modelArg' or cat.active
e = cat.catalogue[key]
print(json.dumps({
    'key': key, 'name': e.display_name, 'file': str(models_dir() / e.filename),
    'ctx': e.context_length, 'ngl': e.gpu_layers, 'vram': e.vram_gb,
}))
"@
if ($LASTEXITCODE -ne 0) { Write-Error "Could not read config/models.yaml" }
$cfg = $json | ConvertFrom-Json

if (-not (Test-Path $cfg.file)) {
    Write-Error "Weights not found: $($cfg.file)`nRun: uv run python scripts\download_model.py"
}

$ctx = if ($Context -gt 0) { $Context } else { $cfg.ctx }
$ngl = if ($GpuLayers -ne -999) { $GpuLayers } else { $cfg.ngl }

# --- Warn before CUDA OOM rather than after ----------------------------------
# Advisory only, and wrapped because it is advisory. nvidia-smi can be present
# and still fail -- a permissions error returns nothing, which left $mem null
# and took the whole script down on a .Trim(). A VRAM *warning* that prevents
# the server starting is worse than no warning at all. Found in 10C.2.
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    try {
        $raw = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        $mem = if ($raw) { $raw -split ',' } else { $null }
        if ($mem -and $mem.Count -ge 2) {
            $freeGb = ([double]$mem[1].Trim() - [double]$mem[0].Trim()) / 1024
            Write-Host ("  VRAM free: {0:N1} GB, model needs ~{1} GB" -f $freeGb, $cfg.vram)
            if ($freeGb -lt $cfg.vram) {
                Write-Warning "Not enough free VRAM. Close GPU applications, or pass -GpuLayers to offload fewer layers."
            }
        } else {
            Write-Host "  VRAM: nvidia-smi gave no reading; skipping the check." -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "  VRAM: could not be read ($($_.Exception.Message)); skipping the check." -ForegroundColor DarkGray
    }
}

# --- Shared secret for the model server --------------------------------------
# llama-server ships with no auth and Access-Control-Allow-Origin: *. Binding to
# loopback is not enough: any web page you visit can POST to 127.0.0.1:8081 and
# use your GPU, or read whatever is in the KV cache. AURA reads the same file
# via llm.api_key_file, so this needs no configuration.
$secretDir = Join-Path $root "var\secrets"
$secretFile = Join-Path $secretDir "model_api_token"
New-Item -ItemType Directory -Force -Path $secretDir | Out-Null
if (-not (Test-Path $secretFile)) {
    $bytes = [byte[]]::new(32)
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    # NOT Set-Content -Encoding utf8: on Windows PowerShell 5.1 that writes a
    # UTF-8 BOM, and the BOM bytes end up inside the token any reader that does
    # not strip them, producing a 401 that looks like a wrong key.
    [System.IO.File]::WriteAllText($secretFile, $token, (New-Object System.Text.UTF8Encoding($false)))
    & "$env:SystemRoot\System32\icacls.exe" $secretFile /inheritance:r /grant:r "$($env:USERNAME):F" | Out-Null
    Write-Host "  Generated model server token: var\secrets\model_api_token" -ForegroundColor DarkGray
}
$apiKey = (Get-Content $secretFile -Raw).Trim()

Write-Host ""
Write-Host "AURA model server" -ForegroundColor Cyan
Write-Host "  Model:     $($cfg.name)  [$($cfg.key)]"
Write-Host "  Context:   $ctx tokens"
Write-Host "  Layers:    $(if ($ngl -eq -1) { 'all on GPU' } else { "$ngl on GPU" })"
Write-Host "  Reasoning: $Reasoning"
Write-Host "  API:       http://127.0.0.1:$Port/v1  (bearer token required)"
Write-Host ""

# --host 127.0.0.1 is deliberate: the model server must not be reachable off
# this machine any more than AURA Core is.
& $server.FullName `
    --model $cfg.file `
    --host 127.0.0.1 `
    --port $Port `
    --api-key $apiKey `
    --ctx-size $ctx `
    --n-gpu-layers $ngl `
    --cache-type-k q8_0 `
    --cache-type-v q8_0 `
    --flash-attn on `
    --jinja `
    --reasoning $Reasoning `
    --alias aura-local
