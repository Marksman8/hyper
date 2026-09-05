"""Resource awareness (spec section 26).

v0.1 only *reports*. The throttling policy -- back off when a game or a
training run is active -- needs vision and background inference to exist before
it has anything to throttle. But the measurement has to be here first, because
the Phase 5/7 budget decisions should be made against real numbers from this
machine rather than estimates.

VRAM comes from `nvidia-smi` rather than a Python CUDA binding, deliberately:
AURA core must not depend on torch or pynvml. A 30 ms subprocess call, cached,
is the cheaper trade.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass

import psutil

from aura.logging import get_logger

log = get_logger(__name__)

_GB = 1024**3
_VRAM_CACHE_TTL_S = 5.0

_vram_cache: tuple[float, tuple[float, float] | None] = (0.0, None)


@dataclass(slots=True, frozen=True)
class ResourceSnapshot:
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    process_ram_mb: float
    vram_used_gb: float | None
    vram_total_gb: float | None


def _read_vram() -> tuple[float, float] | None:
    """Return (used_gb, total_gb) via nvidia-smi, or None if unavailable."""
    global _vram_cache
    now = time.monotonic()
    ts, cached = _vram_cache
    if now - ts < _VRAM_CACHE_TTL_S:
        return cached

    exe = shutil.which("nvidia-smi")
    if exe is None:
        _vram_cache = (now, None)
        return None

    try:
        out = subprocess.run(  # noqa: S603 - fixed executable, fixed args
            [
                exe,
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout
        used_mb, total_mb = (float(x.strip()) for x in out.splitlines()[0].split(","))
        result = (used_mb / 1024.0, total_mb / 1024.0)
    except (subprocess.SubprocessError, OSError, ValueError, IndexError) as exc:
        log.debug("resources.vram_probe_failed", error=str(exc))
        result = None

    _vram_cache = (now, result)
    return result


def snapshot() -> ResourceSnapshot:
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    vram = _read_vram()
    return ResourceSnapshot(
        # interval=None returns the value since the last call, which is
        # non-blocking. The first call after startup reports 0.0.
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_used_gb=round((vm.total - vm.available) / _GB, 2),
        ram_total_gb=round(vm.total / _GB, 2),
        ram_percent=vm.percent,
        process_ram_mb=round(proc.memory_info().rss / (1024**2), 1),
        vram_used_gb=round(vram[0], 2) if vram else None,
        vram_total_gb=round(vram[1], 2) if vram else None,
    )
