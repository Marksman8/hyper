"""AURA command line.

aura serve      run the core (API + orchestrator + memory)
aura chat       interactive client against a running core
aura status     one-shot status report
aura doctor     check this machine can actually run AURA
aura vision     control what AURA may look at
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from aura import __version__
from aura.config import get_model_catalogue, get_settings, models_dir
from aura.logging import configure_logging, get_logger

app = typer.Typer(
    name="aura",
    help="AURA - local-first personal AI runtime.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
log = get_logger(__name__)


@app.command()
def serve(
    host: str = typer.Option("", help="Override api.host (loopback only)."),
    port: int = typer.Option(0, help="Override api.port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Run AURA Core."""
    import uvicorn

    cfg = get_settings()
    configure_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        log_file=cfg.logging.resolved_file(),
        redact_keys=tuple(cfg.logging.redact_keys),
    )

    bind_host = host or cfg.api.host
    bind_port = port or cfg.api.port

    if reload:
        # --reload needs an import string so the reloader can re-import.
        uvicorn.run(
            "aura.api.app:create_app",
            factory=True,
            host=bind_host,
            port=bind_port,
            reload=True,
            log_config=None,
        )
    else:
        from aura.api.app import create_app

        uvicorn.run(create_app(cfg), host=bind_host, port=bind_port, log_config=None)


@app.command()
def chat(
    message: str = typer.Argument("", help="Send one message and exit."),
    session: str = typer.Option("", "--session", "-s", help="Continue a session id."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Wait for the full reply."),
) -> None:
    """Talk to a running AURA Core."""
    from aura.clients.cli import run_chat

    asyncio.run(
        run_chat(
            one_shot=message or None,
            session_id=session or None,
            stream=not no_stream,
        )
    )


@app.command(name="do")
def do_task(
    task: str = typer.Argument(..., help="What you want AURA to do."),
    approve: list[str] = typer.Option(
        [],
        "--approve",
        "-a",
        help="Pre-approve a tool for this task, e.g. --approve write_file. Repeatable.",
    ),
    session: str = typer.Option("", "--session", "-s", help="Continue a session id."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Hide the tool trace."),
) -> None:
    """Give AURA a task, letting it choose its own tools.

    Reading is allowed without asking. Anything that writes or executes needs
    --approve, named per tool, per task — the model cannot grant that itself.
    """
    from aura.clients.agent_cli import run_agent

    asyncio.run(
        run_agent(
            message=task,
            approve=list(approve),
            session_id=session or None,
            show_trace=not quiet,
        )
    )


@app.command()
def desktop() -> None:
    """Run the desktop shell: tray icon, status overlay, approval prompts.

    A client of AURA Core like any other. Start `aura serve` first.
    """
    from aura.desktop.app import DesktopApp

    cfg = get_settings()
    configure_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        log_file=cfg.logging.resolved_file(),
        redact_keys=tuple(cfg.logging.redact_keys),
    )
    console.print(
        "[dim]AURA desktop shell. Close the overlay to hide it; quit from the tray.[/dim]"
    )
    DesktopApp().run()


# -- vision ------------------------------------------------------------------
#
# The human half of the vision controls. The model gets tools that can only
# narrow what AURA may see (aura/tools/vision.py); turning it on is a decision
# a person makes, here.

vision_app = typer.Typer(
    name="vision", help="Control what AURA may look at.", no_args_is_help=True
)
app.add_typer(vision_app)


@vision_app.command("status")
def vision_status() -> None:
    """What AURA is currently allowed to see."""
    from aura.clients.vision_cli import run_status

    asyncio.run(run_status())


@vision_app.command("on")
def vision_on(
    active: bool = typer.Option(
        False,
        "--active",
        help="Also allow background observation, not only when you ask.",
    ),
) -> None:
    """Turn vision on.

    Defaults to assist mode - AURA looks only when asked. `--active` is the
    stronger grant and exists mostly for the observation loop that is not
    built yet.
    """
    from aura.clients.vision_cli import run_set_mode

    asyncio.run(run_set_mode("active" if active else "assist"))


@vision_app.command("off")
def vision_off() -> None:
    """Turn vision off. Capture stops; held frames are discarded."""
    from aura.clients.vision_cli import run_set_mode

    asyncio.run(run_set_mode("private"))


@vision_app.command("deny")
def vision_deny(app_name: str = typer.Argument(..., help="e.g. Chrome")) -> None:
    """Never look at one application, leaving vision otherwise on."""
    from aura.clients.vision_cli import run_deny

    asyncio.run(run_deny(app_name))


@vision_app.command("allow")
def vision_allow(app_name: str = typer.Argument(..., help="e.g. Code")) -> None:
    """Look at ONLY the named applications. Repeat to add more."""
    from aura.clients.vision_cli import run_allow

    asyncio.run(run_allow(app_name))


@vision_app.command("monitors")
def vision_monitors() -> None:
    """List the displays attached to this machine."""
    from aura.clients.vision_cli import run_monitors

    asyncio.run(run_monitors())


@vision_app.command("read")
def vision_read(
    monitor: int = typer.Option(1, "--monitor", "-m", help="Which display, from 1."),
) -> None:
    """Read the screen once, through the same tool the model would call."""
    from aura.clients.vision_cli import run_read

    asyncio.run(run_read(monitor))


@app.command()
def status() -> None:
    """Print AURA Core status."""
    from aura.clients.cli import run_status

    asyncio.run(run_status())


@app.command()
def say(
    text: str = typer.Argument(..., help="Text to speak."),
    save: str = typer.Option("", "--save", help="Also write a WAV file here."),
) -> None:
    """Speak text with the configured TTS voice.

    The smallest useful slice of Phase 2: proves the voice loads, synthesises
    and reaches the speakers, without needing a microphone or the model server.
    """
    import time as _time

    from aura.voice.factory import build_tts

    cfg = get_settings()
    configure_logging(level=cfg.logging.level, fmt=cfg.logging.format)

    tts = build_tts(cfg)
    started = _time.perf_counter()
    tts.load()
    load_ms = int((_time.perf_counter() - started) * 1000)

    chunks: list[Any] = []
    started = _time.perf_counter()
    first_ms = 0
    for chunk in tts.synthesize(text):
        if not chunks:
            first_ms = int((_time.perf_counter() - started) * 1000)
        chunks.append(chunk)
    if not chunks:
        console.print("[yellow]Nothing speakable in that text.[/yellow]")
        raise typer.Exit(1)

    synth_ms = int((_time.perf_counter() - started) * 1000)
    audio_s = sum(c.duration_s for c in chunks)
    console.print(
        f"[dim]{tts.info().model}  load {load_ms} ms  first chunk {first_ms} ms  "
        f"{synth_ms} ms for {audio_s:.1f} s audio  "
        f"(RTF {synth_ms / 1000 / max(audio_s, 0.001):.2f})[/dim]"
    )

    if save:
        _write_wav(Path(save), chunks)
        console.print(f"[green]Wrote[/green] {save}")

    from aura.voice.audio.playback import Speaker

    speaker = Speaker(device=cfg.voice.output_device or None)
    try:
        for chunk in chunks:
            speaker.play(chunk)
        speaker.wait_until_idle(timeout=audio_s + 5)
    finally:
        speaker.close()


def _write_wav(path: Path, chunks: list[Any]) -> None:
    import wave

    import numpy as np

    audio = np.concatenate([c.audio for c in chunks])
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(chunks[0].sample_rate)
        wf.writeframes(pcm.tobytes())


@app.command()
def listen(
    seconds: float = typer.Option(0.0, help="Record a fixed duration instead of using VAD."),
) -> None:
    """Transcribe one utterance from the microphone.

    Wake word not involved: this isolates mic -> VAD -> STT so a problem can be
    attributed to one stage.
    """
    import numpy as np

    from aura.voice.audio.capture import Microphone
    from aura.voice.factory import build_stt
    from aura.voice.vad import Endpointer

    cfg = get_settings()
    configure_logging(level=cfg.logging.level, fmt=cfg.logging.format)

    stt = build_stt(cfg)
    console.print("[dim]loading model...[/dim]")
    stt.load()

    endpointer = Endpointer(
        silence_s=cfg.voice.endpoint_silence_s,
        max_duration_s=seconds or cfg.voice.max_utterance_s,
        start_timeout_s=cfg.voice.listen_timeout_s,
    )
    endpointer.vad.load()

    console.print("[bold green]Speak now.[/bold green] [dim](silence ends it)[/dim]")
    mic = Microphone(device=cfg.voice.input_device or None)
    collector = endpointer.begin()
    audio = None
    with mic:
        for frame in mic.frames(timeout=1.0):
            segment = collector.push(frame)
            if segment is not None:
                audio = segment.audio
                if not segment.has_speech:
                    console.print("[yellow]No speech detected.[/yellow]")
                    raise typer.Exit(1)
                console.print(
                    f"[dim]{len(audio) / 16000:.1f} s captured, "
                    f"{segment.speech_s:.1f} s speech, ended by {segment.ended_by}[/dim]"
                )
                break

    assert audio is not None
    result = stt.transcribe(np.asarray(audio))
    if result.is_empty:
        console.print("[yellow]Nothing transcribed.[/yellow]")
        raise typer.Exit(1)
    console.print(f"\n[bold]{result.text}[/bold]")
    console.print(f"[dim]{result.latency_ms} ms  confidence {result.confidence:.2f}[/dim]")


@app.command()
def voice() -> None:
    """Run the voice loop: wake word, listen, answer aloud.

    A separate process from `aura serve`, talking to it over the same
    authenticated HTTP API any other client uses.
    """
    from aura.voice.core_client import CoreClient
    from aura.voice.factory import build_stt, build_tts, build_wake
    from aura.voice.session import VoiceSession
    from aura.voice.types import VoiceState

    cfg = get_settings()
    configure_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        log_file=cfg.logging.resolved_file(),
        redact_keys=tuple(cfg.logging.redact_keys),
    )

    core = CoreClient(cfg)
    if not core.health():
        console.print(
            f"[red]AURA Core is not running[/red] at {core.base_url}.\n"
            "Start it in another terminal:  [bold]aura serve[/bold]"
        )
        raise typer.Exit(1)

    wake, stt, tts = build_wake(cfg), build_stt(cfg), build_tts(cfg)

    console.print(f"[bold]AURA voice[/bold]  wake word: [cyan]{cfg.voice.wake.model}[/cyan]")
    console.print(f"[dim]{stt.info().model} -> core -> {tts.info().model}[/dim]")
    console.print("[dim]loading models...[/dim]")
    tts.load()
    stt.load()

    icons = {
        VoiceState.SLEEP: "[dim]zzz[/dim]",
        VoiceState.LISTENING: "[bold green]listening[/bold green]",
        VoiceState.FOLLOW_UP: (
            "[bold green]still listening[/bold green] [dim](no wake word needed)[/dim]"
        ),
        VoiceState.THINKING: "[yellow]thinking[/yellow]",
        VoiceState.SPEAKING: "[cyan]speaking[/cyan]",
        VoiceState.ERROR: "[red]error[/red]",
    }

    def on_state(state: VoiceState) -> None:
        console.print(f"  {icons.get(state, state.value)}")

    def on_text(who: str, text: str) -> None:
        if who == "user":
            console.print(f"[bold cyan]you  >[/bold cyan] {text}")
        elif who == "aura":
            console.print(f"[bold magenta]aura >[/bold magenta] {text}")

    session = VoiceSession(cfg, wake, stt, tts, core, on_state=on_state, on_text=on_text)
    follow = (
        f"Follow-ups stay open for {cfg.voice.follow_up_timeout_s:.0f}s after each answer."
        if cfg.voice.follow_up_enabled
        else "Follow-up mode is off; every question needs the wake word."
    )
    console.print(
        f'\n[bold green]Ready.[/bold green] Say "{cfg.voice.wake.model.replace("_", " ")}". '
        f"[dim]{follow} Ctrl+C to stop.[/dim]\n"
    )
    try:
        session.run()
    except KeyboardInterrupt:
        console.print("\n[dim]stopping...[/dim]")
        session.stop()


@app.command()
def devices() -> None:
    """List audio input and output devices."""
    from aura.voice.audio.capture import list_devices
    from aura.voice.audio.playback import list_output_devices

    for title, rows in (("Input", list_devices()), ("Output", list_output_devices())):
        console.print(f"\n[bold]{title}[/bold]")
        for d in rows:
            mark = "[green]*[/green]" if d["default"] else " "
            console.print(f"  {mark} {d['index']:>3}  {d['name'][:48]:48} {d['sample_rate']} Hz")
    console.print(
        "\n[dim]* system default. Override with voice.input_device / "
        "voice.output_device in config.[/dim]"
    )


@app.command()
def calibrate() -> None:
    """Measure this machine's speaker-to-microphone bleed.

    Barge-in has to tell "the user is talking" from "the microphone can hear
    the speakers". Those differ only in loudness, and the gap between them
    depends on the laptop, the speaker volume and where you sit -- so a
    hard-coded threshold is a guess that is wrong on most machines.

    A default that sat *below* the echo floor once made AURA interrupt itself,
    transcribe its own reply, answer it, and repeat. This measures both levels
    instead of guessing.
    """
    import time as _time

    import numpy as np

    from aura.voice.audio.capture import Microphone
    from aura.voice.audio.playback import Speaker
    from aura.voice.factory import build_tts

    cfg = get_settings()
    configure_logging(level="WARNING", fmt=cfg.logging.format)

    def rms_of(frames: list[Any]) -> list[float]:
        return [float(np.sqrt(np.mean(np.square(f)))) for f in frames if len(f)]

    console.print("[bold]AURA[/bold] barge-in calibration\n")

    tts = build_tts(cfg)
    tts.load()
    speaker = Speaker(device=cfg.voice.output_device or None)
    mic = Microphone(device=cfg.voice.input_device or None)

    # --- 1. echo floor: how loud AURA sounds to its own microphone ---------
    console.print("[cyan]1/2[/cyan] Measuring echo. [bold]Stay quiet[/bold] while AURA speaks.")
    line = (
        "I am measuring how loudly my own voice reaches the microphone. "
        "Please stay quiet until I finish speaking."
    )
    echo_frames: list[Any] = []
    with mic:
        _time.sleep(0.3)
        for chunk in tts.synthesize(line):
            speaker.play(chunk)
        deadline = _time.monotonic() + 12.0
        while (speaker.is_playing or not echo_frames) and _time.monotonic() < deadline:
            frame = mic.read(timeout=0.3)
            if frame is not None:
                echo_frames.append(frame)
        speaker.wait_until_idle(timeout=3)
    speaker.close()

    echo = rms_of(echo_frames)
    if not echo:
        console.print("[red]No audio captured. Check the microphone.[/red]")
        raise typer.Exit(1)
    echo_sorted = sorted(echo)
    echo_p95 = echo_sorted[int(len(echo_sorted) * 0.95) - 1]
    echo_max = echo_sorted[-1]

    # --- 2. speech level: how loud you are ---------------------------------
    console.print("\n[cyan]2/2[/cyan] Now [bold]talk normally[/bold] for about five seconds.")
    console.print("[dim]Say anything - count to ten at your usual volume.[/dim]")
    speech_frames: list[Any] = []
    with mic:
        deadline = _time.monotonic() + 5.5
        while _time.monotonic() < deadline:
            frame = mic.read(timeout=0.3)
            if frame is not None:
                speech_frames.append(frame)

    speech = sorted(rms_of(speech_frames))
    if not speech:
        console.print("[red]No audio captured.[/red]")
        raise typer.Exit(1)
    # The loudest quarter: the parts where you were actually speaking rather
    # than pausing between words.
    loud = speech[int(len(speech) * 0.75) :]
    speech_typical = sum(loud) / len(loud)

    # --- 3. verdict --------------------------------------------------------
    console.print("\n[bold]Measured[/bold]")
    console.print(f"  echo from speakers   p95 {echo_p95:.4f}   max {echo_max:.4f}")
    console.print(f"  your speech          typical {speech_typical:.4f}")

    if speech_typical <= echo_max * 1.5:
        console.print(
            "\n[yellow]Your voice is not clearly louder than the echo.[/yellow]\n"
            "  No threshold can separate them reliably. Either use headphones, "
            "lower the speaker volume, or set voice.barge_in_enabled: false."
        )
        raise typer.Exit(1)

    # Midway between the two, in log space, so it sits well clear of both.
    recommended = round((echo_max * speech_typical) ** 0.5, 3)
    console.print(f"\n[bold green]Recommended[/bold green]  barge_in_min_rms: {recommended}")
    console.print(f"[dim]currently {cfg.voice.barge_in_min_rms}[/dim]")
    console.print(
        "\nSet it in [bold]config/default.yaml[/bold] under voice:, "
        "or export AURA__VOICE__BARGE_IN_MIN_RMS=" + str(recommended)
    )


@app.command()
def doctor() -> None:
    """Check that this machine can run AURA, and say what is missing."""
    from aura.runtime.resources import snapshot

    cfg = get_settings()
    res = snapshot()
    problems: list[str] = []

    console.print(f"[bold]AURA {__version__}[/bold]  doctor\n")

    py = sys.version_info
    py_ok = (3, 12) <= (py.major, py.minor) < (3, 13)
    _row("Python", f"{py.major}.{py.minor}.{py.micro}", py_ok)
    if not py_ok:
        problems.append(
            "Python 3.12 is required. Phase 2+ dependencies (torch, whisper) "
            "have no 3.13/3.14 wheels. Run: uv sync"
        )

    ram_ok = res.ram_percent < 90
    _row("RAM", f"{res.ram_used_gb:.1f} / {res.ram_total_gb:.1f} GB used", ram_ok)
    if res.ram_total_gb < 15:
        problems.append("Under 16 GB RAM. Expect to run the 4B model only.")
    if not ram_ok:
        # System RAM, not VRAM, is the binding constraint on this class of
        # machine, so a marker without a message would hide the most likely
        # cause of poor behaviour.
        problems.append(
            f"System RAM is {res.ram_percent:.0f}% used with only "
            f"{res.ram_total_gb - res.ram_used_gb:.1f} GB free. AURA will run, but "
            "voice and vision (Phases 2 and 5) need headroom - close some "
            "browser tabs before adding subsystems."
        )

    if res.vram_total_gb is not None:
        free = res.vram_total_gb - (res.vram_used_gb or 0)
        _row("VRAM", f"{free:.1f} GB free of {res.vram_total_gb:.1f} GB", free > 3.5)
        if free < 3.5:
            problems.append(
                f"Only {free:.1f} GB VRAM free; the default model needs ~3.5 GB. "
                "Close GPU-heavy applications."
            )
    else:
        _row("GPU", "no nvidia-smi found", False)
        problems.append("No NVIDIA GPU detected. AURA will fall back to slow CPU inference.")

    cat = get_model_catalogue()
    entry = cat.active_entry()
    weights = models_dir() / entry.filename
    _row("Model", f"{entry.display_name} ({entry.quantization})", weights.is_file())
    if not weights.is_file():
        problems.append(f"Weights missing at {weights}. Run: python scripts/download_model.py")

    _row("Provider", f"{cfg.llm.provider} -> {cfg.llm.base_url}", True)

    console.print()
    if problems:
        console.print("[bold yellow]Action required:[/bold yellow]")
        for p in problems:
            console.print(f"  - {p}")
        raise typer.Exit(1)
    console.print("[bold green]All checks passed.[/bold green]")


def _row(label: str, value: str, ok: bool) -> None:
    mark = "[green]OK [/green]" if ok else "[red]!! [/red]"
    console.print(f"  {mark} {label:<10} {value}")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
