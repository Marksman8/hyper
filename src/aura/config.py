"""Configuration system.

Layered, with strictly increasing precedence:

    1. config/default.yaml      committed defaults
    2. config/local.yaml        gitignored per-machine overrides
    3. .env / process env       AURA__SECTION__FIELD
    4. explicit constructor kwargs

Rationale for this shape rather than plain env vars: the config is large and
structured (spec sections 5, 8, 10 and 26 all add sections later), and a
commented YAML file is the only form of it that stays readable at that size.
Env vars remain the override path because that is how secrets and CI arrive.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from aura.errors import ConfigError

# Project root: src/aura/config.py -> src/aura -> src -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

_REDACTED = "***redacted***"


class _Section(BaseModel):
    """Base for config sections.

    Frozen, so a section cannot be mutated through a loaded Settings object.
    Without this, `settings.api.port = 0` succeeds even though Settings
    itself is frozen, and a subsystem can quietly change global config at
    runtime.
    """

    model_config = ConfigDict(frozen=True)


class AppConfig(_Section):
    name: str = "aura"
    version: str = "0.1.0"
    emergency_stop: bool = False


class ApiConfig(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8760, ge=1024, le=65535)
    cors_origins: list[str] = Field(default_factory=list)
    request_timeout_s: int = Field(default=300, gt=0)

    @field_validator("host")
    @classmethod
    def _loopback_only(cls, v: str) -> str:
        # AURA's threat model assumes it is reachable only by the local user.
        # Binding to a wildcard address quietly exposes memory, audit logs and
        # (from Phase 3) tool execution to anything on the network. Refuse.
        if v not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError(
                "api.host must be a loopback address, got "
                + repr(v)
                + ". AURA must not be exposed beyond this machine."
            )
        return v


class LLMConfig(_Section):
    provider: str = "openai_compat"
    base_url: str = "http://127.0.0.1:8081/v1"
    api_key: str = ""
    # Where the model server's shared secret lives when api_key is not set
    # explicitly. llama-server has no auth by default and allows all CORS
    # origins, so any web page you visit can reach 127.0.0.1:8081 and use your
    # GPU. start_model_server.ps1 writes this file; AURA reads it.
    api_key_file: str = "var/secrets/model_api_token"
    model: str = "default"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: int = Field(default=180, gt=0)
    connect_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    def resolved_api_key(self) -> str:
        """The model server secret: explicit config, else the key file, else none.

        Absent is tolerated (a bare llama-server with no --api-key still works),
        so upgrading an existing setup does not break it.
        """
        if self.api_key:
            return self.api_key
        path = Path(self.api_key_file)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if path.is_file():
            # utf-8-sig, not utf-8: this file may be written by a PowerShell
            # script, and Windows PowerShell 5.1 emits a UTF-8 BOM. Leaving the
            # BOM in the token yields a 401 that looks like a wrong secret.
            return path.read_text(encoding="utf-8-sig").strip()
        return ""


class MemoryConfig(_Section):
    db_path: str = "var/data/aura.db"
    history_turns: int = Field(default=12, ge=0, le=200)
    history_char_budget: int = Field(default=12000, gt=0)
    # Phase 6. Notice durable preferences from conversation. Runs after the
    # reply, never in the response path. What it learns starts below the
    # injection floor and needs corroboration on a separate occasion before it
    # shapes anything -- see memory/beliefs.py and spec section 16.
    learn_from_conversation: bool = True
    # A fact is put into the system prompt at or above this confidence. It is
    # the line between "AURA noticed" and "AURA believes", so raising it makes
    # AURA more sceptical, not merely quieter.
    injection_floor: float = Field(default=0.5, ge=0.0, le=1.0)
    # How many past turns a memory search may pull back into context.
    search_results: int = Field(default=4, ge=0, le=25)

    def resolved_db_path(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p


class SecurityConfig(_Section):
    api_token: str = ""
    token_file: str = "var/secrets/api_token"  # noqa: S105 - a path, not a secret
    audit_enabled: bool = True
    require_audit: bool = True

    def resolved_token_file(self) -> Path:
        p = Path(self.token_file)
        return p if p.is_absolute() else PROJECT_ROOT / p


class RuntimeConfig(_Section):
    manage_model_server: bool = False
    resource_poll_s: int = Field(default=10, gt=0)


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json"] = "console"
    file: str = "var/logs/aura.log"
    redact_keys: list[str] = Field(
        # Matched exactly, not as substrings, so every new secret-bearing field
        # has to be added here by hand. `auth_token` is the automation service
        # shared secret (ADR-0004 D7).
        default_factory=lambda: [
            "api_token",
            "api_key",
            "password",
            "secret",
            "token",
            "auth_token",
        ]
    )

    def resolved_file(self) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else PROJECT_ROOT / p


class WakeConfig(_Section):
    model: str = "hey_jarvis"
    threshold: float = Field(default=0.5, gt=0.0, le=1.0)
    # Frames of cooldown after a detection, so one utterance of the wake word
    # cannot fire twice.
    cooldown_s: float = Field(default=2.0, ge=0.0)


class STTConfig(_Section):
    provider: str = "faster_whisper"
    # base.en int8 on CPU: ~150 MB, ~300 ms for a 3 s utterance on 16 threads.
    # small.en roughly triples both. GPU needs cuDNN on Windows - see
    # docs/architecture/01-phase2-voice.md.
    model: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    # Drop transcripts the decoder was unsure about. Whisper's avg_logprob
    # is around -0.2 for clean speech and -0.8 to -1.2 for a normal short
    # utterance in a real room; only clear garbage falls below -1.5. An
    # earlier default of -1.0 silently discarded genuine speech, which is
    # the worst failure mode available here: the user talks, AURA hears,
    # and nothing happens.
    min_confidence: float = Field(default=-1.5, le=0.0)
    beam_size: int = Field(default=1, ge=1, le=10)


class TTSConfig(_Section):
    provider: str = "piper"
    voice: str = "en_US-amy-medium"
    # >1.0 is slower speech. Piper's own default is 1.0.
    length_scale: float = Field(default=1.0, gt=0.0, le=3.0)
    volume: float = Field(default=1.0, gt=0.0, le=2.0)


class PersonalityConfig(_Section):
    """Phase 10B.5. Micro-expressions, and the restraint on them.

    Off is a supported setting and a first-class one: the brief asked
    specifically that foreign words not leak into every response, so the way
    to switch this off must be obvious and must actually silence it rather
    than merely reduce it.
    """

    enabled: bool = True
    # off | low (~1 in 8) | medium (~1 in 4). LOW is the default because a
    # micro-expression is a garnish, and the failure mode of a garnish is
    # applying it to everything.
    frequency: str = "low"
    # Which phrase sets are in play. Japanese only by default -- adding a
    # language multiplies how often *something* fits, which is a frequency
    # increase by the back door.
    languages: list[str] = Field(default_factory=lambda: ["ja"])
    # Hard floor between any two, whatever the dice say. Frequency governs the
    # long-run average; this governs how it feels during a burst of short
    # turns, which is what a person actually notices.
    min_gap_s: float = Field(default=90.0, ge=0.0)


class VoiceConfig(_Section):
    """Phase 2. Consumed only by the voice process, never by AURA Core."""

    enabled: bool = True
    input_device: str = ""
    output_device: str = ""
    # A short blip when AURA starts listening, and a lower one when it gives up
    # without answering. Without a cue the only signal that the wake word fired
    # is a log line, and the honest user question during the seconds before the
    # first spoken word is "is this thing even on?".
    wake_cue: bool = True
    cue_volume: float = Field(default=0.18, gt=0.0, le=1.0)
    # After answering, keep listening briefly so a follow-up question needs no
    # wake word. Without this every question begins with "hey jarvis", which
    # makes AURA feel like a vending machine rather than a conversation.
    # The cost is an open microphone after each reply, so it is opt-out.
    follow_up_enabled: bool = True
    follow_up_timeout_s: float = Field(default=7.0, gt=0.0, le=60.0)
    # Interrupting with one of these cancels the answer instead of being
    # treated as the next question. Without it, saying "stop" cuts the reply
    # off and then asks the model what "stop" means.
    stop_words: list[str] = Field(
        default_factory=lambda: [
            # Whisper punctuates and capitalises freely, and people phrase this
            # a dozen ways under mild irritation. Matching is on the whole
            # utterance, so a long list costs nothing and a short one silently
            # sends "stop please" to the model as a question.
            "stop",
            "stop it",
            "stop aura",
            "stop talking",
            "stop please",
            "please stop",
            "stopped",
            "no stop",
            "stop stop",
            "shut up",
            "be quiet",
            "quiet",
            "cancel",
            "cancel that",
            "cancel it",
            "never mind",
            "nevermind",
            "forget it",
            "forget that",
            "enough",
            "that's enough",
            "thats enough",
            "okay stop",
            "ok stop",
            "wait",
            "wait wait",
            "hold on",
        ]
    )
    # Seconds of audio kept before the wake word fires, so a request run into
    # the trigger phrase does not lose its first word.
    preroll_s: float = Field(default=1.5, ge=0.0, le=5.0)
    # Silence that ends an utterance.
    endpoint_silence_s: float = Field(default=0.6, gt=0.0, le=5.0)
    # Hard cap on one utterance, so a stuck VAD cannot listen forever.
    max_utterance_s: float = Field(default=20.0, gt=0.0)
    # How long LISTENING waits for speech to start before giving up on a
    # false wake.
    listen_timeout_s: float = Field(default=6.0, gt=0.0)
    # Speech during playback for this long interrupts it.
    barge_in_s: float = Field(default=0.35, gt=0.0, le=2.0)
    barge_in_enabled: bool = True
    # Without acoustic echo cancellation, AURA's own voice reaches the
    # microphone and reads as speech. These two guards keep it from
    # interrupting itself; headphones make both unnecessary.
    #   grace: ignore the microphone for this long after playback starts
    #   min_rms: require real loudness, since speaker bleed is much quieter
    #            than a person addressing the machine
    barge_in_grace_s: float = Field(default=1.0, ge=0.0, le=10.0)
    # How much louder than AURA's own echo the user has to be.
    #
    # A fixed absolute threshold cannot work here: the right value depends on
    # speaker volume, microphone gain and where the user is sitting. Set too
    # low it made AURA interrupt itself in a loop; set too high it ignored a
    # real person saying "stop" for eighteen seconds. Both were observed.
    #
    # So the echo is measured instead of assumed. During the grace period after
    # playback starts the user is almost certainly not talking yet, so the
    # microphone level in that window *is* the echo floor for this utterance,
    # under this volume, in this room. Interrupting then means exceeding it by
    # this factor.
    barge_in_echo_factor: float = Field(default=2.0, ge=1.0, le=20.0)
    # Absolute floor, for headphones and quiet rooms where the measured echo is
    # near zero and any multiple of it would trigger on a cough.
    barge_in_min_rms: float = Field(default=0.015, ge=0.0, le=1.0)
    # A feedback loop must be impossible by construction, not merely unlikely.
    # This many barge-ins in a row with no wake word between them means AURA is
    # talking to itself; barge-in is then disabled for the session.
    max_consecutive_barge_ins: int = Field(default=3, ge=1, le=20)
    # Unload STT after this long idle. On a 16 GB machine this is not an
    # optimisation, it is what lets AURA coexist with a browser.
    idle_unload_s: float = Field(default=180.0, ge=0.0)

    # 10B.7. Saying the wake word over AURA stops her.
    #
    # ON by default, and it is the interrupt that actually works on this
    # machine: it matches a PATTERN rather than a level, so AURA's own voice
    # coming back through the microphone cannot trigger it however loud the
    # coupling is. Barge-in by loudness has failed three times here in both
    # directions -- see docs/KNOWN-ISSUES.md.
    wake_interrupt: bool = True

    wake: WakeConfig = Field(default_factory=WakeConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)


class ToolsConfig(_Section):
    """Phase 3. What AURA may touch, and how far it may go unattended."""

    enabled: bool = True
    # Allow-list of directories tools may read or write. Empty means no path
    # is accessible at all - a deliberate default, since an unconfigured
    # assistant should be able to do nothing rather than everything.
    workspace_roots: list[str] = Field(default_factory=list)
    # Refused even inside a root. Empty means use the built-in list in
    # tools/workspace.py, which covers .env, var/secrets, keys and .ssh.
    deny_patterns: list[str] = Field(default_factory=list)
    # The highest permission level AURA may use without asking. LEVEL 1 is
    # read-only: writes and code execution need a human first.
    max_autonomous_level: int = Field(default=1, ge=0, le=4)
    # The ceiling a human confirmation can raise it to. LEVEL 4 stays out of
    # reach entirely until there is a UI that can explain what it is approving.
    max_confirmable_level: int = Field(default=3, ge=0, le=4)
    python_enabled: bool = True
    # Network egress. Off by default and deliberately a separate switch from
    # the permission ceiling: a level is about *who may authorise* an action,
    # and this is about whether the capability exists on this machine at all.
    # Once a request leaves, it has left -- no guard can call it back.
    web_enabled: bool = False
    default_timeout_s: float = Field(default=30.0, gt=0.0, le=600.0)

    def resolved_roots(self) -> tuple[str, ...]:
        """Workspace roots as absolute paths."""
        out: list[str] = []
        for raw in self.workspace_roots:
            path = Path(raw).expanduser()
            out.append(str(path if path.is_absolute() else PROJECT_ROOT / path))
        return tuple(out)


class AutomationConfig(_Section):
    """Third-party automation (ADR-0004).

    **Nothing here grants a permission.** Levels live in the manifest, which is
    version-controlled and reviewable in a diff; this section only says which
    provider to use and how to reach it. `enabled: false` is the default
    because AURA must be fully functional with no automation backend at all.
    """

    enabled: bool = False
    provider: str = "n8n"
    # The authority. Relative paths resolve against the project root.
    manifest_path: str = "config/workflows.yaml"
    # Used only for the health check; each workflow carries its own full URL,
    # so a manifest entry cannot be redirected by changing this.
    base_url: str = "http://127.0.0.1:5678"
    # A shared secret authenticating AURA *to* the automation service. Not a
    # third-party credential -- those stay in n8n and never enter AURA (D7).
    auth_header: str = "X-Aura-Token"
    auth_token: str = ""
    auth_token_file: str = "var/secrets/automation_token"  # noqa: S105  (a path)
    verify_tls: bool = True


class AgentConfig(_Section):
    """Phase 4. The budget the loop runs under."""

    enabled: bool = True
    # A loop that decides for itself when to stop is a loop that sometimes
    # does not. Eight steps is enough for read-inspect-answer chains and short
    # of enough for a model to wander.
    max_steps: int = Field(default=8, ge=1, le=50)
    max_seconds: float = Field(default=180.0, gt=0.0, le=3600.0)
    # Lower than conversational temperature: choosing a tool and its arguments
    # is a decision, and creativity there produces malformed calls.
    tool_temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class PersonalizationConfig(_Section):
    """Phase 7. Acting on what Phase 6 learned.

    Adaptation changes behaviour every turn without the user asking for it. If
    it cannot be switched off it cannot be blamed, and anything that cannot be
    blamed cannot be fixed - hence the kill switch and the per-turn opt-out.
    """

    enabled: bool = True
    # Prompt directives follow any believed fact. Sampling changes follow only
    # explicit ones; see personalization/profile.py for why the two paths have
    # different evidence requirements. Turning this off keeps the directives.
    adapt_parameters: bool = True


class VLMConfig(_Section):
    """Phase 10C. The vision-language model.

    **Off by default, and that is a resource fact rather than caution.** The
    language model holds 3.07 GB of 7.96 GB VRAM on a GPU that also drives the
    display; a VLM needs roughly 3.2 GB more. They cannot both be resident, so
    running with no VLM is the normal state and everything above must work in
    it -- see `NullVLM`.
    """

    enabled: bool = False
    provider: str = "openai_compat"
    # A SECOND llama-server, deliberately: one process per model, so evicting
    # the VLM is stopping a process rather than unloading weights inside the
    # one AURA is mid-conversation with.
    base_url: str = "http://127.0.0.1:8082/v1"
    model: str = "default"
    api_key: str = ""
    api_key_file: str = "var/secrets/vlm_api_token"
    # Generous next to the LLM's, because a VLM on this hardware is estimated
    # at 3-8 seconds for a screen question and the estimate is not yet measured.
    timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)
    max_tokens: int = Field(default=512, gt=0, le=4096)
    # Measured on this machine with Qwen2.5-VL-3B, **on the GPU**, cold (no
    # prompt cache):
    #
    #     1920x1080  (2.07 MP)   4.53 s
    #     1536x864   (1.33 MP)   2.88 s
    #     1280x720   (0.92 MP)   2.02 s
    #     1024x576   (0.59 MP)   1.44 s
    #      768x432   (0.33 MP)   0.76 s
    #
    # Roughly linear at ~2.2 seconds per megapixel. A repeat of the SAME image
    # is 0.3-0.5 s at any size, because llama.cpp caches the image tokens --
    # which makes re-observing an unchanged screen nearly free, and matters to
    # 10C.4's trigger policy.
    #
    # 1280 is the default: 2 s cold, most UI text still legible, and full
    # resolution buys detail at 2.2x the cost. Raise it for a task that needs
    # small body text; the whole screen at 1920 is still only 4.5 s.
    #
    # **This number was briefly wrong.** The first measurement gave 83 s at
    # full resolution and set the cap at 768 -- but llama-server had silently
    # fallen back to CPU, so it measured a path AURA does not use. 10C.3 makes
    # that failure loud, because an 18x slowdown that announces itself as
    # nothing is worse than one that crashes.
    max_image_width: int = Field(default=1280, ge=128, le=3840)

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        path = Path(self.api_key_file)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if path.is_file():
            # utf-8-sig: a file written by Windows PowerShell carries a BOM,
            # and a BOM inside a token is a 401 that looks like a wrong secret.
            return path.read_text(encoding="utf-8-sig").strip()
        return ""


class VisionConfig(_Section):
    """Phase 5. What AURA may look at.

    The mode is a config value so that a machine can be set up permanently
    private, but it is also changeable at runtime: "don't look" has to take
    effect when it is said, not at the next restart.
    """

    # PRIVATE by default. Spec 10 - vision is off until the user turns it on.
    # A screen read is bounded by nothing, unlike a file read, so the safe
    # default is "no" rather than "yes with restrictions".
    mode: str = Field(default="private", pattern="^(private|assist|active|sleep)$")
    # Applications never captured, matched against the foreground window title
    # and process name.
    deny_apps: list[str] = Field(default_factory=list)
    # When non-empty, the only applications captured.
    allow_apps: list[str] = Field(default_factory=list)
    denied_monitors: list[int] = Field(default_factory=list)
    # Extra title patterns on top of the built-in credential list in
    # vision/privacy.py. Additive: the built-ins cannot be configured away.
    extra_deny_titles: list[str] = Field(default_factory=list)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    ocr_backend: str = Field(default="rapidocr", pattern="^(rapidocr|null)$")
    # Below this a recognised line is reported as uncertain rather than as text.
    ocr_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Cap on lines returned to the model. A full IDE screenshot can OCR to
    # hundreds of fragments, most of them chrome, and flooding the context with
    # them buries the answer the user asked for.
    max_lines: int = Field(default=200, ge=10, le=2000)
    # Detection resolution cap, long side. RapidOCR's own default does not
    # downscale a 1920x1080 screenshot at all, which measured 42s per read
    # on this machine. Lower is faster and blurrier; see vision/ocr/rapid.py.
    ocr_det_side: int = Field(default=1600, ge=480, le=4096)
    # onnxruntime threads. Bounded because AURA shares the machine with a
    # resident LLM and a voice loop that both care about latency.
    ocr_threads: int = Field(default=8, ge=1, le=32)
    change_threshold: float = Field(default=0.02, ge=0.0, le=1.0)


class PersonaConfig(_Section):
    name: str = "AURA"
    system_prompt: str = "You are AURA, a local-first personal AI assistant."
    # Appended when the reply will be spoken. Models write for the eye by
    # default: asked to compare two things they produce headings, bold runs and
    # bullet lists, which a synthesiser reads out as "hash hash" and "star
    # star". Stripping the markup afterwards fixes the symbols but not the
    # shape - a bulleted spec sheet read aloud is still a bad answer.
    # Prepended when a turn may use tools. Kept separate from the base prompt
    # so a tool-free conversation is not paying for instructions about tools
    # it cannot use.
    agent_system_prompt: str = (
        "You have tools available, listed below. Use them when they would "
        "answer the question better than guessing would.\n"
        "- These are what you can do right now. If something is not in the "
        "list, say you have no tool for it in this conversation -- not that "
        "you are incapable of it.\n"
        "- Look before you conclude. If asked about a file or a directory, "
        "read it rather than assuming what it contains.\n"
        "- One step at a time. Call a tool, read the result, then decide what "
        "to do next.\n"
        # Aligned with the observation `agent.py` feeds back on a refusal. This
        # line used to say "say what you wanted to do and why, and let the user
        # decide", which a 4B reads as "stop here" -- one gated tool then
        # cancelled every other part of a request, silently. The loop's wording
        # was fixed first; this is the other half, and the two must not drift.
        "- If a tool is REFUSED or DENIED, do not retry it. Carry on with "
        "everything else that was asked, then say what you are still waiting "
        "on and why.\n"
        "- When you have enough to answer, answer. Do not keep calling tools "
        "to be thorough."
    )
    voice_system_prompt: str = (
        "This reply will be spoken aloud, so write it to be heard, not read.\n"
        "- Plain sentences only. No markdown, headings, bullet points, "
        "asterisks, or code blocks.\n"
        "- Keep it short: two or three sentences unless more is genuinely "
        "needed. The listener cannot skim.\n"
        "- Spell out what a symbol would mean rather than using the symbol.\n"
        "- If something truly needs a list, say it as prose: 'there are three: "
        "first ..., second ..., third ...'."
    )


class _YamlSource(PydanticBaseSettingsSource):
    """Settings source that reads a YAML file, tolerating absence."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Unused: __call__ is overridden to return the whole mapping at once.
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        if not self._path.is_file():
            return {}
        try:
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError("Invalid YAML in " + str(self._path) + ": " + str(exc)) from exc
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ConfigError(str(self._path) + " must contain a mapping at the top level.")
        return raw


class Settings(BaseSettings):
    """Root configuration object. Immutable once loaded."""

    model_config = SettingsConfigDict(
        env_prefix="AURA__",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    personalization: PersonalizationConfig = Field(
        default_factory=PersonalizationConfig
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # First source wins in pydantic-settings, so this tuple is in
        # DESCENDING precedence order.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls, CONFIG_DIR / "local.yaml"),
            _YamlSource(settings_cls, CONFIG_DIR / "default.yaml"),
        )

    def redacted(self) -> dict[str, Any]:
        """Dump config with secrets removed.

        Every path that shows config to a human, a log, or a model goes through
        this. There is no other public serializer.
        """
        keys = {k.lower() for k in self.logging.redact_keys}

        def scrub(node: Any) -> Any:
            if isinstance(node, dict):
                return {
                    k: (_REDACTED if (k.lower() in keys and v) else scrub(v))
                    for k, v in node.items()
                }
            if isinstance(node, list):
                return [scrub(v) for v in node]
            return node

        result: dict[str, Any] = scrub(self.model_dump())
        return result


class ModelEntry(BaseModel):
    """One entry in the model catalogue."""

    display_name: str
    repo_id: str
    filename: str
    quantization: str = "Q4_K_M"
    context_length: int = 8192
    vram_gb: float = 0.0
    gpu_layers: int = -1
    supports_tools: bool = False
    notes: str = ""


class VisionModelEntry(ModelEntry):
    """A VLM entry. Same as a language model plus the vision projector.

    `mmproj` is not optional and has no default: a VLM GGUF without its
    projector loads happily and then cannot see, which is a failure that looks
    like a bad model rather than a missing file.
    """

    mmproj: str


class ModelCatalogue(BaseModel):
    active: str
    catalogue: dict[str, ModelEntry]
    # Phase 10C. Separate from `active`/`catalogue` rather than mixed in,
    # because these are served by a SECOND llama-server on its own port and
    # the two are never both resident -- see the header of models.yaml.
    active_vision: str = ""
    vision_catalogue: dict[str, VisionModelEntry] = Field(default_factory=dict)

    def active_vision_entry(self) -> VisionModelEntry:
        if not self.active_vision:
            raise ConfigError("No `active_vision` set in config/models.yaml.")
        try:
            return self.vision_catalogue[self.active_vision]
        except KeyError as exc:
            raise ConfigError(
                f"active_vision is {self.active_vision!r}, which is not in "
                f"vision_catalogue. Known: {sorted(self.vision_catalogue)}"
            ) from exc

    def active_entry(self) -> ModelEntry:
        try:
            return self.catalogue[self.active]
        except KeyError as exc:
            raise ConfigError(
                "models.yaml: active model "
                + repr(self.active)
                + " is not in the catalogue. Known: "
                + str(sorted(self.catalogue))
            ) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings. Call `get_settings.cache_clear()` in tests."""
    return Settings()


@lru_cache(maxsize=1)
def get_model_catalogue() -> ModelCatalogue:
    path = CONFIG_DIR / "models.yaml"
    if not path.is_file():
        raise ConfigError("Missing model catalogue at " + str(path))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelCatalogue.model_validate(raw)


def voice_models_dir() -> Path:
    """Where wake-word and TTS voice files live."""
    return models_dir() / "voice"


def models_dir() -> Path:
    """Where GGUF weights live. Overridable for users with a second drive."""
    override = os.environ.get("AURA__MODELS_DIR")
    return Path(override) if override else PROJECT_ROOT / "var" / "models"
