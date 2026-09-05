"""Loading a cached model must not touch the network.

Spec section 4: AURA does not require internet access for normal operation.

faster-whisper breaks that by default -- it resolves a model name through
`snapshot_download`, which contacts Hugging Face to check for updates on every
load, cached or not. Observed live: a voice session hung indefinitely inside
`socket.create_connection` with a complete 141 MB model already on disk.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from aura.voice.stt.faster_whisper import FasterWhisperSTT


class _FakeWhisper:
    """Stands in for WhisperModel, recording how it was constructed."""

    calls: ClassVar[list[dict[str, Any]]] = []
    fail_offline: ClassVar[bool] = False

    def __init__(self, model: str, **kwargs: Any) -> None:
        call = {"model": model, **kwargs}
        _FakeWhisper.calls.append(call)
        if kwargs.get("local_files_only") and _FakeWhisper.fail_offline:
            raise OSError("not in local cache")


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    import faster_whisper

    _FakeWhisper.calls = []
    _FakeWhisper.fail_offline = False
    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisper)


def test_cached_load_asks_for_local_files_only() -> None:
    """The first attempt must never be allowed to reach the network."""
    stt = FasterWhisperSTT(model="base.en")
    stt.load()

    assert len(_FakeWhisper.calls) == 1
    assert _FakeWhisper.calls[0]["local_files_only"] is True


def test_falls_back_to_download_when_not_cached() -> None:
    """A first run still works; it just says what it is doing."""
    _FakeWhisper.fail_offline = True
    stt = FasterWhisperSTT(model="base.en")
    stt.load()

    assert len(_FakeWhisper.calls) == 2
    assert _FakeWhisper.calls[0]["local_files_only"] is True
    # The retry must not pin local_files_only, or the download cannot happen.
    assert not _FakeWhisper.calls[1].get("local_files_only")


def test_load_is_idempotent() -> None:
    stt = FasterWhisperSTT(model="base.en")
    stt.load()
    stt.load()
    assert len(_FakeWhisper.calls) == 1


def test_unload_allows_reload() -> None:
    stt = FasterWhisperSTT(model="base.en")
    stt.load()
    assert stt.is_loaded
    stt.unload()
    assert not stt.is_loaded
    stt.load()
    assert len(_FakeWhisper.calls) == 2


def test_configured_options_reach_the_model() -> None:
    stt = FasterWhisperSTT(model="small.en", device="cuda", compute_type="float16")
    stt.load()
    call = _FakeWhisper.calls[0]
    assert call["model"] == "small.en"
    assert call["device"] == "cuda"
    assert call["compute_type"] == "float16"
