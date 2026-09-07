"""The desktop shell's HTTP client.

The widget code needs a display and is exercised by running it; this covers the
part that has real logic. The property that matters: every failure becomes a
value the UI can render, because a network blip must not take the window down.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from aura.desktop.client import CoreClient

BASE = "http://127.0.0.1:8760"


def _status_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "aura",
        "version": "0.1.0",
        "uptime_s": 12.0,
        "emergency_stop": False,
        "provider": {
            "name": "openai_compat",
            "kind": "http",
            "endpoint": "http://127.0.0.1:8081/v1",
            "model": "aura-local",
            "available": True,
            "detail": "ok",
            "latency_ms": 3,
        },
        "memory": {"sessions": 1, "turns": 2, "facts": 0, "audit": 5},
        "resources": {
            "cpu_percent": 4.0,
            "ram_used_gb": 9.0,
            "ram_total_gb": 15.7,
            "ram_percent": 57.0,
            "process_ram_mb": 70.0,
            "vram_used_gb": 3.1,
            "vram_total_gb": 8.0,
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def client() -> CoreClient:
    return CoreClient()


# -- status -----------------------------------------------------------------


@respx.mock
def test_reads_status(client: CoreClient) -> None:
    respx.get(BASE + "/v1/status").mock(return_value=httpx.Response(200, json=_status_payload()))
    status = client.status()

    assert status.online
    assert status.model == "aura-local"
    assert status.model_available
    assert status.ram_percent == 57.0
    assert status.vram_total_gb == 8.0


@respx.mock
def test_core_being_down_is_a_value_not_an_exception(client: CoreClient) -> None:
    """The overlay shows 'offline'; it does not crash."""
    respx.get(BASE + "/v1/status").mock(side_effect=httpx.ConnectError("refused"))
    status = client.status()

    assert status.online is False
    assert status.detail


@respx.mock
def test_emergency_stop_is_reflected(client: CoreClient) -> None:
    respx.get(BASE + "/v1/status").mock(
        return_value=httpx.Response(200, json=_status_payload(emergency_stop=True))
    )
    assert client.status().emergency_stop is True


# -- tasks ------------------------------------------------------------------


@respx.mock
def test_runs_a_task_and_flattens_the_tool_trace(client: CoreClient) -> None:
    respx.post(BASE + "/v1/agent").mock(
        return_value=httpx.Response(
            200,
            json={
                "session_id": "s1",
                "content": "Python 3.12",
                "model": "aura-local",
                "provider": "openai_compat",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
                "latency_ms": 2400,
                "tool_calls_made": 1,
                "steps": [
                    {
                        "index": 0,
                        "kind": "tool",
                        "content": "",
                        "tools": [
                            {
                                "name": "read_file",
                                "arguments": {"path": "pyproject.toml"},
                                "ok": True,
                                "decision": "allow",
                            }
                        ],
                        "latency_ms": 2000,
                    },
                    {"index": 1, "kind": "answer", "content": "Python 3.12", "tools": []},
                ],
                "stopped_early": False,
                "stop_reason": "complete",
                "pending_confirmations": [],
            },
        )
    )
    result = client.run_task("what python version?")

    assert result.ok
    assert result.content == "Python 3.12"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "read_file"
    assert result.session_id == "s1"


@respx.mock
def test_pending_confirmations_reach_the_caller(client: CoreClient) -> None:
    """This is what triggers the approval dialog."""
    respx.post(BASE + "/v1/agent").mock(
        return_value=httpx.Response(
            200,
            json={
                "session_id": "s1",
                "content": "I need approval.",
                "model": "m",
                "provider": "p",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms": 900,
                "tool_calls_made": 1,
                "steps": [],
                "stopped_early": False,
                "stop_reason": "complete",
                "pending_confirmations": ["write_file"],
            },
        )
    )
    assert client.run_task("write a file").pending_confirmations == ["write_file"]


@respx.mock
def test_approvals_are_sent_as_given(client: CoreClient) -> None:
    """The shell must not silently widen what the user approved."""
    route = respx.post(BASE + "/v1/agent").mock(
        return_value=httpx.Response(
            200,
            json={
                "session_id": "s1",
                "content": "done",
                "model": "m",
                "provider": "p",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "latency_ms": 10,
                "tool_calls_made": 1,
                "steps": [],
                "stopped_early": False,
                "stop_reason": "complete",
                "pending_confirmations": [],
            },
        )
    )
    client.run_task("write it", approve=["write_file"])

    sent = route.calls[0].request.read().decode()
    assert '"write_file"' in sent
    assert "run_python" not in sent


@respx.mock
def test_a_server_error_becomes_a_readable_message(client: CoreClient) -> None:
    respx.post(BASE + "/v1/agent").mock(
        return_value=httpx.Response(
            503, json={"error": {"code": "emergency_stop", "message": "Emergency stop engaged."}}
        )
    )
    result = client.run_task("anything")

    assert result.ok is False
    assert "Emergency stop" in result.error


@respx.mock
def test_an_unreachable_core_during_a_task_is_a_value(client: CoreClient) -> None:
    respx.post(BASE + "/v1/agent").mock(side_effect=httpx.ConnectError("refused"))
    result = client.run_task("anything")

    assert result.ok is False
    assert "Cannot reach AURA Core" in result.error


# -- controls ---------------------------------------------------------------


@respx.mock
def test_estop_returns_the_state_actually_in_force(client: CoreClient) -> None:
    """So the button can never show a state the server does not agree with."""
    respx.post(BASE + "/v1/control/estop").mock(
        return_value=httpx.Response(200, json={"emergency_stop": True})
    )
    assert client.set_emergency_stop(True) is True


@respx.mock
def test_a_failed_estop_reports_the_unchanged_state(client: CoreClient) -> None:
    respx.post(BASE + "/v1/control/estop").mock(side_effect=httpx.ConnectError("refused"))
    # Asked to engage and could not, so it is still released.
    assert client.set_emergency_stop(True) is False
