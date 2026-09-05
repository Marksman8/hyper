"""End-to-end API behaviour, against the real ASGI app and the echo provider."""

from __future__ import annotations

import json

from httpx import AsyncClient

from aura.config import Settings

# -- auth --------------------------------------------------------------------


async def test_healthz_needs_no_credentials(anon_client: AsyncClient) -> None:
    resp = await anon_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_every_other_route_requires_a_token(anon_client: AsyncClient) -> None:
    for path in ("/v1/status", "/v1/memory", "/v1/sessions", "/v1/audit"):
        resp = await anon_client.get(path)
        assert resp.status_code == 401, path
        assert resp.headers.get("WWW-Authenticate") == "Bearer"


async def test_a_wrong_token_is_rejected(anon_client: AsyncClient) -> None:
    resp = await anon_client.get("/v1/status", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


async def test_non_bearer_scheme_is_rejected(anon_client: AsyncClient) -> None:
    resp = await anon_client.get("/v1/status", headers={"Authorization": "Basic abc"})
    assert resp.status_code == 401


async def test_responses_carry_a_correlation_id(client: AsyncClient) -> None:
    resp = await client.get("/v1/status")
    assert resp.headers["X-Request-ID"]


# -- status ------------------------------------------------------------------


async def test_status_reports_provider_and_resources(client: AsyncClient) -> None:
    data = (await client.get("/v1/status")).json()
    assert data["provider"]["available"] is True
    assert data["provider"]["name"] == "echo"
    assert data["resources"]["ram_total_gb"] > 0
    assert set(data["memory"]) == {"sessions", "turns", "facts", "audit"}


async def test_config_endpoint_redacts_the_token(client: AsyncClient) -> None:
    data = (await client.get("/v1/config")).json()
    assert data["security"]["api_token"] == "***redacted***"


# -- chat --------------------------------------------------------------------


async def test_chat_returns_a_reply_and_a_session(client: AsyncClient) -> None:
    resp = await client.post("/v1/chat", json={"message": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "[echo] hello"
    assert data["session_id"]
    assert data["provider"] == "echo"


async def test_conversation_persists_across_requests(client: AsyncClient) -> None:
    first = (await client.post("/v1/chat", json={"message": "one"})).json()
    session = first["session_id"]
    await client.post("/v1/chat", json={"message": "two", "session_id": session})

    turns = (await client.get(f"/v1/sessions/{session}")).json()
    assert [t["content"] for t in turns] == [
        "one",
        "[echo] one",
        "two",
        "[echo] two",
    ]


async def test_omitting_session_id_starts_a_new_conversation(
    client: AsyncClient,
) -> None:
    a = (await client.post("/v1/chat", json={"message": "x"})).json()
    b = (await client.post("/v1/chat", json={"message": "y"})).json()
    assert a["session_id"] != b["session_id"]


async def test_unknown_session_is_a_404(client: AsyncClient) -> None:
    resp = await client.post("/v1/chat", json={"message": "x", "session_id": "nope"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


async def test_empty_message_is_rejected(client: AsyncClient) -> None:
    assert (await client.post("/v1/chat", json={"message": ""})).status_code == 422


async def test_streaming_emits_sse_frames(client: AsyncClient) -> None:
    text = ""
    session_id = None
    async with client.stream(
        "POST", "/v1/chat", json={"message": "stream me", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            frame = json.loads(payload)
            if frame.get("type") == "delta":
                session_id = session_id or frame.get("session_id")
                text += frame["content"]

    assert text == "[echo] stream me"
    assert session_id, "client must learn the session id from the stream"


async def test_streamed_reply_is_persisted(client: AsyncClient) -> None:
    session_id = None
    async with client.stream(
        "POST", "/v1/chat", json={"message": "remember me", "stream": True}
    ) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                frame = json.loads(line[5:].strip())
                session_id = session_id or frame.get("session_id")

    assert session_id
    turns = (await client.get(f"/v1/sessions/{session_id}")).json()
    assert turns[-1]["content"] == "[echo] remember me"
    assert turns[-1]["role"] == "assistant"


async def test_openai_compatible_surface(client: AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "aura", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "[echo] hi"
    assert data["aura"]["session_id"]


# -- memory ------------------------------------------------------------------


async def test_memory_is_inspectable_and_deletable(client: AsyncClient) -> None:
    """Spec section 16: the user can always see and remove what AURA stores."""
    assert (await client.get("/v1/memory")).json() == []

    created = await client.post("/v1/memory", json={"key": "editor", "value": "VS Code"})
    assert created.status_code == 200

    facts = (await client.get("/v1/memory")).json()
    assert facts[0]["key"] == "editor"
    assert facts[0]["permanence"] == "long_term"

    assert (await client.delete("/v1/memory/editor")).status_code == 200
    assert (await client.get("/v1/memory")).json() == []


async def test_forgetting_something_unknown_is_a_404(client: AsyncClient) -> None:
    assert (await client.delete("/v1/memory/nothing")).status_code == 404


async def test_temporary_facts_are_marked_as_such(client: AsyncClient) -> None:
    await client.post(
        "/v1/memory",
        json={"key": "tone", "value": "terse", "permanence": "temporary"},
    )
    facts = (await client.get("/v1/memory")).json()
    assert facts[0]["permanence"] == "temporary"


async def test_sessions_can_be_listed_and_deleted(client: AsyncClient) -> None:
    session = (await client.post("/v1/chat", json={"message": "x"})).json()["session_id"]
    listed = (await client.get("/v1/sessions")).json()
    assert any(s["id"] == session for s in listed)

    assert (await client.delete(f"/v1/sessions/{session}")).status_code == 200
    assert (await client.delete(f"/v1/sessions/{session}")).status_code == 404


# -- control -----------------------------------------------------------------


async def test_emergency_stop_halts_generation(client: AsyncClient) -> None:
    assert (await client.post("/v1/control/estop", json={"engaged": True})).status_code == 200

    resp = await client.post("/v1/chat", json={"message": "hello"})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "emergency_stop"

    await client.post("/v1/control/estop", json={"engaged": False})
    assert (await client.post("/v1/chat", json={"message": "hello"})).status_code == 200


async def test_audit_records_every_turn(client: AsyncClient) -> None:
    await client.post("/v1/chat", json={"message": "audited"})
    rows = (await client.get("/v1/audit")).json()
    actions = [r["action"] for r in rows]
    assert "chat" in actions
    chat_row = next(r for r in rows if r["action"] == "chat")
    assert chat_row["detail"]["provider"] == "echo"
    assert chat_row["session_id"]


async def test_emergency_stop_is_audited(client: AsyncClient) -> None:
    await client.post("/v1/control/estop", json={"engaged": True})
    rows = (await client.get("/v1/audit")).json()
    assert any(r["action"] == "control.emergency_stop" for r in rows)


# -- provider swapping -------------------------------------------------------


async def test_provider_is_selected_by_config(settings: Settings) -> None:
    """The seam, exercised: changing one config value changes the backend."""
    from aura.models.registry import build_provider

    assert build_provider(settings).name == "echo"
    swapped = settings.model_copy(
        update={"llm": settings.llm.model_copy(update={"provider": "openai_compat"})}
    )
    assert build_provider(swapped).name == "openai_compat"
