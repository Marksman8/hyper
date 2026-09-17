"""The agent loop.

Driven with a scripted provider rather than a model, so the tests assert on the
loop's behaviour instead of on whether a 4B model felt like calling a tool. The
properties that matter are the ones AURA owns and must not delegate: budgets,
the permission gate, the emergency stop, and feeding failure back rather than
raising it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from aura.core.agent import AgentLoop, approval_token, tool_name
from aura.core.types import (
    ChatChunk,
    ChatResult,
    GenerationParams,
    Message,
    ProviderHealth,
    ProviderInfo,
    Role,
    ToolInvocation,
    Usage,
)
from aura.errors import EmergencyStopError
from aura.models.base import LLMProvider


class ScriptedProvider(LLMProvider):
    """Returns a queued sequence of results, recording what it was sent."""

    def __init__(self, script: list[ChatResult]) -> None:
        self._script = list(script)
        self.calls: list[list[Message]] = []
        self.tools_seen: list[Any] = []

    @property
    def name(self) -> str:
        return "scripted"

    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self.name, kind="test", model="scripted-v1")

    async def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    async def chat(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ChatResult:
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        if self._script:
            return self._script.pop(0)
        return ChatResult(content="fallback answer", model="scripted-v1", provider=self.name)

    async def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> AsyncIterator[ChatChunk]:  # pragma: no cover - the loop never streams
        yield ChatChunk(delta="", done=True)


class FakeRunner:
    """A ToolRunner that records calls and returns scripted outcomes."""

    def __init__(self, outcomes: dict[str, tuple[bool, str, str]] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[tuple[str, dict[str, Any], bool]] = []

    def wire_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": "read_file", "description": "Read", "parameters": {}},
            }
        ]

    def needs_confirmation(self, name: str) -> bool:
        return name == "write_file"

    async def run(
        self, name: str, arguments: dict[str, Any], *, confirmed: bool
    ) -> tuple[bool, str, str]:
        self.calls.append((name, arguments, confirmed))
        if name in self.outcomes:
            return self.outcomes[name]
        return True, f"result of {name}", "allow"


def answer(text: str) -> ChatResult:
    return ChatResult(content=text, model="scripted-v1", provider="scripted")


def wants(name: str, **arguments: Any) -> ChatResult:
    return ChatResult(
        content="",
        model="scripted-v1",
        provider="scripted",
        finish_reason="tool_calls",
        tool_calls=(ToolInvocation(id="c1", name=name, arguments=arguments),),
    )


def start() -> list[Message]:
    return [Message(role=Role.USER, content="do the thing")]


# -- the happy path ---------------------------------------------------------


async def test_answers_directly_when_no_tool_is_needed() -> None:
    """A question that needs no tool must not cost a round trip."""
    provider = ScriptedProvider([answer("42")])
    runner = FakeRunner()
    outcome = await AgentLoop(provider, runner).run(start())

    assert outcome.content == "42"
    assert outcome.tool_calls_made == 0
    assert runner.calls == []
    assert outcome.stop_reason == "complete"


async def test_runs_a_tool_then_answers() -> None:
    provider = ScriptedProvider([wants("read_file", path="x.txt"), answer("it says hello")])
    runner = FakeRunner({"read_file": (True, "hello", "allow")})
    outcome = await AgentLoop(provider, runner).run(start())

    assert outcome.content == "it says hello"
    assert outcome.tool_calls_made == 1
    assert runner.calls[0][0] == "read_file"
    assert [s.kind for s in outcome.steps] == ["tool", "answer"]


async def test_chains_several_tools() -> None:
    provider = ScriptedProvider(
        [
            wants("list_directory", path="."),
            wants("read_file", path="a.py"),
            answer("done"),
        ]
    )
    runner = FakeRunner()
    outcome = await AgentLoop(provider, runner).run(start())

    assert outcome.tool_calls_made == 2
    assert [c[0] for c in runner.calls] == ["list_directory", "read_file"]


async def test_the_conversation_keeps_the_assistant_tool_turn() -> None:
    """Tool results reference a call the model must be able to see."""
    provider = ScriptedProvider([wants("read_file", path="x"), answer("ok")])
    messages = start()
    await AgentLoop(provider, FakeRunner()).run(messages)

    roles = [m.role for m in messages]
    assert Role.ASSISTANT in roles and Role.TOOL in roles
    assistant = next(m for m in messages if m.role is Role.ASSISTANT)
    tool_msg = next(m for m in messages if m.role is Role.TOOL)
    assert assistant.tool_calls, "the assistant turn lost its tool_calls"
    assert tool_msg.tool_call_id == assistant.tool_calls[0].id


# -- the permission gate ----------------------------------------------------


async def test_an_unapproved_tool_is_refused_and_reported() -> None:
    """The loop cannot approve anything; the refusal comes back as an observation."""
    provider = ScriptedProvider(
        [wants("write_file", path="x", content="y"), answer("I need your approval.")]
    )
    runner = FakeRunner({"write_file": (False, "needs explicit confirmation", "confirm")})
    outcome = await AgentLoop(provider, runner).run(start())

    assert runner.calls[0][2] is False, "the loop confirmed a call by itself"
    assert [tool_name(t) for t in outcome.pending_confirmations] == ["write_file"]


async def test_only_a_human_supplied_approval_confirms_a_call() -> None:
    args = {"path": "x", "content": "y"}
    provider = ScriptedProvider([wants("write_file", **args), answer("done")])
    runner = FakeRunner()
    await AgentLoop(
        provider, runner, approved_tools=[approval_token("write_file", args)]
    ).run(start())

    assert runner.calls[0][2] is True


async def test_approving_one_tool_does_not_approve_another() -> None:
    provider = ScriptedProvider([wants("run_python", code="x"), answer("no")])
    runner = FakeRunner({"run_python": (False, "needs confirmation", "confirm")})
    await AgentLoop(
        provider,
        runner,
        approved_tools=[approval_token("write_file", {"path": "x", "content": "y"})],
    ).run(start())

    assert runner.calls[0][2] is False


# -- security audit 01, finding 2 -------------------------------------------


async def test_an_approval_does_not_carry_to_different_arguments() -> None:
    """Audit 01 finding 2. The escalation this whole binding exists to stop.

    Consent is collected per call *with arguments* -- `executor.py::_describe`
    is explicit that "click_element" is not a decision and
    "click_element(name='Send')" is. It used to be enforced per name, so an
    approval of one call authorised every later call to that tool.

    That mattered because `desktop/app.py` does not resume the approved call:
    it re-runs the whole task with the name pre-approved, and the model is
    re-prompted from scratch. Nothing pinned the second run's arguments to the
    ones a human read. Approve `write_file(path='notes.md')` and the retry was
    free to write anywhere.
    """
    approved = approval_token("write_file", {"path": "notes.md", "content": "safe"})
    provider = ScriptedProvider(
        [wants("write_file", path="C:/startup/run.bat", content="evil"), answer("done")]
    )
    runner = FakeRunner({"write_file": (False, "needs confirmation", "confirm")})
    outcome = await AgentLoop(provider, runner, approved_tools=[approved]).run(start())

    assert runner.calls[0][2] is False, "a grant for one path authorised another"
    assert outcome.pending_confirmations != []


async def test_an_approval_holds_for_the_exact_call_it_was_given_for() -> None:
    """The other half: a re-run of the same task must not re-prompt forever."""
    args = {"path": "notes.md", "content": "safe"}
    provider = ScriptedProvider([wants("write_file", **args), answer("done")])
    runner = FakeRunner()
    await AgentLoop(
        provider, runner, approved_tools=[approval_token("write_file", args)]
    ).run(start())

    assert runner.calls[0][2] is True


async def test_argument_order_does_not_change_the_token() -> None:
    """A model re-emitting the same call with keys in another order is the same
    call, and must not be treated as a new one to approve."""
    a = approval_token("write_file", {"path": "x", "content": "y"})
    b = approval_token("write_file", {"content": "y", "path": "x"})
    assert a == b


async def test_a_bare_tool_name_no_longer_grants_anything() -> None:
    """Fails closed. A caller still passing pre-finding-2 names grants nothing
    rather than granting everything."""
    provider = ScriptedProvider([wants("write_file", path="x", content="y"), answer("no")])
    runner = FakeRunner({"write_file": (False, "needs confirmation", "confirm")})
    await AgentLoop(provider, runner, approved_tools=["write_file"]).run(start())

    assert runner.calls[0][2] is False


async def test_pending_confirmations_carry_the_binding_not_just_a_name() -> None:
    """The approver has to be able to hand back something argument-specific."""
    provider = ScriptedProvider(
        [wants("write_file", path="notes.md", content="hi"), answer("need approval")]
    )
    runner = FakeRunner({"write_file": (False, "needs confirmation", "confirm")})
    outcome = await AgentLoop(provider, runner).run(start())

    assert outcome.pending_confirmations == [
        approval_token("write_file", {"path": "notes.md", "content": "hi"})
    ]
    assert tool_name(outcome.pending_confirmations[0]) == "write_file"


async def test_a_refusal_does_not_cancel_the_rest_of_the_request() -> None:
    """Reported from a real session.

    "open antigravity ide and code display first n natural numbers in python"
    asked for two things. The launch needed approval and was refused; the code
    needed no tool at all. AURA explained the refusal and dropped the code half
    without mentioning it.

    The observation used to end "explain to the user what you would like to do
    and why", and a 4B does exactly that -- it explains, and stops. One refused
    action is one refused action, not the end of the turn, and a refusal that
    quietly cancels the rest of a request is worse than the refusal: the user
    cannot see what was dropped.
    """
    provider = ScriptedProvider(
        [wants("open_application", app="Antigravity"), answer("opened, and here is the code")]
    )
    runner = FakeRunner({"open_application": (False, "needs confirmation", "confirm")})
    await AgentLoop(provider, runner).run(start())

    observations = [
        m.content
        for call in provider.calls
        for m in call
        if m.role is Role.TOOL
    ]
    assert observations, "the refusal never reached the model"
    text = observations[-1].lower()
    assert "carry on" in text, "the model is not told to continue with the rest"
    assert "explain to the user what you would like to do" not in text


async def test_a_denial_also_leaves_the_rest_of_the_turn_alive() -> None:
    """Same rule for DENY. `run_python` is LEVEL 4 and denied outright after
    audit 01 finding 3, so this is now the common case rather than a rare one."""
    provider = ScriptedProvider([wants("run_python", code="print(1)"), answer("done")])
    runner = FakeRunner({"run_python": (False, "denied by policy", "deny")})
    await AgentLoop(provider, runner).run(start())

    observations = [
        m.content for call in provider.calls for m in call if m.role is Role.TOOL
    ]
    assert "carry on with every" in observations[-1].lower()


async def test_a_refusal_tells_the_model_not_to_retry() -> None:
    """Otherwise the model burns the whole budget on the same rejected call."""
    provider = ScriptedProvider([wants("write_file", path="x"), answer("ok")])
    runner = FakeRunner({"write_file": (False, "needs confirmation", "confirm")})
    messages = start()
    await AgentLoop(provider, runner).run(messages)

    observation = next(m for m in messages if m.role is Role.TOOL).content
    assert "REFUSED" in observation
    assert "Do not retry" in observation


# -- failure is fed back, not raised ----------------------------------------


async def test_a_tool_error_becomes_an_observation() -> None:
    provider = ScriptedProvider(
        [wants("read_file", path="missing"), answer("That file does not exist.")]
    )
    runner = FakeRunner({"read_file": (False, "no such file", "allow")})
    outcome = await AgentLoop(provider, runner).run(start())

    assert outcome.content == "That file does not exist."
    assert outcome.stop_reason == "complete"


async def test_a_denied_path_is_reported_as_denied() -> None:
    provider = ScriptedProvider([wants("read_file", path="/etc/shadow"), answer("Cannot.")])
    runner = FakeRunner({"read_file": (False, "outside the workspace", "deny")})
    messages = start()
    await AgentLoop(provider, runner).run(messages)

    assert "DENIED" in next(m for m in messages if m.role is Role.TOOL).content


# -- budgets ----------------------------------------------------------------


async def test_the_step_budget_stops_a_runaway_loop() -> None:
    """A model that only ever asks for tools must still terminate."""
    provider = ScriptedProvider([wants("read_file", path=f"{i}") for i in range(50)])
    runner = FakeRunner()
    outcome = await AgentLoop(provider, runner, max_steps=3).run(start())

    assert outcome.stopped_early
    assert outcome.stop_reason == "step_budget"
    assert outcome.tool_calls_made == 3


async def test_the_model_is_told_it_ran_out_of_steps() -> None:
    """So the final answer can say what is unfinished rather than inventing one."""
    provider = ScriptedProvider([wants("read_file", path="x") for _ in range(10)])
    messages = start()
    await AgentLoop(provider, FakeRunner(), max_steps=2).run(messages)

    assert any("all 2 available tool steps" in m.content for m in messages)


async def test_the_final_call_after_the_budget_offers_no_tools() -> None:
    """Otherwise the model asks for another tool and the loop never closes."""
    provider = ScriptedProvider([wants("read_file", path="x") for _ in range(10)])
    await AgentLoop(provider, FakeRunner(), max_steps=2).run(start())

    assert provider.tools_seen[-1] is None


async def test_the_time_budget_stops_the_loop() -> None:
    class SlowProvider(ScriptedProvider):
        async def chat(
            self,
            messages: Sequence[Message],
            params: GenerationParams | None = None,
            tools: Sequence[dict[str, Any]] | None = None,
        ) -> ChatResult:
            await asyncio.sleep(0.15)
            return wants("read_file", path="x")

    outcome = await AgentLoop(SlowProvider([]), FakeRunner(), max_steps=100, max_seconds=0.3).run(
        start()
    )

    assert outcome.stopped_early
    assert outcome.stop_reason == "time_budget"


# -- the emergency stop -----------------------------------------------------


async def test_the_emergency_stop_halts_a_running_loop() -> None:
    """Checked every iteration: a loop that runs for minutes must be stoppable."""
    stopped = {"value": False}
    calls = {"count": 0}

    class Counting(ScriptedProvider):
        async def chat(
            self,
            messages: Sequence[Message],
            params: GenerationParams | None = None,
            tools: Sequence[dict[str, Any]] | None = None,
        ) -> ChatResult:
            calls["count"] += 1
            if calls["count"] >= 2:
                stopped["value"] = True
            return wants("read_file", path="x")

    with pytest.raises(EmergencyStopError):
        await AgentLoop(Counting([]), FakeRunner(), max_steps=10).run(
            start(), should_stop=lambda: stopped["value"]
        )
    assert calls["count"] < 10, "the loop kept going after the stop was engaged"


# -- accounting -------------------------------------------------------------


async def test_usage_accumulates_across_steps() -> None:
    def with_usage(result: ChatResult, p: int, c: int) -> ChatResult:
        return ChatResult(
            content=result.content,
            model=result.model,
            provider=result.provider,
            usage=Usage(prompt_tokens=p, completion_tokens=c),
            tool_calls=result.tool_calls,
        )

    provider = ScriptedProvider(
        [with_usage(wants("read_file", path="x"), 10, 5), with_usage(answer("done"), 20, 7)]
    )
    outcome = await AgentLoop(provider, FakeRunner()).run(start())

    assert outcome.usage.prompt_tokens == 30
    assert outcome.usage.completion_tokens == 12


async def test_the_trace_records_every_step() -> None:
    """Spec section 25: 'why did AURA do this?' needs an answer that is not a guess."""
    provider = ScriptedProvider([wants("read_file", path="x.txt"), answer("done")])
    outcome = await AgentLoop(provider, FakeRunner()).run(start())

    tool_step = outcome.steps[0]
    assert tool_step.kind == "tool"
    assert tool_step.tool_calls[0]["name"] == "read_file"
    assert tool_step.tool_calls[0]["arguments"] == {"path": "x.txt"}
    assert tool_step.tool_calls[0]["ok"] is True
