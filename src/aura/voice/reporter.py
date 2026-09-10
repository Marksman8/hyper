"""Reports speech to Core's event bus, off the speech path (10B.1).

The voice loop lives in its own process because audio devices and the GIL put
it there, so the only route onto Core's in-process bus is an HTTP POST. The
whole design of this module follows from one constraint:

**Reporting speech must never slow speech down.**

The pipeline has a 1.5 s budget from wake word to first audio, and it is met by
speaking sentence one while the model is still writing sentence two. A
synchronous POST per sentence would put a network round trip inside that loop,
where it would show up as a gap between spoken sentences -- a real regression
in the thing the user actually hears, traded for a subtitle.

So: a bounded queue and a daemon thread. The queue is small and **overflow
drops the oldest**, which is not a compromise but the delivery semantics the
catalog already declares. `SPEECH_*` are DROPPABLE. A subtitle that skipped a
sentence during a burst is a cosmetic loss; a reply that stuttered is not.

Nothing here raises. A dead Core costs subtitles and nothing else.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from aura.logging import get_logger

log = get_logger(__name__)

# Deep enough to absorb a burst of short sentences while the network is slow,
# shallow enough that a backed-up queue drops promptly instead of delivering
# subtitles for a reply that finished a minute ago. A stale subtitle is worse
# than a missing one -- it is actively wrong about what AURA is saying now.
QUEUE_DEPTH = 32


class SpeechReporter:
    """Fire-and-forget publisher for the speech lifecycle."""

    def __init__(self, client: Any, *, depth: int = QUEUE_DEPTH) -> None:
        self._client = client
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=depth)
        self._thread: threading.Thread | None = None
        self._dropped = 0
        self._sent = 0

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def sent(self) -> int:
        return self._sent

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="speech-reporter", daemon=True
        )
        self._thread.start()

    # -- the three events ----------------------------------------------------

    def started(self, length: int) -> None:
        self._put({"type": "speech_started", "length": max(0, length)})

    def chunk(self, text: str, envelope: str = "") -> None:
        """One sentence, with the amplitude envelope that drives the mouth.

        The envelope is optional: a sentence spoken outside the normal turn
        flow has no PCM in hand at this point, and a still mouth is a better
        answer than a fabricated one.
        """
        if text.strip():
            self._put(
                {
                    "type": "speech_chunk",
                    "text": text[:1000],
                    "envelope": envelope[:400],
                }
            )

    def ended(self, *, completed: bool) -> None:
        self._put({"type": "speech_ended", "completed": completed})

    # -- plumbing ------------------------------------------------------------

    def _put(self, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            pass

        # Full. Discard the OLDEST rather than this one: what a subtitle needs
        # is the sentence being spoken now, and the backlog is by definition
        # already behind the audio.
        try:
            self._queue.get_nowait()
            self._dropped += 1
            self._queue.put_nowait(payload)
        except (queue.Empty, queue.Full):
            # Raced with the worker. Losing this event is the documented
            # DROPPABLE outcome, and it is not worth a lock on the speech path.
            self._dropped += 1

    def _run(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:
                return
            try:
                if self._client.publish_event(payload):
                    self._sent += 1
            except Exception as exc:
                # The client already swallows HTTP errors, so reaching here
                # means something unforeseen. Reporting is a convenience; it
                # must never take down the process that is doing the talking.
                log.debug("voice.reporter_failed", error=type(exc).__name__)

    def close(self) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Drain enough room for the sentinel; nothing queued behind it
            # matters once we are shutting down.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        self._thread.join(timeout=2.0)
        self._thread = None
        if self._dropped:
            log.info("voice.reporter_closed", sent=self._sent, dropped=self._dropped)
