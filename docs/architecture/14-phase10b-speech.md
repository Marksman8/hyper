# 10B — Speech you can see

Status: **10B.1–10B.5 complete** · 10B.6–10B.7 outstanding · Date: 2026-09-10

---

## 1. The requirement

Stated in the brief as: *"sometimes I may not be able to hear the audio."*

That one sentence decides the whole milestone. With sound muted the subtitle is
not a nicety layered over speech — **it is the only copy of what AURA said**.
Every rule below follows from treating it that way rather than as decoration.

## 2. The path

```
 voice process                         AURA Core            presence process
 ─────────────                         ─────────            ────────────────
 Piper synthesises
       │
       ├─ envelope(pcm) ──┐
       │                  │
 SpeechReporter ──POST──▶ /v1/events/client ──▶ bus ──SSE──▶ PresenceController
 (queue + thread)                                                  │
                                                        ┌──────────┴─────────┐
                                                        ▼                    ▼
                                                   set_subtitle()      set_mouth()
```

The voice loop is a separate process — audio devices and the GIL put it there —
so the only route onto Core's in-process bus is an HTTP POST.

## 3. Reporting must not slow speech down

The pipeline has a 1.5 s budget from wake word to first audio, met by speaking
sentence one while the model writes sentence two. A synchronous POST per
sentence would put a network round trip *inside* that loop, audible as a gap
between spoken sentences.

So `SpeechReporter` is a bounded queue and a daemon thread. Overflow **drops
the oldest**, which is not a compromise but the delivery semantics the catalog
already declares — `SPEECH_*` are DROPPABLE. A subtitle that skipped a sentence
under load is a cosmetic loss; a reply that stuttered is not.

Measured: 10 sentences queued in under 100 ms with the network blocked.

## 4. Origin, and a rule that needed restating

`ClientEvent` used to say *"origin is forced to USER"*. That was right for the
two events that existed and the wrong statement of the property.

**The property is that a client cannot _choose_ an origin.** The catalog
decides — a version-controlled local file, the same arrangement as the workflow
manifest in ADR-0004.

That distinction matters the moment speech arrives. `USER_SPOKE` is a user
action and is catalogued `Origin.USER`. `SPEECH_CHUNK` is AURA's own utterance
and is catalogued `Origin.CORE`, because the voice loop is not a third-party
client — it is her mouth, running out-of-process for hardware reasons.
Labelling her own words `USER` would be a lie the presence layer then acts on:
`Origin.USER` means a human did something, and idle timers reset on that basis.

What it concedes: an API-token holder can publish a CORE-origin speech event —
put words in a subtitle she never said. Bounded by invariant I4 (an event
cannot cause a tool to run) and dominated by the fact that the same token can
make her genuinely say anything through `/v1/chat`.

## 5. What the subtitle does

| Rule | Why |
|---|---|
| One sentence, replacing the last | A whole reply grows past the screen and stops being readable at exactly the length where reading it matters |
| Held after audio ends, 1.2–5 s by length | The audio ending is not the reader finishing |
| Interrupted → clears in 400 ms | Its last sentence was cut off; a fragment left up reads as the final word |
| PRIVATE/SLEEP **drop** it | Hiding it would let it reappear when privacy ends, carrying the last thing she said before being told to stop |
| Reconnect clears it | `SPEECH_*` are occurrences and absent from the snapshot, so afterwards the overlay does not know if she is still talking |

**The asymmetry with `USER_SPOKE` is deliberate.** That event carries a
*length*; `SPEECH_CHUNK` carries the *text*. The user's words are captured
input from an always-listening microphone. AURA's words are about to be said
out loud in the room.

## 6. The mouth — amplitude, not phonemes

RMS per 20 ms frame, quantised to one digit, normalised against the sentence's
own peak. About 90% of the perceived quality of visemes for about 5% of the
work: no aligner, no second model, no per-language data. A viseme backend can
still arrive behind the same `set_mouth()`.

**The envelope rides with the sentence rather than streaming.** `SPEECH_CHUNK`
is published at the moment a sentence's audio starts, and Piper yields a
sentence's audio in one piece — so the samples are already in hand. No latency,
no extra events, and a bus built for metadata does not become an audio
sidechannel at 50 Hz.

Split across two modules, and the split is the point:

| | |
|---|---|
| `core/envelope.py` | The **format** — frame length, alphabet, `openness()`. Pure Python, zero imports, asserted by a test |
| `voice/envelope.py` | The **measurement** — numpy, RMS, normalisation |

`voice/types.py` says outright that nothing outside `aura.voice` imports it,
and the presence process should not grow a dependency on the voice package to
move a mouth.

**Normalised per sentence.** Absolute normalisation would make quiet speech
read as a rendering fault rather than as quiet speech. The silence floor stops
the same normalisation stretching a noise floor into a mouth flapping at
nothing.

The mouth closes on: the envelope ending, a new sentence, speech ending, an
interruption, privacy, and a reconnect. Any of those left open reads as a
frozen avatar rather than a finished one.

### Measured on real Piper output

`en_US-amy-medium`, one sentence of 60 characters:

| | |
|---|---|
| Chunks returned | **1** — the assumption the design rests on |
| Audio | 3.83 s at 22,050 Hz |
| Envelope | 191 frames, levels 0–9, mean 3.0 |
| Closed frames | 37 of 191 — the gaps between words |

An integration test asserts each of these, and fails loudly if Piper ever
starts fragmenting a sentence.

## 7. The voice itself (10B.4)

**Already delivered in Phase 2**, before the Phase 10 design listed it. The
default is `en_US-amy-medium` — female, US — and
`scripts/download_voice.py --list` offers a curated set with the female voices
first. Switching is a config value plus a download; a missing voice raises with
the command that fetches it.

The design called this "a config change, not engineering", and that turned out
to be true in the strongest sense: there was nothing to build.

## 7a. Micro-expressions (10B.5)

A short fixed phrase in another language, occasionally prepended: 了解 before
starting work, よし on success. The subtitle carries the phrase **and an English
gloss**, so meaning survives when the audio cannot be heard — which is the case
this whole milestone exists for.

**The model never writes these.** It answers in English; this layer decides
whether one fits and prepends it. That separation is the safety property, and
it is enforced by the interface rather than by instruction: `select()` takes a
`Moment` and nothing else, so there is no parameter a model's output could
travel through. A test asserts the signature.

Ask a model to "add personality" instead and foreign phrases leak into every
third response, drift in register, and occasionally mean something other than
intended — none of it reviewable, because the phrase is produced fresh each
time. Here the set is a constant you can read in one screen, and 10B.6 renders
each to a fixed WAV, so every sound AURA can make is auditable in advance.

### Restraint is the feature

| Rule | Why |
|---|---|
| LOW by default, ~1 in 8 | A micro-expression is a garnish; the failure mode of a garnish is applying it to everything |
| MEDIUM is 1 in 4, not a coin flip | A coin flip reads as constant, because listeners remember the hits |
| Never the same phrase twice running | With one candidate left, it stays silent rather than repeating |
| 90 s hard floor between any two | Frequency governs the average; a burst of three short turns is what people actually notice |
| `enabled` and `frequency: off` both silence it | "Off" and "never" are different questions a user might ask |

`Moment` is deliberately *not* `PresenceState` — four members describing what
AURA is reacting to, against thirteen describing what she looks like. Coupling
them would make every new visual state a personality question.

**Measured over a simulated session:** 60 turns across 3.2 hours at the default
settings produced **6 micro-expressions — 10% of turns**, none repeating
consecutively.

## 8. The bug a real screen found

`resize()` moves the bottom and right edges and leaves the top-left alone, so a
bottom-right-anchored overlay that grows a subtitle grows **off the screen**.
The first live run had the avatar and its line both clipped by the screen edge.

The overlay now emits `resized` and the controller re-anchors. That re-anchor
deliberately bypasses `consider()`: the four-trigger rule and the three-second
limit exist to stop AURA *relocating*, and holding the same anchor while her
panel grows is the opposite. Rate-limiting it would leave the panel clipped for
three seconds.

The reason it escaped the tests: **the fake overlay had a fixed size.** It now
grows with the text.

## 9. Outstanding

| | | Note |
|---|---|---|
| 10B.6 | Pre-rendered micro-expressions, glossed in the subtitle | Clips are fixed WAVs so pronunciation is right and every phrase is auditable |
| 10B.7 | Interrupt stops speech | **Blocked on the barge-in measurement** — see `KNOWN-ISSUES.md` |

10B.7 is the one to watch. Saying "stop" does not currently interrupt, three
threshold attempts have failed in opposite directions, and the missing number
is the user's speech level at the microphone. If it overlaps the measured echo
floor (0.024–0.040 RMS), no threshold works and the fix is
wake-word-during-playback, which rejects AURA's own voice by *pattern* rather
than by loudness.
