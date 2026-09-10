# Known issues

Open problems, with the evidence gathered so far. Each entry records what was
tried and why it did not work, so picking it up later does not mean starting
the diagnosis from scratch.

---

## 1. Barge-in does not interrupt on this machine

**Status:** open, but **worked around** in 10B.7 · Deferred 2026-09-05 ·
Severity: **low** (there is now a voice route that works; the loudness one
still does not)

### What changed in 10B.7

**Saying the wake word over her now stops her.** `voice.wake_interrupt`, on by
default. It matches a *pattern* rather than a level, so AURA's own voice coming
back through the microphone cannot trigger it however bad the coupling is —
which sidesteps this entire problem rather than solving it.

The loudness-based path below is **still broken and still unmeasured**. What
follows is unchanged, and the measurement in "Next steps" is still the thing
that would settle it.

### Symptom

Saying "stop" while AURA is speaking does not stop it. The reply runs to
completion.

### What has been tried

Three attempts, all wrong in an instructive way:

| Attempt | Threshold | Result |
|---|---|---|
| Fixed `barge_in_min_rms: 0.02` | below the echo floor | AURA interrupted *itself*, transcribed its own reply, answered it, looped forever |
| Fixed `barge_in_min_rms: 0.10` | above the user's voice | 18 s of speech, **zero** barge-in events |
| Adaptive: `median(echo) × 2.0` | measured per reply | still not triggering |

### Measurements

- AURA's own voice reaching the mic array: **RMS 0.024–0.040** (14 samples,
  mean 0.030), laptop speakers.
- The user's speech level at the mic: **not yet measured.** This is the missing
  number and the reason the problem is still open.

### The most likely explanation

The user's voice, as heard by the Realtek mic array while the speakers are
playing, is not meaningfully louder than the echo. If true, **no threshold can
separate them** and the current approach cannot be fixed by tuning.

### Next steps, in order

1. Run `aura calibrate` and read the two measured numbers. It reports the echo
   floor and the user's speech level, and explicitly says when they overlap.
2. Check `voice.barge_in_not_triggered` in `var/logs/aura.log`. It records the
   loudest sound ignored during a reply and the threshold it was measured
   against — the two numbers that settle this.
3. If speech and echo do overlap, tuning is the wrong tool. Options:
   - **Headphones.** Removes the coupling entirely; barge-in should then work
     with no changes.
   - **Acoustic echo cancellation.** `speexdsp` or WebRTC AEC. Real work, and
     the only fix that makes speakers viable.
   - **Wake-word interrupt.** Run the wake detector during playback so
     "hey jarvis" stops the reply. It rejects AURA's own voice by *pattern*
     rather than by loudness, which sidesteps the whole problem. Cheapest
     robust option, and probably the right one.
   - **A hotkey.** Unglamorous, works every time.

### Do not

Raise or lower `barge_in_min_rms` again without the measurement from step 1.
That has now failed twice in opposite directions.

Nor should the wake-word interrupt be taken as a reason to close this. It gives
the user *a* way to stop her, which is what mattered most; it does not make
talking over her work, and "stop" spoken mid-reply still does nothing unless it
follows the wake word.

### Relevant code

- `src/aura/voice/session.py` — `_check_barge_in`, `_current_barge_threshold`
- `src/aura/config.py` — `VoiceConfig.barge_in_*`
- `src/aura/__main__.py` — `calibrate`

## Vision: OCR is slow on text-dense screens

**Status:** understood, partially mitigated (Phase 5).

Reading a full 1920x1080 screen takes ~2s on a sparse desktop and 10-30s on a
page dense with text. Root cause is measured, not guessed: RapidOCR's
recognition stage is superlinear in text-crop width (30ms per batch at 320px,
364ms at 1280px) and crops come from the full-resolution image.

Mitigated by capping detection resolution and bounding onnxruntime threads
(the `-1` default oversubscribes 24 logical cores and measured 5x slower).
Out of the box this took 42s and exceeded the tool timeout entirely.

**Next fix, when it matters:** capture and read only the foreground window's
rectangle instead of the whole desktop. Faster, and closer to what the user
means by "read my screen".

