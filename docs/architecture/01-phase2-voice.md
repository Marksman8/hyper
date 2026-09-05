# Phase 2 — Voice

Status: **in progress** · Date: 2026-09-05 · Numbers below are measured on this machine

---

## 1. What Phase 2 is

Wake word → speech in → AURA → speech out, with barge-in, running entirely
locally and never touching AURA Core's process.

## 2. Topology: voice is a client, not a subsystem

```
  +-------------------------------------------------------------+
  | AURA VOICE       python -m aura voice                        |
  |                                                              |
  |   mic -> ring buffer -> wake word -> VAD -> STT              |
  |                                              |               |
  |   speaker <- playback <- TTS <- sentence chunker             |
  +----------------------------|---------------------------------+
                               |  HTTP + SSE, bearer token
                               v
  +-------------------------------------------------------------+
  | AURA CORE :8760   (unchanged, still torch-free, still 69 MB) |
  +----------------------------|---------------------------------+
                               v
  +-------------------------------------------------------------+
  | llama-server :8081                                           |
  +-------------------------------------------------------------+
```

**Why voice is a separate process, like the CLI.**

1. *AURA Core stays light.* The speech stack pulls onnxruntime, CTranslate2 and
   ~250 MB of models. Putting it in Core would end the sub-second restarts that
   make development bearable, for no benefit.
2. *Real-time loops do not belong beside an HTTP server.* Audio callbacks have
   hard deadlines; a slow request handler must never underrun the mic.
3. *It crashes alone.* A PortAudio failure or a bad ONNX session kills voice,
   not the assistant's memory and API.
4. *It proves the API.* Voice reaches AURA through the same authenticated
   `/v1/chat` any third-party client would use.

## 3. No torch

The roadmap originally said Phase 2 is where torch enters. That was wrong, and
on a machine with 0.8 GB free RAM the difference is decisive:

| Component | Runtime | Resident |
|---|---|---|
| openWakeWord | onnxruntime | ~60 MB |
| Silero VAD | onnxruntime (bundled with faster-whisper) | ~5 MB |
| faster-whisper | CTranslate2 | ~150 MB (`base.en` int8) |
| Piper | onnxruntime | ~65 MB per voice |
| shared runtime | onnxruntime + ctranslate2 | ~150 MB |

Predicted ~430 MB with everything loaded. **Measured: 430 MB** for the voice
process with wake word, VAD, STT and TTS all resident. Versus ~2.5 GB of disk
and several hundred MB more for an equivalent torch stack. Torch now enters
only at Phase 8, where LoRA training actually requires it.

Full stack on this machine: llama-server 4.0 GB RSS / 3.07 GB VRAM, AURA Core
69 MB, voice 430 MB.

## 4. The seam, again

Same rule as §32, applied to speech. Three interfaces, and no module outside a
provider package may import a speech library:

```
aura.voice.wake.base.WakeWordDetector   -> OpenWakeWordDetector
aura.voice.stt.base.STTProvider         -> FasterWhisperSTT   (later: WhisperCppServer)
aura.voice.tts.base.TTSProvider         -> PiperTTS           (later: KokoroTTS)
```

Each has a null/stub implementation so the state machine can be tested with no
audio hardware, no models, and no GPU — the same discipline that lets the whole
v0.1 suite run against `EchoProvider`.

## 5. State machine

```
                  wake word
     SLEEP  ---------------------->  LISTENING <--------+
       ^                                 |              |
       |                                 | VAD endpoint |
       |                                 v              |
       |                             THINKING           |
       |                                 |              |
       |                                 v              |
       |                             SPEAKING           |
       |                                 |              |
       |                                 v              |
       +---- nothing said ---------  FOLLOW_UP ---------+
                                                 speech
```

**FOLLOW_UP** is what makes this a conversation rather than a vending machine.
After an answer, the microphone stays open for a few seconds so the next
question needs no wake word; saying nothing closes it. Barge-in also lands
here, because talking over an answer is unambiguous evidence that the user has
something to say.

Entered with no pre-roll, deliberately: the ring buffer at that moment holds
AURA'''s own reply, and feeding that to Whisper transcribes the assistant
talking to itself.

- **LISTENING** has a hard timeout, so a false wake never leaves the mic open.
- **Barge-in**: VAD running during SPEAKING. Speech above threshold for ~200 ms
  stops playback immediately and re-enters LISTENING.
- **PRIVATE / mic kill switch** (§23): stops the capture stream. It does not
  gate a downstream consumer — if the switch is off, no audio is being read.

## 6. Latency budget

Target: **under 1.5 s** from end of speech to first audio out.

| Stage | Budget | Measured |
|---|---|---|
| VAD endpoint detection | 400 ms | 600 ms (`endpoint_silence_s`, tunable) |
| STT (`base.en` int8, 16 threads) | 300 ms | 701 ms for 3.9 s audio (RTF 0.18) |
| LLM prefill + first token | 200 ms | ~200 ms (v0.1 measurement) |
| TTS first chunk | 250 ms | **73 ms** (RTF 0.04) |
| **Total** | **~1.15 s** | pending an end-to-end spoken measurement |

STT is the term that came in over budget: 0.18 real-time means a 3-second
command costs ~540 ms. Options if it matters: shrink `endpoint_silence_s`,
or move Whisper to the GPU (4.9 GB VRAM is free; the cost is cuDNN setup on
Windows). TTS came in 3x under budget, which buys some of it back.

The load-bearing trick is **sentence chunking**: AURA Core already streams
tokens over SSE, so voice synthesises the first sentence while the model is
still generating the second. Waiting for the full reply would put a 4-second
answer at 4+ seconds to first audio.

## 7. Model choices

| Role | Choice | Why |
|---|---|---|
| Wake word | openWakeWord `hey_jarvis` | ONNX, CPU, ~1% of one core |
| VAD | Silero (ONNX) | tiny, far better than energy thresholding |
| STT | faster-whisper `base.en` int8 CPU | 150 MB, no cuDNN setup, fast enough |
| TTS | Piper `en_US-amy-medium` | female, natural, ~0.1 RTF on CPU |

**Known limitation: the wake word is not "Aura".** openWakeWord ships
pretrained models for a fixed set of phrases (`hey_jarvis`, `alexa`,
`hey_mycroft`, ...) and "aura" is not among them. Training a custom one is an
offline job (synthetic data + a Colab notebook), not something to block this
phase on. `hey_jarvis` is the default; `voice.wake.model` swaps it, and a
custom `hey_aura.onnx` drops into `var/models/wakeword/` when trained.

**STT runs on CPU, not the GPU.** CTranslate2's CUDA build needs cuDNN 9 DLLs
on Windows, which is a real setup burden for ~200 ms. With 4.9 GB VRAM free the
GPU is available if it becomes worth it — the `STTProvider` seam means that is
a config change, not a rewrite.

## 8. Out of scope for Phase 2

Speaker identification · multi-language · emotion · voice cloning · always-on
transcription. Wake-word-gated only: the mic stream is open, but nothing is
transcribed or sent anywhere until the wake word fires.
