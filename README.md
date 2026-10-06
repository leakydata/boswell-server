# Boswell Server

The home processing server for [Boswell Phone](https://github.com/leakydata/boswell-phone):
your own computer (with an NVIDIA GPU) does the heavy lifting — speech detection,
transcription, speaker separation, voiceprints and sound tagging — with models far larger
than a phone can run, and sends finished transcripts back to the phone.

**Status: the server works; the phone side is being built.** On an RTX 4090 a 30-second
recording takes under a second end to end:


| Step | Model | Time |
|---|---|---|
| Who spoke when | pyannote speaker-diarization **community-1** (GPU) | ~0.2 s |
| Words | NVIDIA **Parakeet TDT 0.6B v3**, full precision (GPU, onnx-asr) | ~0.17 s |
| Voiceprints | the phone's own models: WeSpeaker ResNet34-LM or **ReDimNet2-B6** (GPU) | < 0.2 s |
| Sounds | CED-Mini (GPU) | ~0.18 s |

## Why these models

Each was chosen by measuring the candidates on the owner's own Omi recordings
(scripts in `tools/`), not by leaderboard:

| Job | Kept | Measured against | Result |
|---|---|---|---|
| Words | Parakeet TDT 0.6B v3 | Parakeet v2, Canary 1B v2, Whisper large-v3, the phone's Nemotron | 12.5% words wrong on the owner's corrected lines, vs 15.0–19.4% (`asr_lines.py`) |
| Who spoke when | community-1 | pyannote 3.1 | 31.1% vs 36.1% diarization error, right speaker count 53/80 vs 38/80, on conversations stitched from named speakers (`diar_bench.py`) |
| Voiceprints | ReDimNet2-B6 | WeSpeaker ResNet34-LM, w2v-BERT 2.0 | right person for 92.3% of hand-named voices vs 82.1% and 76.9% (`voice_bench.py`) |
| Speech or not | pyannote segmentation | Silero VAD v5 | 385/400 recordings right vs 358/400, 37 vs 484 ms (`vad_bench.py`) |

Published scores didn't always carry over: w2v-BERT 2.0 has the best published
speaker-verification result (VoxCeleb1-O 0.14%), and was the weakest of the three
on the owner's recordings.

## Running it

```bash
uv sync                          # Linux with an NVIDIA GPU (CUDA 12)
uv run boswell-server fetch-models
uv run boswell-server doctor     # what's in place
uv run boswell-server            # the terminal screen: press p to pair a phone
uv run boswell-server serve      # or headless; then `boswell-server pair` for a code
```

### In the background

```bash
uv run boswell-server install-service     # a systemd user service: starts now, and at every boot
uv run boswell-server uninstall-service   # stops it and removes it
```

`install-service` writes `~/.config/systemd/user/boswell-server.service` (this checkout's
`.venv/bin/boswell-server serve`), enables and starts it, and turns on lingering
(`loginctl enable-linger`) so it runs from boot without anyone logging in. `BOSWELL_PORT`,
`BOSWELL_SERVER_DATA` and `BOSWELL_SERVER_MODELS`, if set, go into the unit. If the port is
already taken (say, by a server in a terminal), it's enabled but not started: stop the other one,
then `systemctl --user start boswell-server`. Its log: `journalctl --user -u boswell-server -f`.
`doctor` says whether it's installed and running.

### The screen

`boswell-server` with nothing running loads the models and runs the server itself; quitting
stops it. With the service (or another server) already on the port, it **attaches** instead: it
loads nothing and shows the running server's models, GPU, recordings and log (from
`GET /v1/local/status`, answered only to this computer, never through a proxy). Pairing (`p`)
and forgetting a phone (`f`) work the same way; `c` clears the recordings list and the log on
this screen only; `q` closes the screen and leaves the server running.

pyannote's diarization model is gated on Hugging Face: accept its terms and run
`huggingface-cli login` once. [Tailscale](https://tailscale.com) on this computer and the
phone lets the phone reach it from anywhere.

### HTTPS

```bash
sudo tailscale serve --bg --https=443 http://127.0.0.1:8765
```

With HTTPS certificates turned on for the tailnet (admin console → DNS), Tailscale passes the
server on at `https://<this computer>.<tailnet>.ts.net` with a real certificate. The server
notices, and its pairing code gives the phone that address; a phone paired over plain HTTP
moves to it by itself.

### AI on this computer

The phone's assistant (questions, titles and summaries, briefs, the listening watcher) can run
here instead of on OpenRouter: Device → Assistant → Where the AI runs → Home server. The server
passes those requests to [Ollama](https://ollama.com) (`POST /v1/chat/completions`, the same
format OpenRouter speaks), free, and nothing leaves the house. Web searches still go to
OpenRouter. Measured beside the speech models on a 24 GB RTX 4090:

| Model | On the GPU | Per tool call | Right tool calls | Titles as JSON |
|---|---|---|---|---|
| **gemma4:e4b**, thinking off (default) | all, about 5 GB | 0.2–2 s | 15/15 | 6/6 |
| gemma4:e4b, thinking on | all | 1–4 s | 14/15 | 0/6 (thinking used up the answer) |
| gpt-oss:20b | 24% | 4–12 s | 10/10 | 4/4 |
| qwen3.6:27b | 28% | 24–98 s | 3/3 | |

Settings: `BOSWELL_LLM_MODEL` (default `gemma4:e4b`), `BOSWELL_LLM_THINK` (`none`),
`BOSWELL_LLM_KEEP_ALIVE` (how long the model stays loaded after an answer, `10m`),
`BOSWELL_OLLAMA` (`http://127.0.0.1:11434`).

## The API

| | |
|---|---|
| `GET /v1/health` | is it up, which models |
| `POST /v1/pair` `{code, device}` | a pairing code → this phone's key (stored here only as a hash) |
| `POST /v1/analyze?voice_model=…` | one recording (Ogg Opus or WAV) → words, speakers with turns and voiceprints, sounds |
| `POST /v1/chat/completions` | the assistant's AI, answered by Ollama here (OpenAI format, with tools) |
| `GET /v1/llm` | is the local AI available, which model |
| `POST /v1/backup` | the phone's daily backup (a zip, streamed to disk): the newest 7 per phone are kept in `~/.local/share/boswell-server/backups/` |
| `GET /v1/backups`, `GET /v1/backups/{name}` | this phone's backups, and one of them to restore from; a phone sees only its own |
| `GET /v1/local/status` | this computer only: what the screen shows (models, GPU, recent recordings, log) |

The phone assembles its transcript from these exactly as it does from its own models, and
matches the voiceprints against its own people.

## The plan

- **Private by design.** The phone reaches the server over [Tailscale](https://tailscale.com),
  a private network between your own devices; nothing is exposed to the internet. Each phone
  pairs once by scanning a QR code and authenticates every request with its own key.
- **The phone stays in charge.** People, names and voices live on the phone; the server returns
  transcripts in the phone's own format, with voiceprints the phone matches. When home is out
  of reach, the phone does the work itself.
- **A small terminal screen** (built with Textual): GPU, queue, paired phones, models, logs.
- **Models chosen by measurement** on real recordings, using CUDA where possible.

## License

[Apache License 2.0](LICENSE). **Credit is required:** keep the copyright notice and the
[NOTICE](NOTICE) file in anything you distribute.

Copyright 2026 Nathan Jones ([@leakydata](https://github.com/leakydata)).
