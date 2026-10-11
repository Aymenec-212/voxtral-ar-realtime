# voxtral-ar-realtime

One container image that serves Mistral's
[Voxtral-Mini-4B-Realtime-Arabic](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-Arabic)
for live, streaming Arabic speech-to-text (Modern Standard Arabic and dialects including
Moroccan Darija), behind a small FastAPI gateway with a browser demo.

Inside the container, vLLM runs on a private localhost port and the gateway is the only public
surface. Clients speak a tiny WebSocket protocol and never see vLLM. Weights are not in the
image; they download on first boot into a volume mounted at `/models`.

## Try it

**Modal (reference deployment, scales to zero):** see [RUNBOOK.md](RUNBOOK.md).

**Local development without a GPU:** the whole system runs against a fake vLLM.

```bash
uv venv .venv --python 3.12 && . .venv/bin/activate && uv pip install -e ".[dev]"
python -m tests.fake_vllm --port 9000 &
UPSTREAM_URL=ws://localhost:9000/v1/realtime python -m voxtral_rt
# open http://localhost:8000 and press Start, or:
python clients/stream_wav.py --url ws://localhost:8000/v1/transcribe --wav sample.wav
```

**Your own GPU machine with Docker** (image publishing is not wired up yet; build locally):

```bash
docker build -f docker/Dockerfile -t voxtral-ar-realtime .
docker run --gpus all -p 8000:8000 -v voxtral-models:/models voxtral-ar-realtime
```

## API

| Route | Purpose |
| --- | --- |
| `GET /` | Demo page |
| `GET /health` | `200 {"status":"ok"}` once the engine is up, `503 {"status":"starting"}` before |
| `WS /v1/transcribe` | One utterance per connection |
| `GET /v1/metrics/stream` | Server-Sent Events, one server-wide metrics snapshot per second |

`/v1/transcribe`: the server sends `{"type":"ready"}`; the client sends binary frames of raw
PCM16 little-endian, mono, 16 kHz (any chunk size; the demo sends 100 ms = 3,200 bytes) and
finally the text frame `{"type":"stop"}`. The server streams `{"type":"delta","text":"…"}`,
about once a second a `{"type":"metrics", …}` snapshot of the session (fields below), then
`{"type":"final","text":"…","metrics":{…}}` and closes with 1000. If `API_KEY` is set, pass
`?token=<key>` in the URL (also for `/v1/metrics/stream`).

Errors are `{"type":"error","code":"…","message":"…"}` followed by a close:

| Code | When | Close code |
| --- | --- | --- |
| `busy` | `MAX_SESSIONS` already active | 1013 |
| `time_limit` | Session exceeded `MAX_SESSION_SECONDS`; a `final` is still sent first | 1000 |
| `bad_request` | Unknown text frame, empty audio | 1003 |
| `upstream_error` | Any vLLM error or dropped upstream connection | 1011 |

## Metrics

Metrics are computed in the gateway from what the model actually does, kept in memory, and
reset when the container restarts. There is no persistence and no external metrics stack.

**Per session** (in each `metrics` event and in `final.metrics`), measured in the gateway's
clock with t0 = the first audio frame:

| Field | Meaning |
| --- | --- |
| `audio_received_s` | Audio received so far: bytes / 32000 |
| `tokens` | Upstream delta events so far, empty ones included (one per generated token) |
| `audio_transcribed_s` | `min(tokens × 0.08 s, audio_received_s)`: the model emits one token per 80 ms of audio |
| `lag_ms` | `(audio_received_s − audio_transcribed_s) × 1000`: how far the model is behind the audio |
| `max_lag_ms` | The largest `lag_ms` seen in the session |
| `first_text_ms` | First delta with non-empty text, relative to t0 |
| `tokens_per_s` | Tokens over the last ~5 s of wall time; 12.5 means keeping up with real-time audio |
| `finalize_ms` | `final` sent minus the moment the engine was told the audio was over |
| `session_s`, `prompt_tokens`, `completion_tokens` | Session length and the engine's usage report |

`lag_ms` and `first_text_ms` include the model's built-in transcription delay of about 480 ms
(6 frames of 80 ms). A lag of 480 ms is the model working as designed, not gateway overhead.
One JSON line per finished session with all of these plus the outcome goes to the log.

**Server-wide** (`GET /v1/metrics/stream`, one `data:` line per second): `active_sessions`,
`max_sessions`, `sessions_completed`, `sessions_abandoned`, `errors` by client-facing code,
p50/p95 of `first_text_ms` and `finalize_ms` and p95 of `max_lag_ms` over the last 100 finished
sessions, and `engine`: `boot_s` (engine start to ready, measured by the launcher, `null` with
`UPSTREAM_URL`), `kv_cache_usage_pct`, `requests_running`, `requests_waiting` and
`generation_tokens_per_s`, read from vLLM's own Prometheus endpoint on the private engine port.
A metric the engine does not expose is `null`; an unreachable engine sets `reachable` to false.

```bash
curl -N http://localhost:8000/v1/metrics/stream
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_ID` | `mistralai/Voxtral-Mini-4B-Realtime-Arabic` | Hugging Face repo, also the served model name |
| `MODEL_REVISION` | `a8b0156f…` | Pinned commit of the repo |
| `MODELS_DIR` | `/models` | Where weights live; mount a volume here |
| `PORT` | `8000` | Public gateway port |
| `ENGINE_PORT` | `8001` | vLLM's port, bound to `127.0.0.1` only |
| `UPSTREAM_URL` | unset | If set, skip weights and engine and use this realtime URL (local dev) |
| `API_KEY` | unset | If set, clients must send `?token=` |
| `MAX_SESSIONS` | `4` | Concurrent transcription sessions |
| `MAX_SESSION_SECONDS` | `300` | Hard cap per session |
| `ENGINE_READY_TIMEOUT_S` | `900` | How long to wait for vLLM on boot |
| `VLLM_EXTRA_ARGS` | empty | Extra `vllm serve` flags, appended verbatim |

## Layout

```
voxtral_rt/      config.py, upstream.py (vLLM protocol), gateway.py, engine.py, weights.py, web/
tests/           fake_vllm.py (CPU stand-in for vLLM) and the test suite
clients/         stream_wav.py
docker/          Dockerfile on top of vllm/vllm-openai:v0.31.0
deploy/          modal_app.py
```

Development: `ruff check . && ruff format --check . && pytest`.
