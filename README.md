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

`/v1/transcribe`: the server sends `{"type":"ready"}`; the client sends binary frames of raw
PCM16 little-endian, mono, 16 kHz (any chunk size; the demo sends 100 ms = 3,200 bytes) and
finally the text frame `{"type":"stop"}`. The server streams `{"type":"delta","text":"…"}`,
then `{"type":"final","text":"…"}` and closes with 1000. If `API_KEY` is set, pass
`?token=<key>` in the URL.

Errors are `{"type":"error","code":"…","message":"…"}` followed by a close:

| Code | When | Close code |
| --- | --- | --- |
| `busy` | `MAX_SESSIONS` already active | 1013 |
| `time_limit` | Session exceeded `MAX_SESSION_SECONDS`; a `final` is still sent first | 1000 |
| `bad_request` | Unknown text frame, empty audio | 1003 |
| `upstream_error` | Any vLLM error or dropped upstream connection | 1011 |

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
