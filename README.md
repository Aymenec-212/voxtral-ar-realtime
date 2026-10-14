# voxtral-ar-realtime

Live Arabic speech-to-text, including Moroccan Darija, on a single GPU.
It serves Mistral AI's
[Voxtral-Mini-4B-Realtime-Arabic](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-Arabic),
a streaming model that turns 16 kHz audio into text about half a second behind the speaker. vLLM
runs the model on a private port inside the container. A small FastAPI gateway in front of it is
the only public surface: a simple WebSocket API, an API key, a browser microphone demo and live
latency metrics. You never have to learn vLLM's flags or its realtime protocol. The model weights
are Mistral AI's and are released under their own license, stated on the model card; they are
not part of this repository.

## Quickstart on Modal

[Modal](https://modal.com) runs the container on an L4 GPU, scales to zero when idle and keeps
the weights in a Modal Volume. Run everything from the repository root.

1. Install the pinned Modal client, log in and clone the repository.

   ```bash
   pip install modal==1.6.1
   modal setup
   git clone https://github.com/Aymenec-212/voxtral-ar-realtime.git
   cd voxtral-ar-realtime
   ```

2. Create the API key and store it as a Modal secret. Keep `$KEY`: the demo page asks for it.
   This comes before the download because Modal resolves every secret of the app on each run.

   ```bash
   KEY=$(openssl rand -hex 16); modal secret create voxtral-api-key API_KEY="$KEY"
   echo "$KEY"
   ```

3. Download the weights once. This runs on CPU, not GPU, and puts about 8.3 GiB into the
   `voxtral-models` volume. The first run also builds the image, which pulls the ~10 GB vLLM base.

   ```bash
   modal run deploy/modal_app.py::download_weights
   ```

4. Start the server. `modal serve` gives a development URL that stops with Ctrl+C;
   `modal deploy` gives a persistent URL that scales to zero after 5 idle minutes. Modal prints
   the URL. The first request after a cold start waits about 5 minutes while the engine boots;
   `GET /health` returns 503 until it is ready.

   ```bash
   modal serve deploy/modal_app.py      # or: modal deploy deploy/modal_app.py
   ```

5. Open the URL, paste the key, press **Start** and speak. Press **Stop** for the final
   transcript.

6. Stop the app when you are done.

   ```bash
   modal app stop voxtral-ar-realtime
   ```

## Authentication

- The gateway reads the key from `API_KEY`. On Modal it comes from the `voxtral-api-key`
  secret, and the server refuses to start if the secret is missing or its `API_KEY` is empty.
  There is no switch to run the Modal endpoint without a key.
- Clients pass the key as `?token=<key>` on `/v1/transcribe` and `/v1/metrics/stream`. It is a
  query parameter because browsers cannot set headers on a WebSocket. A missing or wrong token
  closes the WebSocket with code 1008 and answers the metrics stream with 401.
- `GET /` (the demo page) and `GET /health` stay public.
- Tokens are compared in constant time, and the launcher rewrites `token=<value>` to
  `token=***` in uvicorn's log lines, so the key never reaches the logs.
- To change the key, overwrite the secret, then restart the app: running containers keep the
  old key until they restart.

  ```bash
  KEY=$(openssl rand -hex 16); modal secret create voxtral-api-key API_KEY="$KEY" --force
  modal app stop voxtral-ar-realtime
  modal deploy deploy/modal_app.py
  ```

- Without `API_KEY`, for example in local development, authentication is disabled and the
  launcher logs a warning at startup.

## Measured on one L4, vLLM 0.31.0

| What | Value |
| --- | --- |
| Cold start | about 4 min 45 s: weights load 6 s, torch.compile 84 s, CUDA graph capture about 30 s |
| Model weights in GPU memory | 8.43 GiB |
| KV cache at `--max-model-len 16384` | 9.45 GiB, 27,082 tokens |
| Generation per session while audio streams in real time | 12.5 tokens/s, one token per 80 ms of audio |
| Cost | about $0.80 per GPU hour; scales to zero after 5 idle minutes |
| Concurrent sessions | `MAX_SESSIONS=4` by default |

The compute ceiling under load has not been measured. `MAX_SESSIONS=4` is a cautious default,
not a tested limit.

## Troubleshooting

- **Modal reports that the secret `voxtral-api-key` was not found.** Create it as in step 2.
- **The server stops with "API_KEY is missing or empty".** The secret exists but holds no key.
  Recreate it with the command from the Authentication section, including `--force`.
- **The page says "Unauthorized: check the API key"**, or the metrics stream returns 401. The
  key you pasted does not match the secret.
- **The container exits with "voxtral_rt exited with code N during startup".** The launcher
  died while booting, and the container stops at once instead of waiting out Modal's startup
  timeout. The real error is a few lines above in the log.
- **The download fails with `RevisionNotFoundError`.** The pinned model commit no longer exists
  on the Hub. Set `MODEL_REVISION` in `ENV` in `deploy/modal_app.py` to a current commit of the
  model repository.
- **CUDA out of memory, or not enough KV cache for one sequence.** The engine already passes
  `--max-model-len 16384`. Lower it with `"VLLM_EXTRA_ARGS": "--max-model-len 8192"` in `ENV`;
  later flags win.
- **Boot is too slow.** Add `"VLLM_EXTRA_ARGS": "--enforce-eager"` to `ENV`. It skips
  torch.compile and CUDA graph capture, about two minutes of the cold start, at the cost of
  slower inference.

## Other GPU hosts

`docker/Dockerfile` is the image Modal builds. Building and running it with Docker should work on
any NVIDIA host, but that is untested outside Modal.

```bash
docker build -f docker/Dockerfile -t voxtral-ar-realtime .
docker run --gpus all -p 8000:8000 -v voxtral-models:/models -e API_KEY="$KEY" voxtral-ar-realtime
```

## Run it without a GPU

The whole system runs on a laptop against a fake vLLM that speaks the exact upstream protocol
and returns placeholder words. This is how the gateway is developed and tested.

```bash
uv venv .venv --python 3.12 && . .venv/bin/activate && uv pip install -e ".[dev]"
python -m tests.fake_vllm --port 9000 &
UPSTREAM_URL=ws://localhost:9000/v1/realtime python -m voxtral_rt
# open http://localhost:8000 and press Start, or stream a 16 kHz mono WAV file:
python clients/stream_wav.py --url ws://localhost:8000/v1/transcribe --wav sample.wav
```

Convert any recording first with
`ffmpeg -i in.m4a -ac 1 -ar 16000 -sample_fmt s16 sample.wav`. Add `--token "$KEY"` to the
client when the server has an `API_KEY`.

## API

| Route | Purpose | Auth |
| --- | --- | --- |
| `GET /` | Demo page | public |
| `GET /health` | `200 {"status":"ok"}` once the engine is up, `503 {"status":"starting"}` before | public |
| `WS /v1/transcribe` | One utterance per connection | `?token=` |
| `GET /v1/metrics/stream` | Server-Sent Events, one server-wide metrics snapshot per second | `?token=` |

`/v1/transcribe`: the server sends `{"type":"ready"}`; the client sends binary frames of raw
PCM16 little-endian, mono, 16 kHz (any chunk size; the demo sends 100 ms = 3,200 bytes) and
finally the text frame `{"type":"stop"}`. The server streams `{"type":"delta","text":"…"}`,
about once a second a `{"type":"metrics", …}` snapshot of the session (fields below), then
`{"type":"final","text":"…","metrics":{…}}` and closes with 1000.

Errors are `{"type":"error","code":"…","message":"…"}` followed by a close. A missing or wrong
token is not an error event: the WebSocket closes with 1008 right after it opens.

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
curl -N "https://<your-url>/v1/metrics/stream?token=$KEY"
```

## Configuration

All settings are environment variables. On Modal, add them to `ENV` in `deploy/modal_app.py`;
`API_KEY` comes from the secret instead.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_ID` | `mistralai/Voxtral-Mini-4B-Realtime-Arabic` | Hugging Face repo, also the served model name |
| `MODEL_REVISION` | `a8b0156f…` | Pinned commit of the model repo |
| `MODELS_DIR` | `/models` | Where weights live; mount a volume here |
| `PORT` | `8000` | Public gateway port |
| `ENGINE_PORT` | `8001` | vLLM's port, bound to `127.0.0.1` only |
| `UPSTREAM_URL` | unset | If set, skip weights and engine and use this realtime URL (development) |
| `API_KEY` | unset | Clients must send `?token=<key>`. Unset disables auth and logs a warning; the Modal app requires it |
| `MAX_SESSIONS` | `4` | Concurrent transcription sessions |
| `MAX_SESSION_SECONDS` | `300` | Hard cap per session |
| `ENGINE_READY_TIMEOUT_S` | `900` | How long to wait for vLLM on boot |
| `VLLM_EXTRA_ARGS` | empty | Extra `vllm serve` flags, appended last so they override the defaults |

## Known limitations

- One utterance per connection: `stop` ends it, and a new utterance needs a new connection.
- No voice-activity detection or automatic segmentation.
- The torch.compile cache is not persisted, so every cold start compiles again.
- No load test yet: the number of concurrent sessions one L4 sustains is unknown.

## Development

```
voxtral_rt/      config.py, upstream.py (vLLM protocol), gateway.py, engine.py, weights.py,
                 metrics.py, web/ (demo page)
tests/           fake_vllm.py (CPU stand-in for vLLM) and the test suite
clients/         stream_wav.py
docker/          Dockerfile on top of vllm/vllm-openai:v0.31.0
deploy/          modal_app.py
```

`ruff check . && ruff format --check . && pytest` is what CI runs on every pull request; none of
it needs a GPU. [CLAUDE.md](CLAUDE.md) has the code principles and the CPU-only working rules.

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE). The model
weights are Mistral AI's and are covered by the license on the
[model card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-Arabic).
