# GPU checkpoint runbook

Aymen runs these; the agent cannot. Each step says what to run, what to look for, and what to
paste back. Run everything from the repo root. Target cost for checkpoint A: under $2 (an L4 is
about $0.80/hour; the serve step should stay under 30 minutes of GPU time).

## One-time setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # includes modal==1.6.1 and websockets for the client
modal setup                        # browser login, once
```

You also need `ffmpeg` on your laptop to convert a sample clip (see A3).

---

## Checkpoint A: smoke test

### A1. Fill the weights volume (CPU only, no GPU time)

```bash
time modal run deploy/modal_app.py::download_weights
```

What happens: Modal builds the image from `docker/Dockerfile` the first time (several minutes,
it pulls the ~10 GB vLLM base), then a CPU container downloads the model into the Modal Volume
`voxtral-models` and prints every file with its size.

**Paste back:**
- the `real` time printed by `time`
- the full file list and total size printed at the end
- any error, verbatim (see "If something breaks" below)

### A2. Serve on an L4

```bash
modal serve deploy/modal_app.py
```

Wait for a line with the public URL, something like
`https://<workspace>--voxtral-ar-realtime-server.modal.run` (the exact shape is printed by
Modal). Keep this terminal open; the logs scroll here.

In a second terminal, poll readiness and time it:

```bash
URL=https://<paste the URL here>
time (until [ "$(curl -s -o /dev/null -w '%{http_code}' $URL/health)" = 200 ]; do sleep 5; done)
```

**Paste back, from the `modal serve` logs:**
- the vLLM banner line that prints its version (`INFO ... vLLM API server version 0.31.0`)
- the line(s) that say which weight file vLLM loaded: look for `Loading weights`,
  `consolidated.safetensors` or `model.safetensors`, and `Loading weights took ... seconds`
- the lines with `model weights take ... GiB`, `GPU KV cache size`, and `Maximum concurrency for`
- a line containing `Route: /v1/realtime` (proves the realtime endpoint is registered)
- every `WARNING` and `ERROR` line (paste them all, even if they look harmless)
- the `voxtral_rt.engine INFO engine is ready` line
- the `real` time from the `until` loop above (that is cold start to first `/health` 200)

### A3. Stream a real clip

Make a 16 kHz mono PCM16 WAV from any short Arabic or Darija recording (10 to 30 seconds):

```bash
ffmpeg -i my_clip.m4a -ac 1 -ar 16000 -sample_fmt s16 sample.wav
python clients/stream_wav.py --url wss://<host from the URL above>/v1/transcribe --wav sample.wav
```

The client streams at real-time pace, prints deltas as they arrive, then `[final] ...`.

**Paste back:**
- the full client output (stderr included: it has `[first delta after X.XXs]` and `[done in ...]`)
- one sentence on whether the transcript is right
- any `WARNING`/`ERROR` lines that appeared in the `modal serve` logs during the stream

### A4. Open the page

Open the URL in a browser, press Start, say a sentence, press Stop. This is optional for
checkpoint A; checkpoint B does it properly.

**Paste back:** whether captions appeared, and the browser console errors if any.

### A5. Scale to zero

Press Ctrl+C in the `modal serve` terminal. In the Modal dashboard (Apps → voxtral-ar-realtime)
confirm that no container is running after a minute or two.

**Paste back:** "scaled to zero" or what you saw instead.

---

## If something breaks

Paste the error verbatim; these are the failures the agent anticipates, with the first thing
to try.

- **Image build fails on `uv: not found`.** Edit `docker/Dockerfile`, replace the `RUN` line
  with `RUN python3 -m pip install --no-deps --no-cache-dir .` and retry A1.
- **Modal complains it cannot find Python or pip in the image.** Add `add_python="3.12"` to
  `modal.Image.from_dockerfile(...)` in `deploy/modal_app.py`. Report it: it changes which
  Python runs our launcher and we may need to adjust the `CMD`.
- **vLLM exits with a tokenizer error.** `--tokenizer-mode mistral` is already passed; paste
  the traceback.
- **CUDA out of memory.** Add `"VLLM_EXTRA_ARGS": "--max-model-len 32768"` to `ENV` in
  `deploy/modal_app.py` (the default context is 131072 tokens, about 3 hours of audio).
- **Boot is very slow (compilation).** Add `"VLLM_EXTRA_ARGS": "--enforce-eager"` to `ENV`.
  Boot gets faster, inference slower. The vLLM recipe for this model family instead uses
  `--compilation_config '{"cudagraph_mode": "PIECEWISE"}'` with `VLLM_DISABLE_COMPILE_CACHE=1`;
  that is the second thing to try.
- **`/health` is 200 but the WebSocket fails (client prints `[closed]` or a 4xx).** Paste the
  client error and the Modal log. The likely cause is the `@app.server` decorator not proxying
  WebSockets; the fallback is `@app.function` + `@modal.web_server(8000)`, which the agent will
  write.
- **`modal serve` says `startup_timeout` exceeded.** Our launcher gives up after 900 s with a
  clear message; if Modal's 20-minute limit fires first, paste the last 50 log lines.

## After checkpoint A, the agent will

- narrow `ensure_weights` to the one weight format vLLM loads (halves the download)
- bake any flag that proved necessary into `engine.py` as a default
- write the checkpoint B steps (browser session, four parallel clients, time-limit behaviour)

---

## Checkpoint B: real use

Written after checkpoint A results are in.
