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

## What the first run established (11 Oct 2026)

vLLM v0.31.0 accepted every flag, resolved the architecture `VoxtralRealtimeGeneration`,
inferred the Mistral format from `consolidated*.safetensors`, loaded the weights in 6.9 s, and
the model takes 8.43 GiB of GPU memory. Container start to the vLLM banner took about a minute.
It then crashed in `profile_run` because the base image lacks vLLM's `audio` extra (see "If
something breaks"); the image now installs it and fails the build if it is missing.

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
- the line(s) that say which weight file vLLM loaded: look for `Loading weights` and
  `Loading weights took ... seconds`. Expected: it mentions `consolidated.safetensors` (the
  Mistral format). If it mentions `model.safetensors` instead, say so: that file is no longer
  downloaded and the boot would have failed.
- the lines with `model weights take ... GiB`, `GPU KV cache size`, and `Maximum concurrency for`
- a line containing `Route: /v1/realtime` (proves the realtime endpoint is registered)
- every `WARNING` and `ERROR` line (paste them all, even if they look harmless)
- the `voxtral_rt.engine INFO engine is ready` line
- the `real` time from the `until` loop above (that is cold start to first `/health` 200)

### A3. Speak into the page

Open the URL in Chrome, press **Start**, allow the microphone, speak a few sentences of Arabic or
Darija, press **Stop**. Captions should appear while you speak, then the final transcript
replaces them and the status reads "Done.".

**Paste back:**
- whether captions appeared while speaking, and roughly how far behind your speech they were
- the final transcript and one sentence on whether it is right
- anything in the red error line under the captions, verbatim
- the browser console (F12 → Console) if anything is red
- any `WARNING`/`ERROR` lines that appeared in the `modal serve` logs during the session

### A4. Optional: stream a WAV file

Useful to measure time to first delta without a microphone in the loop.

```bash
ffmpeg -i my_clip.m4a -ac 1 -ar 16000 -sample_fmt s16 sample.wav
python clients/stream_wav.py --url wss://<host from the URL above>/v1/transcribe --wav sample.wav
```

**Paste back:** the full client output (stderr included: it has `[first delta after X.XXs]`).

### A5. Scale to zero

Press Ctrl+C in the `modal serve` terminal. In the Modal dashboard (Apps → voxtral-ar-realtime)
confirm that no container is running after a minute or two.

**Paste back:** "scaled to zero" or what you saw instead.

---

## If something breaks

Paste the error verbatim; these are the failures the agent anticipates, with the first thing
to try.

- **vLLM dies in `profile_run` with `ImportError: soundfile is not installed`.** This is the
  checkpoint A failure. The `vllm/vllm-openai` image ships vLLM without its `audio` extra; the
  Dockerfile now installs it (`av scipy soundfile soxr mistral_common[audio]`) and the image
  build fails on CPU if any of them is missing, so this cannot reach a GPU again. If you still
  see it, the running container is on a stale image: rebuild with
  `MODAL_FORCE_BUILD=1 modal serve deploy/modal_app.py`.
- **The container exits right after `voxtral_rt exited with code N during startup`.** That is
  intended: the Modal server now stops as soon as the launcher dies instead of waiting 20 minutes
  for port 8000. The real error is a few lines above it in the log; paste those.
- **Image build fails on `uv: not found`.** Edit `docker/Dockerfile`, replace the `RUN` line
  with `RUN python3 -m pip install --no-deps --no-cache-dir .` and retry A1.
- **A1 fails with `RevisionNotFoundError`.** The pinned `MODEL_REVISION` may have moved on the
  Hub. Report it verbatim; do not change the pin yourself.
- **vLLM exits with a tokenizer error.** `--tokenizer-mode mistral` is already passed; paste
  the traceback.
- **CUDA out of memory, or "not enough KV cache for one sequence".** The engine already
  passes `--max-model-len 16384`. Lower it further with `"VLLM_EXTRA_ARGS": "--max-model-len
  8192"` in `ENV` (later flags win) and paste the memory lines from the log.
- **Boot is very slow (compilation).** Add `"VLLM_EXTRA_ARGS": "--enforce-eager"` to `ENV`.
  Boot gets faster, inference slower. The `cudagraph_mode: PIECEWISE` flag from the vLLM recipe
  is not needed on v0.31.0: the Voxtral realtime model only rejects pure `FULL` CUDA graphs and
  the default `FULL_AND_PIECEWISE` is allowed.
- **`/health` is 200 but the WebSocket fails (client prints `[closed]` or a 4xx).** Paste the
  client error and the Modal log. The likely cause is the `@app.server` decorator not proxying
  WebSockets; the fallback is `@app.function` + `@modal.web_server(8000)`, which the agent will
  write.
- **`modal serve` says `startup_timeout` exceeded.** Our launcher gives up after 900 s with a
  clear message; if Modal's 20-minute limit fires first, paste the last 50 log lines.

## After checkpoint A, the agent will

- confirm from the logs that vLLM loaded `consolidated.safetensors`, the only format we download
- bake any flag that proved necessary into `engine.py` as a default
- write the checkpoint B steps (browser session, four parallel clients, time-limit behaviour)

---

## Checkpoint B: real use

Written after checkpoint A results are in.
