# voxtral-ar-realtime — notes for coding agents

One portable container image that serves Mistral's Voxtral-Mini-4B-Realtime-Arabic for live,
streaming Arabic speech-to-text, behind a small FastAPI gateway with a browser demo. vLLM runs
on a private localhost port; our gateway is the only public surface. The full plan lives with
Aymen; this file repeats the two sections every session must follow.

## Working constraints (read first)

You, the coding agent, have no GPU, no Modal credentials and no Docker daemon. Aymen runs
everything that needs a GPU on Modal and reports results back. Design all work so it can be
built and tested on CPU.

- **Never** download the model weights (~9 GB per copy). Never pull or build the
  `vllm/vllm-openai` image (~10 GB). Never `pip install vllm`.
- **Do** develop and test the gateway against the fake vLLM server (`tests/fake_vllm.py`) that
  implements the exact upstream protocol. That fake is your stand-in for the GPU.
- **Do** write `deploy/modal_app.py`, the `Dockerfile` and the CI workflow, but treat them as
  untested until Aymen runs them. Keep them short and obvious so they can be reviewed by reading.
- At each GPU checkpoint, stop and hand Aymen a runbook (`RUNBOOK.md`): exact commands, what to
  look for, what to paste back.
- End every phase with a short report: what changed, decisions taken, open questions. Aymen
  relays these to his reviewer.
- When a fact about vLLM or Modal is uncertain, check the current docs or source, pin versions,
  and note the assumption in the code with a one-line comment. Do not guess silently.

## Code principles

Write **deep classes with small interfaces**, in the sense of Ousterhout's *A Philosophy of
Software Design*: each module hides a lot of complexity behind a few simple methods. Clean,
boring, readable code. Do not overengineer.

- **Deep, not wide.** The vLLM protocol (session handshake, base64, commit/final, error codes)
  lives entirely inside `voxtral_rt/upstream.py`. Nothing outside it knows vLLM's event names.
- **Small interfaces.** Prefer 2 to 4 public methods per class. If a caller must call methods in
  a specific order, hide that order inside the class.
- **No speculative abstractions.** No abstract base classes, plugin registries, factories of
  factories, dependency-injection frameworks or generic "provider" layers. One concrete
  implementation; a plain function parameter is enough to swap in the fake for tests.
- **Few files, few dependencies.** Runtime deps: `fastapi`, `uvicorn`, `websockets`,
  `huggingface_hub`. All four already ship in the vLLM image. Dev deps: `pytest`,
  `pytest-asyncio`, `httpx`. No numpy in the gateway (it forwards raw PCM bytes).
- **Config is a frozen dataclass read from env vars once** (`voxtral_rt/config.py`). No YAML,
  no config classes per module.
- **Errors are explicit.** Map upstream failures to a small set of client-facing error codes
  (`busy`, `time_limit`, `bad_request`, `upstream_error`). Log once, at the boundary.
- Type hints everywhere, Python 3.12, `ruff` for lint/format. Comments explain *why*, not *what*.
- Tests exercise behaviour through public interfaces, not private helpers.

## Everyday commands

```bash
uv venv .venv --python 3.12 && . .venv/bin/activate
uv pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
# manual CPU end-to-end:
python -m tests.fake_vllm --port 9000 &
UPSTREAM_URL=ws://localhost:9000/v1/realtime python -m voxtral_rt
```
