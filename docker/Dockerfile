# v0.31.0 (released 2026-10-04) is the newest stable vLLM release on Docker Hub. It carries the
# realtime WebSocket endpoint (vllm/entrypoints/speech_to_text/realtime) that upstream.py
# targets; the protocol files are identical between this tag and main as of 2026-10-11.
FROM vllm/vllm-openai:v0.31.0
# The base entrypoint runs `vllm serve`; Modal needs an entrypoint that execs "$@".
ENTRYPOINT []
ENV PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1
# The base ships vLLM without its "audio" extra, so Voxtral's profile run dies on a billed GPU with
# "soundfile is not installed". Install it under a freeze of the base's pins; fail the build if missing.
RUN uv pip freeze --system | grep -E '^[A-Za-z0-9_.-]+==' > /tmp/constraints.txt \
 && uv pip install --system --no-cache -c /tmp/constraints.txt \
      av scipy soundfile soxr "mistral_common[audio]" \
 && python3 -c "import av, scipy, soundfile, soxr; \
from mistral_common.imports import assert_soundfile_installed, assert_soxr_installed; \
assert_soundfile_installed(); assert_soxr_installed()" \
 && rm /tmp/constraints.txt
WORKDIR /app
COPY pyproject.toml README.md ./
COPY voxtral_rt ./voxtral_rt
# The base ships python3.12 and pip but may have no unversioned `python`, which Modal's
# runtime can expect. Harmless for plain `docker run` users.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python
# --no-deps: everything we need already ships with vLLM; never touch its pins.
# The base image manages its packages with `uv pip install --system`, so we do the same.
RUN uv pip install --system --no-deps --no-cache .
CMD ["python3", "-m", "voxtral_rt"]
