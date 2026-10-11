# v0.31.0 (released 2026-10-04) is the newest stable vLLM release on Docker Hub. It carries the
# realtime WebSocket endpoint (vllm/entrypoints/speech_to_text/realtime) that upstream.py
# targets; the protocol files are identical between this tag and main as of 2026-10-11.
FROM vllm/vllm-openai:v0.31.0
# The base entrypoint runs `vllm serve`; Modal needs an entrypoint that execs "$@".
ENTRYPOINT []
ENV PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY voxtral_rt ./voxtral_rt
# --no-deps: everything we need already ships with vLLM; never touch its pins.
# The base image manages its packages with `uv pip install --system`, so we do the same.
RUN uv pip install --system --no-deps --no-cache .
CMD ["python3", "-m", "voxtral_rt"]
