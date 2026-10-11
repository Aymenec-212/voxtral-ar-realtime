// Demo client: microphone → PCM16 16 kHz frames → WS /v1/transcribe → live captions.

const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const tokenInput = document.getElementById("token");
const statusEl = document.getElementById("status");
const captionsEl = document.getElementById("captions");
const metricsEl = document.getElementById("metrics");
const errorEl = document.getElementById("error");

let ws = null;
let audioContext = null;
let mediaStream = null;
let workletNode = null;

function setStatus(text) { statusEl.textContent = text; }

// Lag and first-text include the model's built-in ~480 ms transcription delay.
function showMetrics(m, finished) {
  if (!m) { metricsEl.textContent = ""; return; }
  const parts = [`lag ${Math.round(m.lag_ms)} ms`, `${m.tokens_per_s.toFixed(1)} tok/s`];
  if (m.first_text_ms !== null) parts.push(`first text ${Math.round(m.first_text_ms)} ms`);
  if (finished) {
    parts.length = 0;
    parts.push(`${m.audio_received_s.toFixed(1)} s audio`, `${m.tokens} tokens`,
      `max lag ${Math.round(m.max_lag_ms)} ms`);
    if (m.first_text_ms !== null) parts.push(`first text ${Math.round(m.first_text_ms)} ms`);
    if (m.finalize_ms !== null) parts.push(`finalize ${Math.round(m.finalize_ms)} ms`);
  }
  metricsEl.textContent = parts.join(" · ");
}
function showError(text) { errorEl.textContent = text; }

function transcribeUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const token = tokenInput.value.trim();
  const query = token ? `?token=${encodeURIComponent(token)}` : "";
  return `${scheme}://${location.host}/v1/transcribe${query}`;
}

async function start() {
  startBtn.disabled = true;
  showError("");
  captionsEl.textContent = "";
  captionsEl.classList.remove("final");
  showMetrics(null);
  setStatus("Connecting…");
  try {
    await openSocket();
    await startMicrophone();
    setStatus("Listening… speak now.");
    stopBtn.disabled = false;
  } catch (err) {
    showError(String(err.message || err));
    await teardown();
  }
}

function openSocket() {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(transcribeUrl());
    ws.binaryType = "arraybuffer";
    let ready = false;
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "ready") { ready = true; resolve(); }
      else if (msg.type === "delta") captionsEl.textContent += msg.text;
      else if (msg.type === "metrics") showMetrics(msg, false);
      else if (msg.type === "final") {
        captionsEl.textContent = msg.text;
        captionsEl.classList.add("final");
        showMetrics(msg.metrics, true);
        setStatus("Done.");
      } else if (msg.type === "error") {
        showError(`${msg.code}: ${msg.message}`);
        if (msg.code !== "time_limit") setStatus("Failed.");
      }
    };
    ws.onerror = () => { if (!ready) reject(new Error("WebSocket connection failed")); };
    ws.onclose = (event) => {
      if (!ready) reject(new Error(event.code === 1008 ? "Unauthorized: wrong API key" : `Connection closed (${event.code})`));
      teardown();
    };
  });
}

async function startMicrophone() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  // Let the browser pick its native rate; the worklet resamples to 16 kHz.
  audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("/static/pcm-worklet.js");
  workletNode = new AudioWorkletNode(audioContext, "pcm-writer", {
    numberOfInputs: 1, numberOfOutputs: 0, processorOptions: { frameMs: 100 },
  });
  workletNode.port.onmessage = (event) => {
    const { pcm, last } = event.data;
    if (ws && ws.readyState === WebSocket.OPEN) {
      if (pcm.byteLength > 0) ws.send(pcm);
      if (last) ws.send(JSON.stringify({ type: "stop" }));
    }
  };
  audioContext.createMediaStreamSource(mediaStream).connect(workletNode);
}

async function stop() {
  stopBtn.disabled = true;
  setStatus("Finishing…");
  if (workletNode) {
    workletNode.port.postMessage("flush"); // the worklet answers with last=true → we send stop
  } else if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
  }
  await stopMicrophone();
}

async function stopMicrophone() {
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  if (audioContext) await audioContext.close().catch(() => {});
  mediaStream = null; audioContext = null; workletNode = null;
}

async function teardown() {
  await stopMicrophone();
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  ws = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
