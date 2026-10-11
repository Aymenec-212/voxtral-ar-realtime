// AudioWorklet processor: downmix to mono, resample to 16 kHz, emit PCM16 frames of ~frameMs.
// Runs off the main thread; posts {pcm: ArrayBuffer, last: bool} messages to mic.js.

const TARGET_RATE = 16000;

class PcmWriter extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const frameMs = (options.processorOptions && options.processorOptions.frameMs) || 100;
    this.frameSamples = Math.round((TARGET_RATE * frameMs) / 1000);
    this.ratio = sampleRate / TARGET_RATE; // `sampleRate` is the worklet's global
    this.pending = new Float32Array(0);    // input samples not yet consumed by the resampler
    this.pos = 0;                          // fractional read position into `pending`
    this.out = new Int16Array(this.frameSamples);
    this.outLen = 0;
    this.port.onmessage = (event) => {
      if (event.data === "flush") this.flush(true);
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const n = input[0].length;
    const mono = new Float32Array(n);
    for (const channel of input) {
      for (let i = 0; i < n; i++) mono[i] += channel[i] / input.length;
    }

    const merged = new Float32Array(this.pending.length + n);
    merged.set(this.pending);
    merged.set(mono, this.pending.length);
    this.pending = merged;

    // Linear interpolation is plenty for speech going into a 16 kHz model.
    while (this.pos + 1 < this.pending.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const sample = this.pending[i] * (1 - frac) + this.pending[i + 1] * frac;
      this.out[this.outLen++] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
      this.pos += this.ratio;
      if (this.outLen === this.frameSamples) this.flush(false);
    }
    const consumed = Math.floor(this.pos);
    this.pending = this.pending.slice(consumed);
    this.pos -= consumed;
    return true;
  }

  flush(last) {
    const chunk = this.out.slice(0, this.outLen);
    this.out = new Int16Array(this.frameSamples);
    this.outLen = 0;
    this.port.postMessage({ pcm: chunk.buffer, last }, [chunk.buffer]);
  }
}

registerProcessor("pcm-writer", PcmWriter);
