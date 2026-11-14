"""Run a Sender or Receiver on the sound card (full duplex)."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np


def run_live(endpoint, fs: int, should_stop, device=None, blocksize: int = 1024,
             lead_s: float = 1.0, timeout_s: float | None = None, status=None) -> None:
    """Pump audio between ``endpoint`` and the default (or given) devices.

    ``should_stop()`` is polled from the main thread; DSP runs there too, so
    the audio callback only copies buffers.
    """
    try:
        import sounddevice as sd
    except OSError as e:  # PortAudio missing
        raise SystemExit(f"sounddevice could not load PortAudio: {e}\n"
                         "Install it (macOS: brew install portaudio; Debian/Ubuntu: "
                         "apt install libportaudio2), or use encode/decode/simulate.")

    mic_q: queue.Queue = queue.Queue()
    lock = threading.Lock()
    outbuf = [np.zeros(0, dtype=np.float32)]
    xruns = [0]
    starved = [0]

    def callback(indata, outdata, frames, t, st):
        if st:
            xruns[0] += 1
        mic_q.put(indata[:, 0].copy())
        with lock:
            b = outbuf[0]
            n = min(frames, len(b))
            outdata[:n, 0] = b[:n]
            outdata[n:, 0] = 0.0
            if 0 < n < frames:
                starved[0] += 1
            outbuf[0] = b[n:]

    start = time.time()
    last_status = 0.0
    stopping = False
    with sd.Stream(samplerate=fs, channels=1, dtype="float32", blocksize=blocksize,
                   callback=callback, device=device):
        while True:
            with lock:
                have = len(outbuf[0])
            stopping = stopping or should_stop()
            if have < lead_s * fs and not stopping:
                chunk = endpoint.next_chunk(blocksize * 4)
                with lock:
                    outbuf[0] = np.concatenate([outbuf[0], chunk])
            got = []
            while True:
                try:
                    got.append(mic_q.get_nowait())
                except queue.Empty:
                    break
            if got:
                endpoint.feed(np.concatenate(got))
            if stopping:
                with lock:
                    left = len(outbuf[0])
                if left == 0:
                    break
            if timeout_s and time.time() - start > timeout_s:
                print("timed out")
                break
            if status and time.time() - last_status > 2.0:
                status()
                last_status = time.time()
            if not got:
                time.sleep(0.01)
    if xruns[0]:
        print(f"note: {xruns[0]} audio over/underruns")
    if starved[0] > 2:
        print(f"note: the decoder fell behind {starved[0]} times; gaps in the output "
              "can break frames (try a larger --step or a faster machine)")
