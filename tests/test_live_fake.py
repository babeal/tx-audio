"""Drive the live sound-card loop against a fake sounddevice "room".

Two run_live() loops (sender and receiver) run in threads; a fake Stream
calls both audio callbacks on a shared clock, mixing both speakers into both
microphones.  This exercises live.py's threading and buffering without
hardware.
"""

import os
import sys
import threading
import time
import types

import numpy as np
import pytest

from txaudio.params import Params
from txaudio.protocol import Receiver, Sender

FS = 48000
BLOCK = 1024
SPEEDUP = 3.0


class Air:
    def __init__(self):
        self.streams = []
        self.lock = threading.Lock()
        self.prev = {}
        self.stop = False

    def run(self):
        period = BLOCK / FS / SPEEDUP
        nxt = time.time()
        while not self.stop:
            with self.lock:
                streams = list(self.streams)
            outs = {}
            for s in streams:
                mic = np.zeros((BLOCK, 1), dtype=np.float32)
                for o, x in self.prev.items():
                    mic[:, 0] += x * (1.0 if o is s else 0.3)
                mic[:, 0] += 0.002 * np.random.standard_normal(BLOCK).astype(np.float32)
                out = np.zeros((BLOCK, 1), dtype=np.float32)
                s.callback(mic, out, BLOCK, None, None)
                outs[s] = out[:, 0].copy()
            self.prev = outs
            nxt += period
            time.sleep(max(0.0, nxt - time.time()))


AIR = Air()


class FakeStream:
    def __init__(self, samplerate, channels, dtype, blocksize, callback, device=None):
        assert samplerate == FS and blocksize == BLOCK
        self.callback = callback

    def __enter__(self):
        with AIR.lock:
            AIR.streams.append(self)
        return self

    def __exit__(self, *a):
        with AIR.lock:
            AIR.streams.remove(self)


@pytest.mark.slow
def test_live_loop_with_fake_soundcard(monkeypatch):
    from txaudio import live

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(Stream=FakeStream))
    p = Params()
    data = os.urandom(90)
    tx_log, rx_log = [], []
    tx = Sender(data, p, log=tx_log.append)
    rx = Receiver(p, log=rx_log.append)
    air = threading.Thread(target=AIR.run, daemon=True)
    air.start()
    t_rx = threading.Thread(target=live.run_live,
                            args=(rx, FS, lambda: tx.finished),
                            kwargs=dict(blocksize=BLOCK, timeout_s=120), daemon=True)
    t_rx.start()
    live.run_live(tx, FS, lambda: tx.finished, blocksize=BLOCK, timeout_s=120)
    t_rx.join(timeout=30)
    AIR.stop = True
    assert tx.done, (tx_log[-10:], rx_log[-10:])
    assert rx.result() == data
