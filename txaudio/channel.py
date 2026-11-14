"""Acoustic channel simulator for testing without speakers.

Models what happens between a speaker and a microphone: delay, gain (with
slow wobble), early echoes plus a reverb tail, sample-clock drift between
the two devices, and background noise.  ``simulate_duplex`` runs a sender
and a receiver against each other where each microphone hears *both*
speakers, so the receiver's tones and the sender's music overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .params import FS


@dataclass
class ChannelConfig:
    gain: float = 0.5
    delay_ms: float = 5.0
    echoes: list = field(default_factory=list)  # [(delay_ms, gain), ...]
    reverb_rt60: float = 0.0  # seconds; 0 disables the diffuse tail
    reverb_level: float = 0.0  # tail energy relative to the direct sound
    drift_ppm: float = 0.0  # receiver clock relative to sender clock
    noise_db: Optional[float] = None  # noise level in dBFS (RMS), None = silent
    wobble_db: float = 0.0  # slow +/- gain variation
    wobble_hz: float = 0.2
    dropouts: list = field(default_factory=list)  # [(start_s, dur_s), ...] heavy hits
    seed: int = 0


def room_ir(cfg: ChannelConfig, fs: int = FS) -> np.ndarray:
    d = int(cfg.delay_ms * fs / 1000)
    length = d + 1
    for ms, _ in cfg.echoes:
        length = max(length, int(ms * fs / 1000) + d + 1)
    if cfg.reverb_rt60 > 0:
        length = max(length, d + int(cfg.reverb_rt60 * fs))
    h = np.zeros(length, dtype=np.float64)
    h[d] = 1.0
    for ms, g in cfg.echoes:
        h[d + int(ms * fs / 1000)] += g
    if cfg.reverb_rt60 > 0 and cfg.reverb_level > 0:
        rng = np.random.default_rng(cfg.seed + 77)
        n = int(cfg.reverb_rt60 * fs)
        t = np.arange(n) / fs
        tail = rng.standard_normal(n) * np.exp(-6.91 * t / cfg.reverb_rt60)
        tail[: int(0.005 * fs)] = 0  # tail starts after the direct sound
        tail *= np.sqrt(cfg.reverb_level / np.sum(tail ** 2))
        h[d: d + n] += tail
    return (h * cfg.gain).astype(np.float32)


class StreamConv:
    """Overlap-save FFT convolution for streaming chunks."""

    def __init__(self, h: np.ndarray):
        self.h = h
        self.hist = np.zeros(len(h) - 1, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if len(self.h) == 1:
            return x * self.h[0]
        z = np.concatenate([self.hist, x])
        n = 1 << int(np.ceil(np.log2(len(z) + len(self.h))))
        y = np.fft.irfft(np.fft.rfft(z, n) * np.fft.rfft(self.h, n), n)
        self.hist = z[len(z) - (len(self.h) - 1):] if len(self.h) > 1 else self.hist
        return y[len(self.h) - 1: len(z)].astype(np.float32)


class StreamResample:
    """Fractional-rate resampler: output clock runs (1 + ppm) vs input."""

    def __init__(self, ppm: float, latency: int = 4800):
        self.ratio = 1.0 + ppm * 1e-6
        self.buf = np.zeros(latency, dtype=np.float32)
        self.base = 0  # absolute input index of buf[0]
        self.pos = 0.0  # absolute input position of the next output sample

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.ratio == 1.0:
            return x
        self.buf = np.concatenate([self.buf, x])
        n = len(x)
        t = self.pos + np.arange(n) * self.ratio - self.base
        self.pos += n * self.ratio
        i = np.floor(t).astype(np.int64)
        f = (t - i).astype(np.float32)
        ok = (i >= 1) & (i + 2 < len(self.buf))
        ic = np.clip(i, 1, len(self.buf) - 3)
        xm, x0, x1, x2 = (self.buf[ic + k] for k in (-1, 0, 1, 2))
        # Catmull-Rom cubic interpolation
        y = x0 + 0.5 * f * (x1 - xm + f * (2 * xm - 5 * x0 + 4 * x1 - x2 + f * (3 * (x0 - x1) + x2 - xm)))
        y = np.where(ok, y, 0.0).astype(np.float32)
        cut = int(self.pos - self.base) - 8
        if cut > 0:
            self.buf = self.buf[cut:]
            self.base += cut
        return y


class Channel:
    def __init__(self, cfg: ChannelConfig, fs: int = FS):
        self.cfg = cfg
        self.fs = fs
        self.conv = StreamConv(room_ir(cfg, fs))
        self.rs = StreamResample(cfg.drift_ppm)
        self.n = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = self.rs(self.conv(np.asarray(x, dtype=np.float32)))
        if self.cfg.wobble_db:
            t = (self.n + np.arange(len(y))) / self.fs
            db = self.cfg.wobble_db * np.sin(2 * np.pi * self.cfg.wobble_hz * t)
            y = y * (10 ** (db / 20)).astype(np.float32)
        if self.cfg.dropouts:
            t = (self.n + np.arange(len(y))) / self.fs
            for t0, dur in self.cfg.dropouts:
                hit = (t >= t0) & (t < t0 + dur)
                if hit.any():
                    # A loud broadband crash (door slam, dropped pan) over the music.
                    rng = np.random.default_rng(int(t0 * 1000) + self.n)
                    y = np.where(hit, y * 0.1 + 0.5 * rng.standard_normal(len(y)), y).astype(np.float32)
        self.n += len(y)
        return y


class Noise:
    def __init__(self, db: Optional[float], seed: int = 0, fs: int = FS):
        self.rms = None if db is None else 10 ** (db / 20)
        self.rng = np.random.default_rng(seed)

    def __call__(self, n: int) -> np.ndarray:
        if self.rms is None:
            return np.zeros(n, dtype=np.float32)
        # Pinkish noise: white plus a smoothed random walk.
        w = self.rng.standard_normal(n)
        return (self.rms * w).astype(np.float32)


@dataclass
class DuplexResult:
    ok: bool
    seconds: float
    frames_sent: int
    blocks: int
    handshake_s: Optional[float]
    safety_codes: tuple
    rx_stats: dict
    tx_log: list
    rx_log: list
    rx_mic: Optional[np.ndarray] = None
    tx_mic: Optional[np.ndarray] = None


def simulate_duplex(data: bytes, p=None, forward: ChannelConfig = None, back: ChannelConfig = None,
                    tx_self: ChannelConfig = None, rx_self: ChannelConfig = None,
                    noise_db: Optional[float] = -60.0, max_seconds: float = 3600,
                    chunk: int = 2400, record: bool = False, seed: Optional[int] = 7, passphrase: Optional[str] = None,
                    receiver_kwargs=None, sender_kwargs=None) -> DuplexResult:
    """Run a full send with each mic hearing both speakers."""
    from .params import Params
    from .protocol import Receiver, Sender

    p = p or Params()
    forward = forward or ChannelConfig()
    back = back or ChannelConfig(gain=forward.gain, delay_ms=forward.delay_ms,
                                 echoes=forward.echoes, reverb_rt60=forward.reverb_rt60,
                                 reverb_level=forward.reverb_level,
                                 drift_ppm=-forward.drift_ppm, seed=forward.seed + 1)
    tx_self = tx_self or ChannelConfig(gain=1.0, delay_ms=1.0, echoes=[(4, 0.3)], seed=3)
    rx_self = rx_self or ChannelConfig(gain=1.0, delay_ms=1.0, echoes=[(4, 0.3)], seed=4)
    tx_log, rx_log = [], []
    tx = Sender(data, p, passphrase=passphrase, seed=seed, log=lambda m: tx_log.append(m),
                **(sender_kwargs or {}))
    rx = Receiver(p, passphrase=passphrase, log=lambda m: rx_log.append(m),
                  **(receiver_kwargs or {}))
    ab, ba, aa, bb = Channel(forward), Channel(back), Channel(tx_self), Channel(rx_self)
    n_rx, n_tx = Noise(noise_db, 11), Noise(noise_db, 12)
    rec_rx, rec_tx = [], []
    t = 0
    while not tx.finished and t < max_seconds * p.fs:
        a = tx.next_chunk(chunk)
        b = rx.next_chunk(chunk)
        rx_mic = ab(a) + bb(b) + n_rx(chunk)
        tx_mic = aa(a) + ba(b) + n_tx(chunk)
        rx.feed(rx_mic)
        tx.feed(tx_mic)
        if record:
            rec_rx.append(rx_mic)
            rec_tx.append(tx_mic)
        t += chunk
    rx.flush()
    return DuplexResult(
        ok=rx.result() == data,
        seconds=t / p.fs,
        frames_sent=tx.frames_sent,
        blocks=tx.total,
        handshake_s=tx.handshake_s,
        safety_codes=(tx.safety_code, rx.safety_code),
        rx_stats=dict(rx.demod.stats),
        tx_log=tx_log,
        rx_log=rx_log,
        rx_mic=np.concatenate(rec_rx) if record else None,
        tx_mic=np.concatenate(rec_tx) if record else None,
    )
