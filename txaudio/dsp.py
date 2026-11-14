"""Small DSP helpers: tone energy banks and WAV I/O."""

from __future__ import annotations

import wave
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def _basis(win_n: int, freqs: tuple, fs: int) -> np.ndarray:
    n = np.arange(win_n)
    w = np.hanning(win_n)
    f = np.asarray(freqs)[None, :]
    return (w[:, None] * np.exp(-2j * np.pi * f * n[:, None] / fs)).astype(np.complex64)


def tone_energy(x: np.ndarray, starts: np.ndarray, win_n: int, freqs, fs: int) -> np.ndarray:
    """Energy of each tone in a Hann window starting at each index.

    Returns an array of shape (len(starts), len(freqs)).  Windows that run
    past either end of ``x`` see zeros there.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if starts.size == 0:
        return np.zeros((0, len(freqs)), dtype=np.float32)
    idx = starts[:, None] + np.arange(win_n)[None, :]
    valid = (idx >= 0) & (idx < len(x))
    frames = np.where(valid, x[np.clip(idx, 0, max(len(x) - 1, 0))], 0.0).astype(np.float32)
    spec = frames @ _basis(win_n, tuple(float(f) for f in freqs), fs)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32)


def write_wav(path: str, x: np.ndarray, fs: int) -> None:
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


def read_wav(path: str, fs: int) -> np.ndarray:
    """Read a PCM WAV as mono float, resampled to ``fs`` if needed."""
    with wave.open(path, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != fs:
        t = np.arange(int(len(x) * fs / rate)) * (rate / fs)
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x
