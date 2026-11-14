"""Receiver -> sender feedback: short 16-FSK tone bursts.

Every message is a 2-symbol preamble, one type symbol, then a Reed-Solomon
codeword sent as nibbles (60 ms per symbol):

* ACK / NAK / DONE: block index + CRC-8, RS(7,3) -> 17 symbols (~1 s).
* KEY: the receiver's 32-byte X25519 public key + CRC-16, RS(46,34) -> 95
  symbols (~5.7 s), sent once per handshake.

The tone order and preamble come from the scheme, and every CRC also covers
a scheme tag, so bursts from another session (or noise that happens to look
like a preamble) are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from reedsolo import RSCodec

from .dsp import tone_energy
from .framing import Codec
from .modem import plateau_center
from .params import FB_GUARD, FB_SYMBOL, FB_TONES, FB_WINDOW, FS

ACK, NAK, DONE, KEY = 1, 2, 3, 4
NAMES = {ACK: "ACK", NAK: "NAK", DONE: "DONE", KEY: "KEY"}
SHORT_NSYM, KEY_NSYM = 4, 12
SHORT_LEN = 1 + 2 * (3 + SHORT_NSYM)
KEY_LEN = 1 + 2 * (34 + KEY_NSYM)
FB_THRESHOLD = 1.5  # out of 2
AMP_FB = 0.25


def crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def crc16(data: bytes) -> int:
    c = 0xFFFF
    for b in data:
        c ^= b << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


def _nibbles(b: bytes) -> list[int]:
    return [v for x in b for v in (x >> 4, x & 0xF)]


def _bytes(n: list[int]) -> bytes:
    return bytes((n[i] << 4) | n[i + 1] for i in range(0, len(n), 2))


@dataclass(frozen=True)
class Feedback:
    kind: int
    block: int = 0
    key: bytes = b""

    def __str__(self):
        if self.kind == KEY:
            return f"KEY({self.key[:4].hex()}..)"
        return f"{NAMES.get(self.kind, '?')}({self.block})"


def encode_symbols(fb: Feedback, scheme) -> list[int]:
    if fb.kind == KEY:
        body = fb.key + crc16(scheme.tag + bytes([KEY]) + fb.key).to_bytes(2, "big")
        cw = RSCodec(KEY_NSYM).encode(body)
    else:
        b = fb.block & 0xFFF
        body = bytes([(fb.kind << 4) | (b >> 8), b & 0xFF])
        cw = RSCodec(SHORT_NSYM).encode(body + bytes([crc8(scheme.tag + body)]))
    vals = [fb.kind] + _nibbles(bytes(cw))
    return list(scheme.fb_preamble) + [scheme.fb_perm[v] for v in vals]


@lru_cache(maxsize=None)
def _tone(k: int, fs: int) -> np.ndarray:
    n = int(FB_SYMBOL * fs)
    t = np.arange(n) / fs
    r = int(0.004 * fs)
    env = np.ones(n)
    env[:r] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))
    env[-r:] = env[:r][::-1]
    return (AMP_FB * env * np.sin(2 * np.pi * FB_TONES[k] * t)).astype(np.float32)


def synth(fb: Feedback, scheme, fs: int = FS) -> np.ndarray:
    return np.concatenate([_tone(k, fs) for k in encode_symbols(fb, scheme)])


def decode_body(tones: list[int], conf, scheme) -> Optional[Feedback]:
    inv = {t: v for v, t in enumerate(scheme.fb_perm)}
    v = [inv[t] for t in tones]
    kind = v[0]
    nsym = KEY_NSYM if kind == KEY else SHORT_NSYM
    raw = _bytes(v[1:])
    bconf = [min(conf[1 + 2 * i], conf[2 + 2 * i]) for i in range(len(raw))]
    msg, _ = Codec._rs_decode(RSCodec(nsym), raw, bconf, nsym, nsym // 2)
    if msg is None:
        return None
    if kind == KEY:
        key, c = msg[:32], int.from_bytes(msg[32:34], "big")
        if crc16(scheme.tag + bytes([KEY]) + key) != c:
            return None
        return Feedback(KEY, 0, key)
    body = msg[:2]
    if crc8(scheme.tag + body) != msg[2] or body[0] >> 4 != kind or kind not in (ACK, NAK, DONE):
        return None
    return Feedback(kind, ((body[0] & 0xF) << 8) | body[1])


def cancel_carryover(e: np.ndarray, lags: int = 2) -> np.ndarray:
    """Subtract reverb tails of the previous one or two decided tones.

    In a live room the last tone is still ringing when the next one is read;
    its share is estimated from the burst itself (median over symbols whose
    tone changed), then removed by decision feedback.
    """
    e = np.array(e, dtype=float)
    am = np.argmax(e, axis=1)
    top = e[np.arange(len(e)), am]
    cs = []
    for lag in range(1, lags + 1):
        r = [e[i, am[i - lag]] / (top[i - lag] + 1e-12)
             for i in range(lag, len(e)) if am[i] != am[i - lag]]
        cs.append(min(0.5, float(np.median(r))) if r else 0.0)
    out = e.copy()
    for i in range(len(e)):
        for lag, c in enumerate(cs, 1):
            if i >= lag:
                j = int(np.argmax(out[i - lag]))
                out[i, j] = max(out[i, j] - c * e[i - lag, j], 0.0)
    return out


class FeedbackDemod:
    def __init__(self, scheme, fs: int = FS):
        self.scheme = scheme
        self.fs = fs
        self.T = int(FB_SYMBOL * fs)
        self.W = int(FB_WINDOW * fs)
        self.G = int(FB_GUARD * fs)
        self.hop = self.T // 10
        self.buf = np.zeros(0, dtype=np.float32)
        self.base = 0
        self.search = 0
        self.key_type_tone = scheme.fb_perm[KEY]

    def _e(self, starts_abs):
        return tone_energy(self.buf, np.asarray(starts_abs) - self.base, self.W, FB_TONES, self.fs)

    def _score(self, starts):
        s = np.zeros(len(starts))
        for j, k in enumerate(self.scheme.fb_preamble):
            e = self._e(np.asarray(starts) + j * self.T + self.G)
            s += e[:, k] / (e.sum(axis=1) + 1e-12)
        return s

    def feed(self, x) -> list[Feedback]:
        self.buf = np.concatenate([self.buf, np.asarray(x, dtype=np.float32)])
        out = []
        while True:
            end = self.base + len(self.buf)
            last = end - (2 + SHORT_LEN + 1) * self.T
            if self.search > last:
                break
            starts = np.arange(self.search, last + 1, self.hop)
            sc = self._score(starts)
            hits = np.nonzero(sc >= FB_THRESHOLD)[0]
            if len(hits) == 0:
                self.search = int(starts[-1] + self.hop)
                break
            k = int(hits[0])
            look = self.T // self.hop
            if k + look >= len(starts):
                self.search = int(starts[k])
                break
            c0 = int(starts[k + int(np.argmax(sc[k: k + look + 1]))])
            offs = np.arange(-self.hop, self.hop + 1, 8)
            start = int(round(c0 + plateau_center(offs, self._score(c0 + offs), 0.1)))
            first = self._e([start + 2 * self.T + self.G])[0]
            n = KEY_LEN if int(np.argmax(first)) == self.key_type_tone else SHORT_LEN
            if start + (2 + n + 1) * self.T > end:
                self.search = c0 - self.hop
                break
            pos = start + (2 + np.arange(n)) * self.T + self.G
            e = cancel_carryover(self._e(pos))
            srt = np.sort(e, axis=1)
            conf = 1.0 - srt[:, -2] / (srt[:, -1] + 1e-12)
            fb = decode_body(list(np.argmax(e, axis=1)), conf, self.scheme)
            if fb is None:
                self.search = c0 + self.T // 2
            else:
                out.append(fb)
                self.search = start + (2 + n) * self.T - self.T // 2
        keep = self.search - self.T - self.base
        if keep > 5 * self.fs:
            self.buf = self.buf[keep:]
            self.base += keep
        return out
