"""Streaming demodulator for the music data channel."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from .dsp import tone_energy
from .framing import KIND_DATA, KIND_HELLO, Codec, Header, notes_to_byte
from .params import SYNC, SYNC_STEPS, VOICE_FREQS, VOICES, Params

SYNC_THRESHOLD = 10.5  # out of 16; random music averages ~4


@dataclass
class FrameEvent:
    pos: int  # absolute sample index of the frame start
    header: Header
    payload: Optional[bytes]  # None if a data frame failed to decode
    scheme: object = None
    seq: int = 0
    header_conf: float = 0.0


def plateau_center(offsets: np.ndarray, scores: np.ndarray, tol: float) -> float:
    k = int(np.argmax(scores))
    lim = scores[k] - tol
    lo = k
    while lo > 0 and scores[lo - 1] >= lim:
        lo -= 1
    hi = k
    while hi < len(scores) - 1 and scores[hi + 1] >= lim:
        hi += 1
    return 0.5 * (offsets[lo] + offsets[hi])


class DataDemod:
    """Finds frames of any of the given schemes in a stream and decodes them."""

    def __init__(self, p: Params, schemes=(), codec: Optional[Codec] = None):
        self.p = p
        self.codec = codec or Codec(p)
        self.schemes = list(schemes)
        self.last_seq: dict[int, int] = {}
        self.buf = np.zeros(0, dtype=np.float32)
        self.base = 0  # absolute index of buf[0]
        self.search = 0  # next absolute frame-start candidate to scan
        self.pending = None
        self.prev = None  # (start, steps) of the last decoded frame
        self.drift = 0.0
        self.hop = p.step_n // 16
        self._gk0 = 0  # grid cache: tone ratios per voice at k*hop + guard
        self._grid = [np.zeros((0, 4), dtype=np.float32) for _ in range(VOICES)]
        self.stats = {"sync": 0, "header_fail": 0, "payload_fail": 0, "ok": 0}

    def add_scheme(self, scheme) -> None:
        if all(s is not scheme for s in self.schemes):
            self.schemes.append(scheme)

    # -- low level ------------------------------------------------------
    def _energies(self, starts_abs: np.ndarray, voice: int) -> np.ndarray:
        p = self.p
        return tone_energy(self.buf, np.asarray(starts_abs) - self.base, p.win_n,
                           VOICE_FREQS[voice], p.fs)

    def _voice_off(self, scheme, v: int) -> int:
        return scheme.slots[v] * self.p.quarter_n

    def sync_scores(self, starts: np.ndarray, scheme, rate: float = 1.0) -> np.ndarray:
        """Score of every motif of ``scheme`` at each start: shape (motifs, n)."""
        p = self.p
        starts = np.asarray(starts, dtype=np.int64)
        motifs = np.asarray(scheme.motifs)  # (m, step, voice)
        score = np.zeros((len(motifs), len(starts)))
        for j in range(SYNC_STEPS):
            for v in range(VOICES):
                off = int(round((j * p.step_n + self._voice_off(scheme, v)) * rate)) + p.guard_n
                e = self._energies(starts + off, v)
                r = e / (e.sum(axis=1, keepdims=True) + 1e-12)
                score += r[:, motifs[:, j, v]].T
        return score

    def _grid_ratios(self, k0: int, k1: int) -> list:
        """Per-voice tone-energy ratios on the coarse grid, cached.

        Grid point k is a window starting at k*hop + guard.  Sync motifs,
        voice slots and steps are all whole multiples of the hop, so one grid
        serves every scheme and motif.
        """
        have_end = self._gk0 + len(self._grid[0])
        if k0 < self._gk0 or k0 > have_end:
            self._gk0, have_end = k0, k0
            self._grid = [g[:0] for g in self._grid]
        if k1 > have_end:
            ks = np.arange(have_end, k1)
            pos = ks * self.hop + self.p.guard_n
            for v in range(VOICES):
                e = self._energies(pos, v)
                r = e / (e.sum(axis=1, keepdims=True) + 1e-12)
                self._grid[v] = np.concatenate([self._grid[v], r.astype(np.float32)])
        # Drop grid points far behind the search position.
        drop = min(k0, self.search // self.hop - 64) - self._gk0
        if drop > 4096:
            self._grid = [g[drop:] for g in self._grid]
            self._gk0 += drop
        return [g[k0 - self._gk0: k1 - self._gk0] for g in self._grid]

    def grid_scores(self, k0: int, n: int, scheme) -> np.ndarray:
        """Motif scores at grid starts k0..k0+n-1: shape (motifs, n)."""
        per_step = self.p.step_n // self.hop
        per_slot = self.p.quarter_n // self.hop
        reach = (SYNC_STEPS - 1) * per_step + 3 * per_slot + 1
        R = self._grid_ratios(k0, k0 + n + reach)
        motifs = np.asarray(scheme.motifs)
        score = np.zeros((len(motifs), n))
        for j in range(SYNC_STEPS):
            for v in range(VOICES):
                o = j * per_step + scheme.slots[v] * per_slot
                score += R[v][o: o + n][:, motifs[:, j, v]].T
        return score

    def _refine(self, coarse: int, scheme, motif: int) -> int:
        span = self.p.quarter_n
        offs = np.arange(-span, span + 1, 16)
        sc = self.sync_scores(coarse + offs, scheme)[motif]
        c = plateau_center(offs, sc, 0.3)
        return int(round(coarse + c - sync_bias(self.p)))

    def demod_steps(self, start: int, first_step: int, n_steps: int, rate: float, scheme):
        """Note decisions and per-byte confidence for n_steps steps.

        Room reverb leaves the previous note of each voice ringing into the
        next window.  A first pass makes decisions, estimates how much energy
        carries over from one step to the next, and a second pass subtracts
        that carry-over before deciding again.
        """
        p = self.p
        steps = np.arange(first_step - 1, first_step + n_steps)
        notes = np.zeros((n_steps, VOICES), dtype=np.int64)
        conf = np.ones(n_steps)
        for v in range(VOICES):
            pos = start + np.round((steps * p.step_n + self._voice_off(scheme, v)) * rate
                                   ).astype(np.int64) + p.guard_n
            e_all = self._energies(pos, v).astype(np.float64)
            prev, e = e_all[:-1], e_all[1:]
            d = np.argmax(e_all, axis=1)
            changed = d[1:] != d[:-1]
            rows = np.nonzero(changed)[0]
            c = 0.0
            if len(rows) >= 4:
                k = d[:-1][rows]
                c = float(np.median(e[rows, k] / (prev[rows, k] + 1e-12)))
                c = min(max(c, 0.0), 0.5)
            ec = np.maximum(e - c * prev, 0.0) if c > 0.02 else e
            srt = np.sort(ec, axis=1)
            notes[:, v] = np.argmax(ec, axis=1)
            conf = np.minimum(conf, 1.0 - srt[:, -2] / (srt[:, -1] + 1e-12))
        data = bytes(notes_to_byte(n) for n in notes)
        return data, conf

    def _seq_for(self, scheme, motif: int) -> int:
        n = len(scheme.motifs)
        if n == 1:
            return 0
        nxt = self.last_seq.get(id(scheme), -1) + 1
        return nxt + ((motif - nxt) % n)

    # -- streaming -------------------------------------------------------
    def feed(self, x: np.ndarray) -> list[FrameEvent]:
        self.buf = np.concatenate([self.buf, np.asarray(x, dtype=np.float32)])
        events = []
        while True:
            ev, progressed = self._step()
            if ev is not None:
                events.append(ev)
            if not progressed:
                break
        self._trim()
        return events

    def flush(self) -> list[FrameEvent]:
        """Call at end of input: pad with silence so trailing frames decode."""
        return self.feed(np.zeros(self.p.step_n * 4, dtype=np.float32))

    @property
    def end(self) -> int:
        return self.base + len(self.buf)

    def _rate(self) -> float:
        return 1.0 + self.drift

    def _step(self):
        p = self.p
        sn = p.step_n
        if self.pending is not None:
            start, hdr, rate, hconf, scheme, seq = self.pending
            need = start + int((p.data_frame_steps + 1) * sn * rate) + sn
            if self.end < need:
                return None, False
            raw, conf = self.demod_steps(start, SYNC_STEPS + p.header_steps, p.payload_steps,
                                         rate, scheme)
            payload = self.codec.decode_payload(hdr, raw, conf, scheme, seq)
            self.stats["ok" if payload is not None else "payload_fail"] += 1
            self.pending = None
            self._accept(start, p.data_frame_steps)
            return FrameEvent(start, hdr, payload, scheme, seq, hconf), True

        if not self.schemes:
            self.search = max(self.search, self.end - 8 * sn)
            return None, False
        span = SYNC_STEPS * sn + p.quarter_n * 2
        last = self.end - span
        if self.search > last:
            return None, False
        k0 = -(-self.search // self.hop)
        starts = np.arange(k0 * self.hop, last + 1, self.hop)
        if len(starts) == 0:
            return None, False
        per = [self.grid_scores(k0, len(starts), sch) for sch in self.schemes]
        best = np.max([sc.max(axis=0) for sc in per], axis=0)
        hits = np.nonzero(best >= SYNC_THRESHOLD)[0]
        if len(hits) == 0:
            self.search = int(starts[-1] + self.hop)
            return None, False
        k = int(hits[0])
        # Need the peak neighbourhood and the header to be in the buffer.
        lookahead = (p.quarter_n * 2) // self.hop
        if k + lookahead >= len(starts):
            self.search = int(starts[k])
            return None, False
        k = k + int(np.argmax(best[k: k + lookahead + 1]))
        rate = self._rate()
        if self.end < starts[k] + int((SYNC_STEPS + p.header_steps + 1) * sn * rate) + sn:
            self.search = int(starts[max(0, k - lookahead)])
            return None, False
        si = int(np.argmax([sc[:, k].max() for sc in per]))
        scheme = self.schemes[si]
        motif = int(np.argmax(per[si][:, k]))
        start = self._refine(int(starts[k]), scheme, motif)
        seq = self._seq_for(scheme, motif)
        self.stats["sync"] += 1
        raw, conf = self.demod_steps(start, SYNC_STEPS, p.header_steps, rate, scheme)
        hdr = self.codec.decode_header(raw, conf, scheme, seq)
        if hdr is None:
            self.stats["header_fail"] += 1
            self.search = int(starts[k] + sn // 2)
            return None, True
        if len(scheme.motifs) > 1:
            self.last_seq[id(scheme)] = seq
        hconf = float(np.mean(conf))
        if hdr.kind in (KIND_DATA, KIND_HELLO):
            self.pending = (start, hdr, rate, hconf, scheme, seq)
            return None, True
        self._accept(start, p.idle_frame_steps)
        return FrameEvent(start, hdr, None, scheme, seq, hconf), True

    def _accept(self, start: int, steps: int):
        sn = self.p.step_n
        if self.prev is not None:
            pstart, psteps = self.prev
            nominal = psteps * sn
            dt = start - pstart
            if abs(dt - nominal * (1 + self.drift)) < sn // 3:
                est = dt / nominal - 1.0
                self.drift = float(np.clip(0.5 * self.drift + 0.5 * est, -0.005, 0.005))
        self.prev = (start, steps)
        # Keep scanning inside this frame: another sender's frame may overlap it.
        self.search = start + sn

    def _trim(self):
        keep_from = self.search - 2 * self.p.step_n
        if self.pending is not None:
            keep_from = min(keep_from, self.pending[0] - self.p.step_n)
        cut = keep_from - self.base
        if cut > 10 * self.p.fs:
            self.buf = self.buf[cut:]
            self.base += cut


@lru_cache(maxsize=None)
def sync_bias(p: Params) -> float:
    """Offset between the sync plateau centre and the true frame start.

    Measured once per parameter set on a clean synthetic sync so the
    estimator lands windows where the synth puts notes.
    """
    from .music import note_template
    from .scheme import public_scheme

    sch = public_scheme()
    sn, q = p.step_n, p.quarter_n
    lead = 2 * sn
    x = np.zeros(lead + (SYNC_STEPS + 2) * sn, dtype=np.float32)
    for j in range(SYNC_STEPS):
        for v in range(VOICES):
            pos = lead + j * sn + sch.slots[v] * q
            x[pos: pos + sn] += note_template(VOICE_FREQS[v][SYNC[j][v]], sn, p.fs)
    d = DataDemod(p)
    d.buf = x
    offs = np.arange(-q, q + 1, 16)
    sc = d.sync_scores(lead + offs, sch)[0]
    return float(plateau_center(offs, sc, 0.3))
