"""The synth: turns note choices into music.

Data rides on a four-voice pentatonic arpeggio (pure sine plucks, one note per
voice per step, voices staggered by a quarter step).  Around it a per-session
"song" adds a kick, hats, a bass line and a soft pad.  The accompaniment never
carries data and stays out of the data band (below ~350 Hz, hats above 9 kHz),
so it only has to sound good.  The session seed picks the chord progressions,
drum and bass patterns, and changes them every few bars.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .params import FS, VOICE_FREQS, Params

# Chords: (bass root Hz, pad voicing Hz).  Pad stays below 350 Hz.
CHORDS = {
    "Am": (110.00, (220.00, 261.63, 329.63)),
    "F": (87.31, (174.61, 220.00, 261.63)),
    "C": (130.81, (196.00, 261.63, 329.63)),
    "G": (98.00, (196.00, 246.94, 293.66)),
    "Dm": (73.42, (174.61, 220.00, 293.66)),
    "Em": (82.41, (196.00, 246.94, 329.63)),
}
PROGRESSIONS = [
    ["Am", "F", "C", "G"],
    ["Am", "G", "F", "G"],
    ["C", "G", "Am", "F"],
    ["F", "G", "Am", "Am"],
    ["Am", "C", "G", "F"],
    ["Dm", "F", "Am", "G"],
    ["Am", "Em", "F", "G"],
    ["C", "Am", "F", "G"],
]
# 16 steps per bar (4 beats of 4 steps).
KICKS = [
    [0, 4, 8, 12],
    [0, 6, 8, 14],
    [0, 7, 10],
    [0, 8, 11],
    [0, 4, 8, 11, 12],
]
HATS = [
    [2, 6, 10, 14],
    list(range(1, 16, 2)),
    [2, 3, 6, 10, 11, 14],
    list(range(16)),
]
BASS = [  # (step, length in steps, octave multiplier)
    [(0, 4, 1), (4, 4, 1), (8, 4, 1), (12, 4, 1)],
    [(0, 3, 1), (3, 3, 1), (6, 2, 2), (8, 3, 1), (11, 3, 1), (14, 2, 2)],
    [(0, 8, 1), (10, 2, 2), (12, 4, 1)],
    [(0, 2, 1), (2, 2, 2), (4, 2, 1), (6, 2, 2), (8, 2, 1), (10, 2, 2), (12, 2, 1), (14, 2, 2)],
]
BARS_PER_SECTION = 4

AMP_NOTE = 0.16
AMP_KICK = 0.32
AMP_BASS = 0.11
AMP_PAD = 0.025
AMP_HAT = 0.035


@lru_cache(maxsize=None)
def note_template(freq: float, n: int, fs: int = FS) -> np.ndarray:
    """A sine pluck that fits exactly in n samples."""
    t = np.arange(n) / fs
    att = max(1, int(0.003 * fs))
    rel = max(1, int(0.1 * n))
    env = np.exp(-t / (0.45 * n / fs))
    env[:att] *= np.linspace(0.0, 1.0, att)
    env[-rel:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, rel)))
    return (AMP_NOTE * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@lru_cache(maxsize=None)
def _kick(fs: int = FS) -> np.ndarray:
    n = int(0.28 * fs)
    t = np.arange(n) / fs
    f = 45 + 100 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / fs
    env = np.exp(-t / 0.11)
    env[: int(0.002 * fs)] *= np.linspace(0, 1, int(0.002 * fs))
    return (AMP_KICK * env * np.sin(ph)).astype(np.float32)


@lru_cache(maxsize=None)
def _hat(variant: int, fs: int = FS) -> np.ndarray:
    n = int(0.06 * fs)
    rng = np.random.default_rng(1000 + variant)
    spec = np.fft.rfft(rng.standard_normal(n * 2))
    freqs = np.fft.rfftfreq(n * 2, 1 / fs)
    spec[(freqs < 9000) | (freqs > 15000)] = 0
    x = np.fft.irfft(spec)[:n]
    x /= np.max(np.abs(x)) + 1e-9
    t = np.arange(n) / fs
    return (AMP_HAT * x * np.exp(-t / (0.012 + 0.006 * variant))).astype(np.float32)


@lru_cache(maxsize=None)
def _tone(freq: float, n: int, amp: float, att: float, harm2: float = 0.0, fs: int = FS) -> np.ndarray:
    t = np.arange(n) / fs
    a = max(1, int(att * fs))
    r = max(1, min(n // 2, int(0.03 * fs)))
    env = np.ones(n)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] *= np.linspace(1, 0, r)
    y = np.sin(2 * np.pi * freq * t) + harm2 * np.sin(4 * np.pi * freq * t)
    return (amp * env * y).astype(np.float32)


class Song:
    """Accompaniment and data-note rendering for one send session."""

    def __init__(self, p: Params, session: int):
        self.p = p
        self.session = session

    def _section(self, bar: int):
        rng = np.random.default_rng([self.session, bar // BARS_PER_SECTION])
        return (
            PROGRESSIONS[rng.integers(len(PROGRESSIONS))],
            KICKS[rng.integers(len(KICKS))],
            HATS[rng.integers(len(HATS))],
            BASS[rng.integers(len(BASS))],
            bool(rng.integers(4)),  # pad on/off
        )

    def render(self, start_step: int, notes: list, slots=(0, 1, 2, 3), tail_s: float = 0.5) -> np.ndarray:
        """Render len(notes) steps starting at global step ``start_step``.

        ``notes[i]`` is a list of 4 note indices (one per voice) or None for
        a step with no data notes.  Returns the audio plus a tail that the
        caller overlaps onto the next block.  ``slots[v]`` is the arpeggio
        position of voice v (a per-session permutation).
        """
        p = self.p
        sn, q = p.step_n, p.quarter_n
        n_steps = len(notes)
        out = np.zeros(n_steps * sn + int(tail_s * p.fs), dtype=np.float32)

        def add(pos, x):
            end = min(len(out), pos + len(x))
            if end > pos:
                out[pos:end] += x[: end - pos]

        for i, ns in enumerate(notes):
            g = start_step + i
            base = i * sn
            if ns is not None:
                for v, k in enumerate(ns):
                    add(base + slots[v] * q, note_template(VOICE_FREQS[v][k], sn, p.fs))
            bar, pos = divmod(g, 16)
            prog, kicks, hats, bass, pad = self._section(bar)
            chord = CHORDS[prog[bar % 4]]
            if pos in kicks:
                add(base, _kick(p.fs))
            if pos in hats:
                add(base + sn // 2, _hat(g % 3, p.fs))
            for (bpos, blen, mult) in bass:
                if bpos == pos:
                    add(base, _tone(chord[0] * mult, blen * sn, AMP_BASS, 0.005, 0.3, p.fs))
            if pos == 0 and pad:
                for f in chord[1]:
                    add(base, _tone(f, 16 * sn, AMP_PAD, 0.25, 0.0, p.fs))
        return out

    def outro(self, start_step: int) -> np.ndarray:
        """A closing chord that rings out (no data, no sync), different each song."""
        p = self.p
        rng = np.random.default_rng([self.session, 0xE0D])
        prog = self._section(start_step // 16)[0]
        bass, pad = CHORDS[prog[rng.integers(len(prog))]]
        n = int(2.5 * p.fs)
        out = np.zeros(n, dtype=np.float32)
        out[: len(_kick(p.fs))] += _kick(p.fs)
        for f in pad + (bass,):
            out += _tone(f, n, AMP_PAD * 2, 0.01, 0.0, p.fs)
        t = np.arange(n) / p.fs
        gap = rng.uniform(0.08, 0.2)
        for j, v in enumerate(rng.permutation(4)):
            pos = int(j * gap * p.fs)
            x = note_template(VOICE_FREQS[v][rng.integers(4)], int(1.5 * p.fs), p.fs)
            out[pos: pos + len(x)] += x[: n - pos]
        return out * np.exp(-t / 0.9).astype(np.float32)
