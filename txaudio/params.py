"""Shared constants and tunable parameters."""

from __future__ import annotations

from dataclasses import dataclass

FS = 48000


def _note(semitones_from_a4: int) -> float:
    return 440.0 * 2.0 ** (semitones_from_a4 / 12.0)


# Data notes: three octaves of C major / A minor pentatonic, A4 .. A7.
# Any combination of pentatonic notes sounds consonant, which is what lets
# random data still sound like music.
_PENTA = [0, 3, 5, 7, 10, 12, 15, 17, 19, 22, 24, 27, 29, 31, 34, 36]
DATA_NOTES = [_note(s) for s in _PENTA]

VOICES = 4  # four registers, each owns four adjacent pentatonic notes
NOTES_PER_VOICE = 4  # -> 2 bits per note, 8 bits (one byte) per step
VOICE_FREQS = [DATA_NOTES[v * 4:(v + 1) * 4] for v in range(VOICES)]

# Sync motif: note index per (step, voice).  Four steps = 16 notes.
SYNC = [
    [3, 2, 1, 0],
    [0, 1, 2, 3],
    [3, 1, 2, 0],
    [0, 2, 1, 3],
]
SYNC_STEPS = len(SYNC)

# Feedback (receiver -> sender) tones, 16-FSK.  Placed in gaps that avoid the
# 2nd..6th harmonics of the data notes, so speaker distortion of the music
# does not land on them.
FB_TONES = [4460, 4550, 4640, 4790, 4880, 4970, 5060, 5150,
            5360, 5450, 5540, 5630, 5720, 5960, 6050, 6140]
FB_SYMBOL = 0.060  # seconds per feedback symbol
FB_WINDOW = 0.040
FB_GUARD = 0.015  # skip early echoes of the previous tone
FB_PREAMBLE = [1, 14]

GUARD_FRAC = 0.22  # analysis window starts this far into each note (skips early echoes)
WIN_FRAC = 0.55
HEADER_DATA = 7  # kind, session, idx(2), total(2), length
TAG_LEN = 16  # AEAD tag (or keyed MAC on public frames)


@dataclass(frozen=True)
class Params:
    """Modem parameters that sender and receiver must agree on."""

    step: float = 0.100  # seconds for one 4-note arpeggio cycle (one byte)
    payload: int = 64  # file bytes per block
    nsym_header: int = 6  # Reed-Solomon parity bytes for the header
    nsym_payload: int = 16  # Reed-Solomon parity bytes for the payload
    fs: int = FS

    def __post_init__(self):
        n = round(self.step * self.fs)
        if n % 16:
            raise ValueError("step must be a multiple of 16 samples")

    @property
    def step_n(self) -> int:
        return round(self.step * self.fs)

    @property
    def quarter_n(self) -> int:
        return self.step_n // 4

    @property
    def win_n(self) -> int:
        return int(round(WIN_FRAC * self.step_n))

    @property
    def guard_n(self) -> int:
        return int(round(GUARD_FRAC * self.step_n))

    @property
    def header_steps(self) -> int:
        return HEADER_DATA + self.nsym_header

    @property
    def payload_steps(self) -> int:
        return self.payload + TAG_LEN + self.nsym_payload

    @property
    def data_frame_steps(self) -> int:
        return SYNC_STEPS + self.header_steps + self.payload_steps

    @property
    def idle_frame_steps(self) -> int:
        return SYNC_STEPS + self.header_steps

    def net_bps(self) -> float:
        return self.payload * 8 / (self.data_frame_steps * self.step)
