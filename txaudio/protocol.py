"""Sender and receiver state machines, driven chunk by chunk.

Both sides are full duplex: ``next_chunk(n)`` returns what to play and
``feed(mic)`` takes what the microphone heard.  The same objects run on the
live sound card, in WAV file mode and in the acoustic simulator.

Live (with feedback):
  1. Sender plays an intro bar, then a HELLO frame in the public scheme with
     an ephemeral X25519 key and a random salt, then short idle frames while
     it waits.  HELLO is repeated if no answer comes.
  2. Receiver answers with its own X25519 key in tones.  Both derive the
     session scheme (sync motifs, note mapping, arpeggio order, feedback
     tones, cipher key) and show the same 6-digit safety code.
  3. Sender plays every block once, in the session scheme.  Receiver ACKs
     each good block and NAKs each damaged one.
  4. Sender resends NAKed blocks, then anything never ACKed, then plays idle
     frames, which prompt the receiver to NAK whatever is still missing.
  5. Receiver says DONE once it has every block; sender plays an outro.

File mode (no feedback): HELLO carries only the salt; the session is
derived from the salt and the optional passphrase, and every block is sent
``rounds`` times.
"""

from __future__ import annotations

import os
import zlib
from collections import deque
from typing import Optional

import numpy as np

from . import feedback as fbk
from .framing import (HELLO_SALT_ONLY, HELLO_X25519, KIND_DATA, KIND_HELLO, KIND_IDLE, Codec,
                      Header, Hello, byte_to_notes)
from .modem import DataDemod
from .music import Song
from .params import Params
from .scheme import KeyPair, public_scheme, session_scheme

INTRO_STEPS = 16
HELLO_WAIT_S = 10.0  # idle time after a HELLO before repeating it
MAX_HELLOS = 8


class Sender:
    def __init__(self, data: bytes, p: Params = Params(), passphrase: Optional[str] = None,
                 use_feedback: bool = True, rounds: int = 1, max_idle: int = 6,
                 log=None, seed: Optional[int] = None):
        if len(data) == 0:
            raise ValueError("empty file")
        self.p = p
        self.data = data
        self.passphrase = passphrase
        rnd = os.urandom(32) if seed is None else bytes((seed * 7 + i) & 0xFF for i in range(32))
        self.sess_byte = rnd[0]
        self.salt = rnd[1:17]
        self.codec = Codec(p)
        self.song = Song(p, int.from_bytes(rnd[17:21], "big"))
        self.blocks = [data[i: i + p.payload] for i in range(0, len(data), p.payload)]
        self.total = len(self.blocks)
        if self.total > 0xFFF:
            raise ValueError(f"file too large: {self.total} blocks (max 4095)")
        self.use_feedback = use_feedback
        self.rounds = rounds
        self.max_idle = max_idle
        self.log = log or (lambda *a: None)

        self.public = public_scheme(passphrase)
        self.session = None
        self.safety_code = ""
        if use_feedback:
            self.keys = KeyPair()
            self.hello = Hello(HELLO_X25519, self.salt, self.keys.public)
            self.fb_demod = fbk.FeedbackDemod(self.public, p.fs)
        else:
            self.keys = None
            self.hello = Hello(HELLO_SALT_ONLY, self.salt, b"")
            self._set_session(session_scheme(self.salt, passphrase))
            self.fb_demod = None
        self.hellos_sent = 0
        self.hello_due = 0  # sample time at which to (re)send HELLO
        self.handshake_s: Optional[float] = None

        self.acked: set[int] = set()
        self.naks: deque[int] = deque()
        self.last_sent: dict[int, int] = {}
        self.order = deque()
        self.round = 0
        self.done = False
        self.failed = False
        self.finished = False  # all audio, including outro, has been emitted
        self.idle_run = 0
        self.frames_sent = 0
        self.seq = 0  # session frame counter (selects the sync motif)
        self.step = 0  # global musical step counter
        self.t = 0  # samples emitted
        self.out = np.zeros(0, dtype=np.float32)
        self.tail = np.zeros(0, dtype=np.float32)
        self.ending = False
        self.note_log: list = []  # every data/sync step played, for tests and analysis
        self._append(self.song.render(0, [None] * INTRO_STEPS, self.public.slots), INTRO_STEPS)

    def _set_session(self, scheme):
        self.session = scheme
        self.safety_code = scheme.safety_code

    # -- audio scheduling -------------------------------------------------
    @property
    def now(self) -> int:
        """Sample time at which newly scheduled audio will start playing."""
        return self.t + len(self.out)

    def _append(self, audio: np.ndarray, steps: int):
        n = steps * self.p.step_n
        body = audio[:n].copy()
        k = min(len(self.tail), len(body))
        body[:k] += self.tail[:k]
        rest = self.tail[k:]
        tail = audio[n:].copy()
        if len(rest):
            t2 = np.zeros(max(len(rest), len(tail)), dtype=np.float32)
            t2[: len(rest)] += rest
            t2[: len(tail)] += tail
            tail = t2
        self.tail = tail
        self.out = np.concatenate([self.out, body])
        self.step += steps

    def _play_frame(self, scheme, hdr: Header, payload: Optional[bytes]):
        if scheme is self.public:
            seq, motif = 0, 0
        else:
            seq = self.seq
            motif = seq % len(scheme.motifs)
            self.seq += 1
        notes = [list(s) for s in scheme.motifs[motif]]
        cw = self.codec.encode_header(hdr, scheme, seq)
        if payload is not None:
            cw += self.codec.encode_payload(hdr, payload, scheme, seq)
        notes += [byte_to_notes(b) for b in cw]
        self.note_log.extend(notes)
        self._append(self.song.render(self.step, notes, scheme.slots), len(notes))

    def _next_round(self) -> bool:
        if self.round >= self.rounds:
            return False
        self.round += 1
        self.order.extend(range(self.total))
        return True

    def _choose(self):
        """Pick the next block to send, or None for an idle frame."""
        while self.naks:
            b = self.naks.popleft()
            if b not in self.acked:
                return b
        while self.order:
            b = self.order.popleft()
            if b not in self.acked:
                return b
        if not self.use_feedback:
            return None
        # Resend anything never acknowledged after a grace period.
        grace = (self.p.data_frame_steps + 40) * self.p.step_n
        stale = [b for b in range(self.total)
                 if b not in self.acked and self.now - self.last_sent.get(b, -10**12) > grace]
        if stale:
            return min(stale, key=lambda b: self.last_sent.get(b, -1))
        return None

    def _end(self):
        self.ending = True
        self.out = np.concatenate([self.out, self.tail, self.song.outro(self.step)])
        self.tail = np.zeros(0, dtype=np.float32)

    def _schedule(self):
        if self.ending:
            return
        if self.done or self.idle_run >= self.max_idle:
            self._end()
            return
        p = self.p
        # Handshake (live) or salt announcement (file mode).
        if self.use_feedback and self.session is None:
            if self.now >= self.hello_due:
                if self.hellos_sent >= MAX_HELLOS:
                    self.log("tx no key from receiver, giving up")
                    self.failed = True
                    self._end()
                    return
                self.log("tx HELLO")
                self._play_hello()
                self.hello_due = self.now + int(HELLO_WAIT_S * p.fs)
            else:
                self._play_frame(self.public, Header(KIND_IDLE, self.sess_byte, 0, 1, 0), None)
            return
        if not self.use_feedback and not self.order:
            if self._next_round():
                self.log(f"tx HELLO (round {self.round})")
                self._play_hello()
                return
            self.idle_run += 1
            if self.idle_run > 1:
                self._end()
                return
        if self.use_feedback and self.round == 0:
            self._next_round()
        b = self._choose()
        if b is None:
            hdr = Header(KIND_IDLE, self.sess_byte, 0, self.total, 0)
            if self.use_feedback and len(self.acked) == self.total:
                self.idle_run += 1
            self.log(f"tx idle frame ({len(self.acked)}/{self.total} acked)")
            self._play_frame(self.session, hdr, None)
        else:
            blk = self.blocks[b]
            hdr = Header(KIND_DATA, self.sess_byte, b, self.total, len(blk))
            self.last_sent[b] = self.now
            self.frames_sent += 1
            self.log(f"tx block {b}")
            self._play_frame(self.session, hdr, blk)

    def _play_hello(self):
        # Random padding: zero padding under the fixed public keystream would
        # play the same notes every time.
        hp = self.hello.pack()
        hp += os.urandom(self.p.payload - len(hp))
        self.hellos_sent += 1
        self._play_frame(self.public, Header(KIND_HELLO, self.sess_byte, 0, 0, len(hp)), hp)

    def next_chunk(self, n: int) -> np.ndarray:
        while len(self.out) < n and not self.ending:
            self._schedule()
        chunk = self.out[:n]
        self.out = self.out[n:]
        if len(chunk) < n:
            chunk = np.concatenate([chunk, np.zeros(n - len(chunk), dtype=np.float32)])
            self.finished = True
        self.t += n
        return chunk

    def feed(self, mic: np.ndarray):
        if self.fb_demod is None:
            return
        for fb in self.fb_demod.feed(mic):
            self.log(f"tx heard {fb}")
            if fb.kind == fbk.KEY and self.session is None:
                shared = self.keys.shared(fb.key)
                self._set_session(session_scheme(self.salt, self.passphrase, shared,
                                                 self.keys.public, fb.key))
                self.fb_demod = fbk.FeedbackDemod(self.session, self.p.fs)
                self.handshake_s = self.t / self.p.fs
                self.log(f"tx session ready, safety code {self.safety_code}")
            elif fb.kind == fbk.ACK and fb.block < self.total:
                self.acked.add(fb.block)
            elif fb.kind == fbk.NAK and fb.block < self.total:
                if fb.block not in self.acked and fb.block not in self.naks:
                    self.naks.append(fb.block)
            elif fb.kind == fbk.DONE and fb.block == self.total & 0xFFF:
                self.done = True


class _RxSession:
    """One sender's transfer as seen by the receiver."""

    def __init__(self, hello: Hello, sess_byte: int, scheme, keys: Optional[KeyPair]):
        self.hello, self.sess_byte, self.scheme, self.keys = hello, sess_byte, scheme, keys
        self.total: Optional[int] = None
        self.blocks: dict[int, bytes] = {}
        self.last_heard = 0

    @property
    def complete(self) -> bool:
        return self.total is not None and len(self.blocks) == self.total

    def missing(self) -> list[int]:
        return [] if self.total is None else [i for i in range(self.total) if i not in self.blocks]


class Receiver:
    """Listens for HELLOs and frames; answers the session it is talking to.

    Several senders may be audible at once (another tx-audio in the room).
    Every HELLO opens its own session and frames are routed by scheme, so
    they do not get mixed up.  Feedback goes only to the active session: the
    first one heard, until it has been silent for ``stale_s``.
    """

    MAX_SESSIONS = 4

    def __init__(self, p: Params = Params(), passphrase: Optional[str] = None,
                 send_feedback: bool = True, log=None, stale_s: float = 60.0):
        self.p = p
        self.passphrase = passphrase
        self.public = public_scheme(passphrase)
        self.demod = DataDemod(p, [self.public])
        self.send_feedback = send_feedback
        self.log = log or (lambda *a: None)
        self.stale = int(stale_s * p.fs)
        self.sessions: list[_RxSession] = []
        self.active: Optional[_RxSession] = None
        self.queue: deque = deque()
        self.out = np.zeros(0, dtype=np.float32)
        self.t_out = 0
        self.key_sent_until = -1
        self.done_sent = 0
        self.events = []

    # -- views of the active (or first complete) session --------------------
    def _best(self) -> Optional[_RxSession]:
        for s in self.sessions:
            if s.complete:
                return s
        return self.active

    @property
    def safety_code(self) -> str:
        return self.active.scheme.safety_code if self.active else ""

    @property
    def total(self) -> Optional[int]:
        s = self._best()
        return s.total if s else None

    @property
    def blocks(self) -> dict:
        s = self._best()
        return s.blocks if s else {}

    @property
    def complete(self) -> bool:
        s = self._best()
        return bool(s and s.complete)

    def missing(self) -> list[int]:
        s = self._best()
        return s.missing() if s else []

    def result(self) -> Optional[bytes]:
        s = self._best()
        if not (s and s.complete):
            return None
        return b"".join(s.blocks[i] for i in range(s.total))

    # -- input ---------------------------------------------------------------
    def _say(self, fb: fbk.Feedback, scheme):
        if self.send_feedback and all(f != fb or sc is not scheme for f, sc in self.queue):
            self.queue.append((fb, scheme))

    def feed(self, mic: np.ndarray):
        for ev in self.demod.feed(mic):
            self._handle(ev)

    def flush(self):
        for ev in self.demod.flush():
            self._handle(ev)

    def _on_hello(self, ev):
        hello = Hello.unpack(ev.payload or b"")
        if hello is None:
            return
        for s in self.sessions:
            if s.hello.pack() == hello.pack():
                s.last_heard = ev.pos
                if (s is self.active and hello.mode == HELLO_X25519
                        and ev.pos > self.key_sent_until):
                    self.log("rx HELLO again, resending key")
                    self._say(fbk.Feedback(fbk.KEY, 0, s.keys.public), self.public)
                return
        stale = self.active is None or ev.pos - self.active.last_heard > self.stale
        keys = None
        if hello.mode == HELLO_X25519:
            if not self.send_feedback or not stale:
                self.log("rx ignoring a second sender's HELLO")
                return
            keys = KeyPair()
            scheme = session_scheme(hello.salt, self.passphrase, keys.shared(hello.spub),
                                    hello.spub, keys.public)
        else:
            scheme = session_scheme(hello.salt, self.passphrase)
        sess = _RxSession(hello, ev.header.session, scheme, keys)
        sess.last_heard = ev.pos
        self.sessions = (self.sessions + [sess])[-self.MAX_SESSIONS:]
        if stale:
            self.active = sess
        self.demod.schemes = [self.public] + [s.scheme for s in self.sessions]
        if keys is not None:
            self._say(fbk.Feedback(fbk.KEY, 0, keys.public), self.public)
        self.log(f"rx HELLO, session ready, safety code {scheme.safety_code}")

    def _handle(self, ev):
        h = ev.header
        if ev.scheme is self.public:
            if h.kind == KIND_HELLO and ev.payload is not None:
                self._on_hello(ev)
            return
        sess = next((s for s in self.sessions if s.scheme is ev.scheme), None)
        if sess is None or h.session != sess.sess_byte:
            return
        sess.last_heard = ev.pos
        talk = sess is self.active
        self.events.append(ev)
        tag = "" if talk else " (other sender)"
        if h.kind == KIND_DATA:
            if ev.payload is None:
                self.log(f"rx block {h.idx} FAILED{tag}")
                if talk:
                    self._say(fbk.Feedback(fbk.NAK, h.idx), sess.scheme)
                return
            if sess.total is None:
                sess.total = h.total
                self.log(f"rx file has {h.total} blocks{tag}")
            new = h.idx not in sess.blocks
            sess.blocks[h.idx] = ev.payload
            self.log(f"rx block {h.idx} ok{'' if new else ' (dup)'} "
                     f"[{len(sess.blocks)}/{sess.total}]{tag}")
            if talk:
                self._say(fbk.Feedback(fbk.ACK, h.idx), sess.scheme)
                if sess.complete and new:
                    self._say(fbk.Feedback(fbk.DONE, sess.total & 0xFFF), sess.scheme)
        elif h.kind == KIND_IDLE:
            if sess.total is None:
                sess.total = h.total
            if not talk:
                return
            if sess.complete:
                self._say(fbk.Feedback(fbk.DONE, sess.total & 0xFFF), sess.scheme)
            else:
                for b in sess.missing()[:3]:
                    self._say(fbk.Feedback(fbk.NAK, b), sess.scheme)

    # -- output --------------------------------------------------------------
    def next_chunk(self, n: int) -> np.ndarray:
        while len(self.out) < n and self.queue:
            fb, scheme = self.queue.popleft()
            self.log(f"rx says {fb}")
            if fb.kind == fbk.DONE:
                self.done_sent += 1
            gap = np.zeros(int(0.03 * self.p.fs), dtype=np.float32)
            self.out = np.concatenate([self.out, fbk.synth(fb, scheme, self.p.fs), gap])
            if fb.kind == fbk.KEY:
                self.key_sent_until = self.t_out + len(self.out)
        chunk = self.out[:n]
        self.out = self.out[n:]
        if len(chunk) < n:
            chunk = np.concatenate([chunk, np.zeros(n - len(chunk), dtype=np.float32)])
        self.t_out += n
        return chunk


def file_crc(data: bytes) -> str:
    return f"{zlib.crc32(data):08x}"
