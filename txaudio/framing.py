"""Frame layout: Reed-Solomon coded header and payload, CRC32, whitening.

A frame on air is::

    SYNC (4 steps) | HEADER codeword | PAYLOAD codeword (data frames only)

Each step carries one byte (four voices x two bits).  The payload is sealed
(ChaCha20-Poly1305 after a key exchange, a keyed MAC otherwise) and both
codewords are XORed with the session keystream, so repetitive files still
produce varied music and no two sends map data to notes the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from reedsolo import ReedSolomonError, RSCodec

from .params import HEADER_DATA, Params

VERSION = 1
KIND_DATA = 0
KIND_IDLE = 1
KIND_HELLO = 2

HELLO_X25519 = 1  # live: receiver answers with its own key
HELLO_SALT_ONLY = 0  # file mode: session derived from salt (+ passphrase)


def _xor(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


@dataclass
class Header:
    kind: int
    session: int
    idx: int
    total: int
    length: int

    def pack(self) -> bytes:
        return bytes([
            (VERSION << 4) | self.kind, self.session & 0xFF,
            self.idx >> 8, self.idx & 0xFF,
            self.total >> 8, self.total & 0xFF,
            self.length,
        ])

    @classmethod
    def unpack(cls, b: bytes, payload_max: int) -> Optional["Header"]:
        if len(b) != HEADER_DATA or b[0] >> 4 != VERSION:
            return None
        kind = b[0] & 0x0F
        h = cls(kind, b[1], (b[2] << 8) | b[3], (b[4] << 8) | b[5], b[6])
        if kind == KIND_DATA:
            if h.total == 0 or h.idx >= h.total or not 0 < h.length <= payload_max:
                return None
        elif kind == KIND_IDLE:
            if h.total == 0 or h.idx or h.length:
                return None
        elif kind == KIND_HELLO:
            if h.idx or h.total or not 0 < h.length <= payload_max:
                return None
        else:
            return None
        return h


@dataclass
class Hello:
    mode: int
    salt: bytes  # 16 bytes
    spub: bytes  # 32 bytes, or empty in salt-only mode

    def pack(self) -> bytes:
        return bytes([self.mode]) + self.salt + self.spub

    @classmethod
    def unpack(cls, b: bytes) -> Optional["Hello"]:
        if len(b) < 17 or b[0] not in (HELLO_X25519, HELLO_SALT_ONLY):
            return None
        if b[0] == HELLO_X25519:
            if len(b) < 49:
                return None
            return cls(b[0], b[1:17], b[17:49])
        return cls(b[0], b[1:17], b"")


class Codec:
    """Turns headers and blocks into coded bytes for one scheme, and back.

    Every codeword is XORed with the scheme's keystream for the frame's
    sequence number, so the byte -> note mapping changes on every step.
    """

    def __init__(self, p: Params):
        self.p = p
        self.rs_h = RSCodec(p.nsym_header)
        self.rs_p = RSCodec(p.nsym_payload)
        self.corrected = 0  # bytes repaired by Reed-Solomon in payloads
        self.payload_bytes = 0

    # -- encode ---------------------------------------------------------
    def encode_header(self, h: Header, scheme, seq: int) -> bytes:
        cw = bytes(self.rs_h.encode(h.pack()))
        return _xor(cw, scheme.stream(b"hdr", seq, len(cw)))

    def encode_payload(self, h: Header, data: bytes, scheme, seq: int) -> bytes:
        p = self.p
        padded = data + bytes(p.payload - len(data))
        sealed = scheme.seal(h.pack(), h.idx, padded)
        cw = bytes(self.rs_p.encode(sealed))
        return _xor(cw, scheme.stream(b"pay", seq, len(cw)))

    # -- decode ---------------------------------------------------------
    @staticmethod
    def _rs_decode(rs: RSCodec, cw: bytes, conf: Sequence[float], nsym: int,
                   max_erasures: int):
        """Errors-only first, then mark the least confident bytes as erasures."""
        tries = [0] + sorted({max(1, nsym // 4), nsym // 2, (3 * nsym) // 4, nsym})
        tries = [t for t in tries if t <= max_erasures]
        order = sorted(range(len(cw)), key=lambda i: conf[i])
        for n_er in tries:
            try:
                msg, full, _ = rs.decode(bytearray(cw), erase_pos=order[:n_er] if n_er else None)
                return bytes(msg), sum(a != b for a, b in zip(full, cw))
            except ReedSolomonError:
                continue
        return None, 0

    def decode_header(self, raw: bytes, conf, scheme, seq: int) -> Optional[Header]:
        cw = _xor(raw, scheme.stream(b"hdr", seq, len(raw)))
        # Keep half the header parity as a check against false syncs.
        msg, _ = self._rs_decode(self.rs_h, cw, conf, self.p.nsym_header,
                                 self.p.nsym_header // 2)
        if msg is None:
            return None
        return Header.unpack(msg, self.p.payload)

    def decode_payload(self, h: Header, raw: bytes, conf, scheme, seq: int) -> Optional[bytes]:
        cw = _xor(raw, scheme.stream(b"pay", seq, len(raw)))
        msg, fixed = self._rs_decode(self.rs_p, cw, conf, self.p.nsym_payload,
                                     self.p.nsym_payload)
        if msg is None:
            return None
        self.corrected += fixed
        self.payload_bytes += len(cw)
        plain = scheme.open(h.pack(), h.idx, msg)
        if plain is None:
            return None
        return plain[:h.length]


def byte_to_notes(b: int) -> list[int]:
    return [(b >> (6 - 2 * v)) & 3 for v in range(4)]


def notes_to_byte(n: Sequence[int]) -> int:
    return (n[0] << 6) | (n[1] << 4) | (n[2] << 2) | n[3]
