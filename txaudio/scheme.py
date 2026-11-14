"""Per-session encoding schemes and the key material behind them.

Nothing about how a send sounds or is encoded is fixed except the opening
"public" scheme used for the handshake, and even that is hidden when both
sides share a passphrase.

A Scheme decides:
  * the sync motifs that mark frame starts (16 per session, rotating),
  * the order the four voices play in each arpeggio,
  * a keystream XORed onto every coded byte, which re-maps data to notes
    differently on every step of every frame,
  * the order and preamble of the 16 feedback tones,
  * the cipher for block contents (ChaCha20-Poly1305 after a key exchange).

Key agreement: the sender puts an ephemeral X25519 public key and a random
salt in its HELLO frame; the receiver answers with its own public key in
tones.  Both run HKDF over the shared secret, the salt, both public keys and
(optionally) a scrypt-stretched passphrase.  WAV file mode has no return
channel, so it derives the session from the salt and the passphrase only.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives import serialization

from .params import FB_PREAMBLE, SYNC, SYNC_STEPS, TAG_LEN, VOICES

PUBLIC_MOTIFS = 1
SESSION_MOTIFS = 16


class _Stream:
    """Deterministic integers from a keyed BLAKE2b counter stream."""

    def __init__(self, key: bytes, label: bytes):
        self.key, self.label, self.ctr, self.buf = key, label, 0, b""

    def byte(self) -> int:
        if not self.buf:
            self.buf = hashlib.blake2b(self.label + self.ctr.to_bytes(4, "big"),
                                       key=self.key, digest_size=64).digest()
            self.ctr += 1
        b, self.buf = self.buf[0], self.buf[1:]
        return b

    def below(self, n: int) -> int:
        lim = 256 - 256 % n
        while True:
            b = self.byte()
            if b < lim:
                return b % n

    def perm(self, n: int) -> list[int]:
        p = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.below(i + 1)
            p[i], p[j] = p[j], p[i]
        return p


def hkdf(ikm: bytes, salt: bytes, info: bytes, n: int) -> bytes:
    prk = hmac.new(salt or bytes(32), ikm, hashlib.sha256).digest()
    out, t, i = b"", b"", 1
    while len(out) < n:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:n]


def stretch_passphrase(passphrase: Optional[str]) -> bytes:
    if not passphrase:
        return b""
    return hashlib.scrypt(passphrase.encode(), salt=b"tx-audio/passphrase/v1",
                          n=2 ** 14, r=8, p=1, dklen=32)


@dataclass
class Scheme:
    name: str
    key: bytes
    motifs: list  # [motif][step][voice] -> note index
    slots: list  # slots[v] = arpeggio position (0..3) of voice v
    fb_perm: list  # symbol value -> tone index
    fb_preamble: list
    tag: bytes
    aead_key: Optional[bytes] = None
    safety_code: str = ""

    # -- keystream ---------------------------------------------------------
    def stream(self, label: bytes, seq: int, n: int) -> bytes:
        out = b""
        ctr = 0
        while len(out) < n:
            out += hashlib.blake2b(label + seq.to_bytes(8, "big") + ctr.to_bytes(4, "big"),
                                   key=self.key, digest_size=64).digest()
            ctr += 1
        return out[:n]

    # -- block protection --------------------------------------------------
    def _nonce(self, idx: int) -> bytes:
        return b"txa1" + idx.to_bytes(8, "big")

    def seal(self, ad: bytes, idx: int, plain: bytes) -> bytes:
        if self.aead_key is not None:
            return ChaCha20Poly1305(self.aead_key).encrypt(self._nonce(idx), plain, ad)
        mac = hashlib.blake2b(ad + plain, key=self.key, digest_size=TAG_LEN).digest()
        return plain + mac

    def open(self, ad: bytes, idx: int, sealed: bytes) -> Optional[bytes]:
        if self.aead_key is not None:
            try:
                return ChaCha20Poly1305(self.aead_key).decrypt(self._nonce(idx), sealed, ad)
            except InvalidTag:
                return None
        plain, mac = sealed[:-TAG_LEN], sealed[-TAG_LEN:]
        good = hashlib.blake2b(ad + plain, key=self.key, digest_size=TAG_LEN).digest()
        return plain if hmac.compare_digest(mac, good) else None


def _build(name: str, key: bytes, n_motifs: int, keyed_layout: bool,
           aead_key: Optional[bytes] = None, safety_code: str = "") -> Scheme:
    if not keyed_layout:
        return Scheme(name, key, [SYNC], list(range(VOICES)), list(range(16)),
                      list(FB_PREAMBLE), key[:2], aead_key, safety_code)
    r = _Stream(key, b"layout")
    motifs = [[[r.below(4) for _ in range(VOICES)] for _ in range(SYNC_STEPS)]
              for _ in range(n_motifs)]
    slots = r.perm(VOICES)
    fb_perm = r.perm(16)
    pre = r.perm(16)[:2]
    return Scheme(name, key, motifs, slots, fb_perm, pre, key[:2], aead_key, safety_code)


def public_scheme(passphrase: Optional[str] = None) -> Scheme:
    """The handshake scheme.  Fixed and recognisable unless a passphrase is set."""
    pk = stretch_passphrase(passphrase)
    if not pk:
        return _build("public", hashlib.blake2b(b"tx-audio/public/v1", digest_size=32).digest(),
                      PUBLIC_MOTIFS, keyed_layout=False)
    return _build("public+pass", hkdf(pk, b"", b"tx-audio/public", 32), PUBLIC_MOTIFS, True)


def session_scheme(salt: bytes, passphrase: Optional[str] = None, shared: bytes = b"",
                   spub: bytes = b"", rpub: bytes = b"") -> Scheme:
    pk = stretch_passphrase(passphrase)
    okm = hkdf(shared, salt, b"tx-audio/session/v1|" + pk + spub + rpub, 72)
    code = f"{int.from_bytes(okm[64:68], 'big') % 1_000_000:06d}"
    aead = okm[32:64] if (shared or pk) else None
    return _build("session", okm[:32], SESSION_MOTIFS, True, aead_key=aead, safety_code=code)


class KeyPair:
    def __init__(self):
        self._priv = X25519PrivateKey.generate()
        self.public = self._priv.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def shared(self, peer_public: bytes) -> bytes:
        return self._priv.exchange(X25519PublicKey.from_public_bytes(peer_public))
