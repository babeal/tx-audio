import os

import numpy as np
import pytest

from txaudio import feedback as fbk
from txaudio.framing import KIND_DATA, Codec, Header, byte_to_notes, notes_to_byte
from txaudio.music import Song
from txaudio.params import Params
from txaudio.scheme import KeyPair, public_scheme, session_scheme

P = Params()


def test_byte_note_mapping_roundtrip():
    for b in range(256):
        assert notes_to_byte(byte_to_notes(b)) == b


def test_header_and_payload_roundtrip_with_errors():
    c = Codec(P)
    sch = session_scheme(b"s" * 16)
    h = Header(KIND_DATA, 9, 3, 10, 40)
    data = os.urandom(40)
    hc = bytearray(c.encode_header(h, sch, 5))
    pc = bytearray(c.encode_payload(h, data, sch, 5))
    hc[2] ^= 0x41  # header can fix 3 errors
    for i in (0, 7, 30, 50, 61, 70, 90):  # payload can fix 8 errors
        pc[i] ^= 0x5A
    conf = np.ones(len(pc))
    assert c.decode_header(bytes(hc), np.ones(len(hc)), sch, 5) == h
    assert c.decode_payload(h, bytes(pc), conf, sch, 5) == data


def test_erasures_extend_correction():
    c = Codec(P)
    sch = session_scheme(b"e" * 16)
    h = Header(KIND_DATA, 1, 0, 1, 64)
    data = os.urandom(64)
    pc = bytearray(c.encode_payload(h, data, sch, 0))
    bad = list(range(0, 28, 2))  # 14 errors: too many for errors-only (max 8)
    conf = np.ones(len(pc))
    for i in bad:
        pc[i] ^= 0xFF
        conf[i] = 0.05  # but the demodulator knew they were shaky
    assert c.decode_payload(h, bytes(pc), conf, sch, 0) == data


def test_tampering_and_wrong_scheme_rejected():
    c = Codec(P)
    a, b = session_scheme(b"a" * 16, shared=b"k" * 32), session_scheme(b"b" * 16, shared=b"k" * 32)
    h = Header(KIND_DATA, 1, 0, 1, 10)
    pc = c.encode_payload(h, b"0123456789", a, 0)
    assert c.decode_payload(h, pc, np.ones(len(pc)), b, 0) is None
    assert c.decode_payload(Header(KIND_DATA, 1, 0, 1, 9), pc, np.ones(len(pc)), a, 0) is None


def test_session_schemes_differ_and_agree():
    s1, s2 = session_scheme(b"x" * 16), session_scheme(b"y" * 16)
    assert s1.motifs != s2.motifs and s1.key != s2.key
    assert session_scheme(b"x" * 16).motifs == s1.motifs
    assert len(s1.motifs) == 16
    # Same codeword bytes land on different notes in different sessions / frames.
    assert s1.stream(b"pay", 0, 32) != s2.stream(b"pay", 0, 32)
    assert s1.stream(b"pay", 0, 32) != s1.stream(b"pay", 1, 32)


def test_x25519_agreement_and_passphrase():
    a, b = KeyPair(), KeyPair()
    salt = os.urandom(16)
    sa = session_scheme(salt, "pw", a.shared(b.public), a.public, b.public)
    sb = session_scheme(salt, "pw", b.shared(a.public), a.public, b.public)
    assert sa.key == sb.key and sa.aead_key == sb.aead_key and sa.safety_code == sb.safety_code
    sc = session_scheme(salt, "other", b.shared(a.public), a.public, b.public)
    assert sc.key != sa.key
    assert public_scheme("pw").motifs != public_scheme().motifs


def test_feedback_roundtrip_under_music():
    sch = session_scheme(b"f" * 16)
    pub = public_scheme()
    msgs = [(fbk.Feedback(fbk.ACK, 7), sch), (fbk.Feedback(fbk.NAK, 4095), sch),
            (fbk.Feedback(fbk.DONE, 12), sch), (fbk.Feedback(fbk.KEY, 0, os.urandom(32)), pub)]
    song = Song(P, 1)
    rng = np.random.default_rng(0)
    notes = [[int(v) for v in rng.integers(0, 4, 4)] for _ in range(120)]
    music = song.render(0, notes)
    for fb, s in msgs:
        tone = fbk.synth(fb, s)
        x = music[: len(tone) + 48000].copy() * 1.5  # sender hears its own music loudly
        x[20000: 20000 + len(tone)] += 0.2 * tone
        x += 0.003 * rng.standard_normal(len(x)).astype(np.float32)
        d = fbk.FeedbackDemod(s)
        got = []
        for i in range(0, len(x), 1000):
            got += d.feed(x[i: i + 1000])
        assert got == [fb]
        # A different session's demodulator hears nothing.
        other = fbk.FeedbackDemod(session_scheme(b"g" * 16) if s is sch else public_scheme("pw"))
        assert other.feed(x) == []


def test_music_varies_and_stays_in_range():
    rng = np.random.default_rng(1)
    notes = [[int(v) for v in rng.integers(0, 4, 4)] for _ in range(64)]
    a = Song(P, 1).render(0, notes)
    b = Song(P, 2).render(0, notes)
    assert np.max(np.abs(a)) < 0.99 and np.max(np.abs(b)) < 0.99
    assert not np.allclose(a, b)
    # Arpeggio order changes the audio too.
    c = Song(P, 1).render(0, notes, slots=(3, 2, 1, 0))
    assert not np.allclose(a, c)


def test_bad_step_rejected():
    with pytest.raises(ValueError):
        Params(step=0.1001)
