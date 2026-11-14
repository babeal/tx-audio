import os

import numpy as np
import pytest

from txaudio.channel import Channel, ChannelConfig, Noise, simulate_duplex
from txaudio.dsp import read_wav, write_wav
from txaudio.params import Params
from txaudio.protocol import Receiver, Sender

P = Params()
ROOM = dict(echoes=[(9, 0.5), (23, 0.35), (41, 0.2)], reverb_rt60=0.5, reverb_level=0.4)


def encode(data, **kw):
    s = Sender(data, P, use_feedback=False, **kw)
    out = []
    while not s.finished:
        out.append(s.next_chunk(4800))
    return np.concatenate(out), s


def decode(x, **kw):
    r = Receiver(P, send_feedback=False, **kw)
    for i in range(0, len(x), 4800):
        r.feed(x[i: i + 4800])
    r.flush()
    return r


def through(x, cfg, noise_db=-50):
    ch, n = Channel(cfg), Noise(noise_db, 3)
    return np.concatenate([ch(x[i: i + 4800]) + n(len(x[i: i + 4800]))
                           for i in range(0, len(x), 4800)])


def test_file_mode_clean_roundtrip(tmp_path):
    data = os.urandom(100)
    x, _ = encode(data, rounds=1)
    path = str(tmp_path / "a.wav")
    write_wav(path, x, P.fs)
    assert decode(read_wav(path, P.fs)).result() == data


def test_file_mode_room_noise_drift_gain():
    data = os.urandom(150)
    x, _ = encode(data, rounds=1)
    cfg = ChannelConfig(gain=0.25, drift_ppm=-400, wobble_db=6, **ROOM)
    r = decode(through(x, cfg, noise_db=-30))
    assert r.result() == data


def test_same_file_never_sounds_the_same():
    data = b"\x00" * 200  # worst case: a file of zeros
    _, a = encode(data, rounds=1)
    _, b = encode(data, rounds=1)
    n = min(len(a.note_log), len(b.note_log))
    na, nb = np.array(a.note_log[:n]), np.array(b.note_log[:n])
    # Apart from the 4-step public sync that opens the HELLO, notes agree
    # only as often as chance (1 in 4).
    same = (na[4:] == nb[4:]).mean()
    assert 0.15 < same < 0.35
    # And a zero file does not repeat itself within one send either.
    assert len({tuple(map(tuple, a.note_log[i: i + 8])) for i in range(0, n - 8, 8)}) > 0.95 * (n // 8)


def test_wrong_passphrase_cannot_decode():
    data = os.urandom(60)
    x, _ = encode(data, rounds=1, passphrase="correct horse")
    assert decode(x, passphrase="correct horse").result() == data
    eve = decode(x, passphrase="wrong")
    assert eve.result() is None and not eve.blocks
    assert decode(x).result() is None  # no passphrase: can't even find the HELLO


def test_duplex_handshake_acks_and_overlap():
    data = os.urandom(120)
    res = simulate_duplex(data, P, forward=ChannelConfig(gain=0.3, drift_ppm=150, **ROOM),
                          noise_db=-45, record=True)
    assert res.ok, res.rx_log[-10:]
    assert res.safety_codes[0] == res.safety_codes[1] != ""
    assert any("DONE" in m for m in res.tx_log)
    # A passive eavesdropper with the full recording and the right code
    # still cannot decode: it never had either private key.
    eve = decode(res.rx_mic)
    assert eve.result() is None and not eve.blocks


def test_duplex_recovers_lost_frame_with_nak():
    data = os.urandom(3 * P.payload)
    # Smash the middle of block 1 (intro + hello + idles + block 0 ~ 35-40 s).
    fwd = ChannelConfig(gain=0.3, dropouts=[(43.0, 2.5)], **ROOM)
    res = simulate_duplex(data, P, forward=fwd, noise_db=-45)
    assert res.ok, res.rx_log
    assert res.frames_sent > res.blocks  # something was resent
    assert any("NAK" in m for m in res.tx_log)


@pytest.mark.slow
def test_duplex_harsh_room():
    data = os.urandom(4 * P.payload)
    fwd = ChannelConfig(gain=0.2, drift_ppm=-800, wobble_db=8, reverb_rt60=0.9,
                        reverb_level=0.7, echoes=ROOM["echoes"])
    res = simulate_duplex(data, P, forward=fwd, noise_db=-30)
    assert res.ok
