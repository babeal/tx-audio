"""Robustness sweep: encode once, push it through simulated rooms, decode.

    python3 tools/sweep.py [--step 0.1] [--only "room 1.2,noise -15"]

Reports blocks recovered and the share of payload bytes Reed-Solomon had to
repair (how close each condition is to the cliff: 16 parity bytes per 96).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from txaudio.channel import Channel, ChannelConfig, Noise  # noqa: E402
from txaudio.params import Params  # noqa: E402
from txaudio.protocol import Receiver, Sender  # noqa: E402

ROOM = dict(echoes=[(9, 0.5), (23, 0.35), (41, 0.2)])
TESTS = {
    # name: (channel, noise dBFS, interfering second sender dB rel. to ours)
    "clean": (ChannelConfig(gain=0.3), None, None),
    "noise -28": (ChannelConfig(gain=0.3), -28, None),
    "noise -20": (ChannelConfig(gain=0.3), -20, None),
    "noise -15": (ChannelConfig(gain=0.3), -15, None),
    "room .4": (ChannelConfig(gain=0.3, reverb_rt60=0.4, reverb_level=0.3, **ROOM), -50, None),
    "room .8": (ChannelConfig(gain=0.3, reverb_rt60=0.8, reverb_level=0.6, **ROOM), -50, None),
    "room 1.2": (ChannelConfig(gain=0.3, reverb_rt60=1.2, reverb_level=1.0, **ROOM), -50, None),
    "drift +1000": (ChannelConfig(gain=0.3, drift_ppm=1000), -50, None),
    "drift -2000": (ChannelConfig(gain=0.3, drift_ppm=-2000), -50, None),
    "wobble 10dB": (ChannelConfig(gain=0.3, wobble_db=10, wobble_hz=0.5), -50, None),
    "other tx -10dB": (ChannelConfig(gain=0.3), -50, -10),
    "other tx -6dB": (ChannelConfig(gain=0.3), -50, -6),
    "all": (ChannelConfig(gain=0.3, reverb_rt60=0.6, reverb_level=0.4, drift_ppm=500,
                          wobble_db=6, **ROOM), -30, None),
}


def render(p, data, seed):
    s = Sender(data, p, use_feedback=False, seed=seed)
    out = []
    while not s.finished:
        out.append(s.next_chunk(4800))
    return np.concatenate(out), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--only")
    a = ap.parse_args()
    p = Params(step=a.step)
    data = os.urandom(p.payload * a.blocks)
    x, s = render(p, data, 3)
    other, _ = render(p, os.urandom(p.payload * a.blocks * 2), 99)
    print(f"step {a.step}s, {p.net_bps():.1f} bit/s net, {len(x) / p.fs:.0f} s of audio")
    only = a.only.split(",") if a.only else None
    for name, (cfg, noise_db, interf_db) in TESTS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        ch, noise = Channel(cfg), Noise(noise_db, 5)
        r = Receiver(p, send_feedback=False)
        for i in range(0, len(x), 4800):
            c = ch(x[i: i + 4800])
            c = c + noise(len(c))
            if interf_db is not None:
                o = other[i + 12345: i + 12345 + len(c)]
                c[: len(o)] += o * 10 ** (interf_db / 20) * cfg.gain
            r.feed(c)
        r.flush()
        cd = r.demod.codec
        fixed = cd.corrected / max(1, cd.payload_bytes)
        ok = r.result() == data
        print(f"{name:16s} {'ok  ' if ok else 'FAIL'} {len(r.blocks)}/{s.total} blocks, "
              f"{fixed * 100:5.2f}% bytes repaired  ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
